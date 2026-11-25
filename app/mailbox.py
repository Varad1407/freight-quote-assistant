"""Read quote requests from an email inbox (IMAP) and save draft replies.

It NEVER sends email. Drafts appear in the mailbox's Drafts folder, threaded under the
customer's message; a human reviews and clicks Send.

Usage (from the project folder, with Ollama running and IMAP_* set in .env):
    python -m app.mailbox --dry-run      # show what would be drafted, change nothing
    python -m app.mailbox                # process new emails once, save drafts
    python -m app.mailbox --watch 120    # check every 120 seconds until Ctrl+C
    python -m app.mailbox --folder demo_mail   # folder mode: files instead of an email account

Design notes:
- Messages are fetched with BODY.PEEK, so they stay unread in the inbox.
- Processed Message-IDs are remembered in .mailbox_state.json, so nothing is drafted twice.
- Emails flagged "needs_review" get a clearly marked internal block at the top of the draft
  and "[REVIEW]" in the subject, so they cannot be sent by accident without being noticed.
"""

import argparse
import email
import imaplib
import json
import logging
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from email import policy
from email.message import EmailMessage, Message
from email.utils import formatdate, make_msgid, parseaddr
from html.parser import HTMLParser
from pathlib import Path

from app.config import PROJECT_ROOT, build_llm, load_dotenv
from app.extractor import ExtractionError, LLMClient, LLMUnavailableError
from app.models import QuoteDraft, ShipmentRequest, TransportMode
from app.pipeline import process_email

logger = logging.getLogger("quote_assistant.mailbox")

STATE_FILE = PROJECT_ROOT / ".mailbox_state.json"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
REVIEW_MARKER = "=== INTERNAL: REVIEW REQUIRED. DELETE THIS BLOCK BEFORE SENDING ==="


# --- reading emails -----------------------------------------------------------

class _TextFromHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        elif tag in ("br", "p", "div", "li", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextFromHTML()
    parser.feed(html)
    lines = (line.strip() for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def get_text_body(msg: Message) -> str:
    """The readable text of an email: plain text if present, otherwise HTML converted to text.

    Attachments are ignored (reading PDF/Excel packing lists is a later feature).
    """
    plain, html = None, None
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.get_content_maintype() == "multipart" or part.get_filename():
            continue
        content_type = part.get_content_type()
        if content_type not in ("text/plain", "text/html"):
            continue
        try:
            text = part.get_content()
        except (LookupError, UnicodeDecodeError):
            payload = part.get_payload(decode=True) or b""
            text = payload.decode("utf-8", errors="replace")
        if content_type == "text/plain" and plain is None:
            plain = text
        elif content_type == "text/html" and html is None:
            html = text
    if plain and plain.strip():
        return plain.strip()
    return html_to_text(html).strip() if html else ""


def email_with_subject(msg: Message) -> str:
    """The text the pipeline sees: subject line plus body (subjects often hold the route)."""
    subject = str(msg.get("Subject", "")).strip()
    body = get_text_body(msg)
    return f"Subject: {subject}\n\n{body}" if subject else body


# --- deciding which emails to answer ---------------------------------------------
# The first real-inbox test drafted "quote replies" to Google security alerts sent from
# no-reply@accounts.google.com. Real inboxes are full of automatic mail, so we filter it.

AUTOMATED_SENDER = re.compile(
    r"^(no-?reply|do-?not-?reply|mailer-daemon|postmaster|bounces?|notifications?|alerts?)([+._-].*)?@",
    re.IGNORECASE,
)


def automated_reason(msg: Message) -> str | None:
    """Why this email looks automatic (newsletter, notification, bounce), or None if it looks personal."""
    sender = parseaddr(str(msg.get("From", "")))[1]
    if AUTOMATED_SENDER.match(sender):
        return f"automated sender ({sender})"
    auto_submitted = str(msg.get("Auto-Submitted", "")).strip().lower()
    if auto_submitted and auto_submitted != "no":
        return "marked as automatically sent"
    if str(msg.get("Precedence", "")).strip().lower() in ("bulk", "list", "junk"):
        return "bulk or mailing-list email"
    if msg.get("List-Unsubscribe") or msg.get("List-Id"):
        return "newsletter or mailing list"
    return None


def looks_like_quote_request(shipment: ShipmentRequest) -> bool:
    """At least one shipment detail found.

    Deliberately lenient: a missed customer costs money, while an unneeded draft is simply deleted
    (nothing is ever sent automatically). A vague "price to Canada?" still gets a reply.
    """
    signals = [
        shipment.origin.is_known(),
        shipment.destination.is_known(),
        bool(shipment.cargo_description),
        shipment.weight_kg is not None,
        shipment.volume_cbm is not None or bool(shipment.dimensions),
        shipment.mode != TransportMode.UNKNOWN,
    ]
    return any(signals)


# --- writing drafts -------------------------------------------------------------

def _reply_subject(original: str, needs_review: bool) -> str:
    subject = original.strip() or "Your quote request"
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"
    return f"[REVIEW] {subject}" if needs_review else subject


def build_reply(original: Message, draft: QuoteDraft, from_address: str) -> EmailMessage:
    """A reply email, threaded under the original, ready to be saved as a draft."""
    needs_review = draft.status == "needs_review"
    reply = EmailMessage()
    reply["From"] = from_address
    reply["To"] = original.get("Reply-To") or original.get("From", "")
    reply["Subject"] = _reply_subject(str(original.get("Subject", "")), needs_review)
    reply["Date"] = formatdate(localtime=True)
    reply["Message-ID"] = make_msgid()
    if original_id := original.get("Message-ID"):
        reply["In-Reply-To"] = original_id
        reply["References"] = f"{original.get('References', '')} {original_id}".strip()

    body = draft.reply_body
    if needs_review:
        notes = "\n".join(f"- {w}" for w in draft.warnings) or "- (no details)"
        body = f"{REVIEW_MARKER}\n{notes}\n{'=' * len(REVIEW_MARKER)}\n\n{body}"
    reply.set_content(body)
    return reply


# --- state ------------------------------------------------------------------------

def load_state(path: Path = STATE_FILE) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("processed", []))


def save_state(processed: set[str], path: Path = STATE_FILE) -> None:
    path.write_text(json.dumps({"processed": sorted(processed)}, indent=2), encoding="utf-8")


# --- the worker ---------------------------------------------------------------------

def _quoted(folder: str) -> str:
    """imaplib sends folder names as-is, so names with spaces must be quoted."""
    return f'"{folder}"' if " " in folder and not folder.startswith('"') else folder


def imap_date(day: date) -> str:
    """IMAP wants dates like 25-Nov-2026, independent of the computer's language settings."""
    return f"{day.day:02d}-{MONTHS[day.month - 1]}-{day.year}"


def process_inbox(
    imap,
    llm: LLMClient,
    own_address: str,
    processed: set[str],
    folder: str = "INBOX",
    drafts_folder: str = "[Gmail]/Drafts",
    since_days: int = 7,
    max_messages: int = 20,
    dry_run: bool = False,
    on_processed=None,
) -> list[dict]:
    """Draft replies for new messages. Returns one summary dict per message handled."""
    status, _ = imap.select(_quoted(folder), readonly=True)
    if status != "OK":
        raise RuntimeError(f"Could not open folder {folder!r}")
    since = imap_date(datetime.now().astimezone().date() - timedelta(days=since_days))
    status, data = imap.search(None, "SINCE", since)
    if status != "OK":
        raise RuntimeError("IMAP search failed")
    ids = data[0].split()[-max_messages:] if data and data[0] else []

    results = []
    for msg_id in ids:
        status, fetched = imap.fetch(msg_id, "(BODY.PEEK[])")  # PEEK: leave the email unread
        if status != "OK" or not fetched or not isinstance(fetched[0], tuple):
            continue
        msg = email.message_from_bytes(fetched[0][1], policy=policy.default)
        message_id = str(msg.get("Message-ID", "")).strip() or f"no-id:{msg_id.decode()}"
        sender = parseaddr(str(msg.get("From", "")))[1].lower()
        subject = str(msg.get("Subject", ""))

        if message_id in processed:
            continue
        if sender == own_address.lower():
            continue  # never reply to ourselves (our own drafts or sent mail)
        text = email_with_subject(msg)
        if len(text) < 20:
            continue

        summary = {"message_id": message_id, "from": sender, "subject": subject}

        # Automatic mail is checked before the AI runs (saves time on newsletters and alerts).
        skip_reason = automated_reason(msg)
        draft = None
        if skip_reason is None:
            try:
                draft = process_email(text, llm)
            except ExtractionError as exc:
                summary.update(status="error", error=str(exc))
                logger.warning("extraction_failed subject=%r", subject)
                results.append(summary)
                continue  # not marked processed: it will be retried next run
            if not looks_like_quote_request(draft.shipment):
                skip_reason = "no shipment details found; not a quote request"
        if skip_reason:
            summary.update(status="skipped", reason=skip_reason)
            results.append(summary)
            if not dry_run:
                processed.add(message_id)  # remembered, so it is not checked again every run
                if on_processed:
                    on_processed(processed)
            continue

        reply = build_reply(msg, draft, own_address)
        summary.update(status=draft.status, draft_subject=str(reply["Subject"]), missing=draft.missing)
        if not dry_run:
            imap.append(_quoted(drafts_folder), "\\Draft", imaplib.Time2Internaldate(time.time()), reply.as_bytes())
            processed.add(message_id)
            if on_processed:
                on_processed(processed)
        summary["draft_body"] = str(reply.get_content())
        results.append(summary)
    return results


def find_drafts_folder(imap, fallback: str = "[Gmail]/Drafts") -> str:
    """Ask the server which folder is marked \\Drafts (its name depends on the account language)."""
    status, folders = imap.list()
    if status == "OK":
        for line in folders or []:
            text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line)
            if "\\Drafts" in text:
                name = text.rsplit(' "/" ', 1)[-1].strip()
                return name.strip('"')
    return fallback


# --- folder mode: the same worker, with files instead of an email account ------------

LOCAL_ADDRESS = "assistant@local.test"


class LocalFolderMailbox:
    """Pretends to be an IMAP server, backed by two folders on disk.

    inbox/   customer emails as .eml files (e.g. dragged out of Apple Mail) or .txt files
    drafts/  draft replies written here as .eml files (double-click to open in Apple Mail)

    process_inbox() cannot tell the difference, so folder mode exercises exactly the same code
    as a real mailbox. Useful for demos and testing without any email account.
    """

    def __init__(self, root: Path):
        self.inbox = root / "inbox"
        self.drafts = root / "drafts"
        self.drafts.mkdir(parents=True, exist_ok=True)
        self._files: list[Path] = []

    def select(self, folder, readonly=False):
        self._files = sorted(p for p in self.inbox.glob("*") if p.suffix.lower() in (".eml", ".txt"))
        return "OK", [str(len(self._files)).encode()]

    def search(self, charset, *criteria):  # date filters don't apply to local files
        return "OK", [b" ".join(str(i + 1).encode() for i in range(len(self._files)))]

    def fetch(self, msg_id, parts):
        path = self._files[int(msg_id) - 1]
        return "OK", [(b"1 (BODY[])", self._as_email_bytes(path)), b")"]

    def append(self, folder, flags, date_time, message: bytes):
        subject = email.message_from_bytes(message, policy=policy.default).get("Subject", "draft")
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in str(subject))[:60].strip()
        path = self.drafts / f"{len(list(self.drafts.glob('*.eml'))) + 1:02d} {safe}.eml"
        path.write_bytes(message)
        return "OK", [b"APPEND completed"]

    def list(self):
        return "OK", [b'(\\Drafts) "/" "drafts"']

    def logout(self):
        return "OK", [b"bye"]

    @staticmethod
    def _as_email_bytes(path: Path) -> bytes:
        if path.suffix.lower() == ".eml":
            return path.read_bytes()
        # A .txt file: optional "Subject: ..." first line, the rest is the body.
        text = path.read_text(encoding="utf-8")
        subject = path.stem.replace("_", " ")
        if text.lower().startswith("subject:"):
            first, _, text = text.partition("\n")
            subject = first.split(":", 1)[1].strip()
        msg = EmailMessage()
        msg["From"] = f"Customer <customer+{path.stem}@example.com>"
        msg["To"] = LOCAL_ADDRESS
        msg["Subject"] = subject
        msg["Message-ID"] = f"<{path.stem}@local.test>"
        msg.set_content(text.strip())
        return msg.as_bytes()


def prepare_demo_folder(root: Path) -> bool:
    """Create root/inbox with the sample emails if it doesn't exist yet. Returns True if created."""
    inbox = root / "inbox"
    if inbox.exists():
        return False
    inbox.mkdir(parents=True)
    for sample in sorted((PROJECT_ROOT / "app" / "samples").glob("*.txt")):
        (inbox / sample.name).write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def connect() -> imaplib.IMAP4_SSL:
    host = os.getenv("IMAP_HOST", "imap.gmail.com")
    user, password = os.getenv("IMAP_USER"), os.getenv("IMAP_PASSWORD")
    if not user or not password:
        sys.exit("IMAP_USER and IMAP_PASSWORD must be set in .env")
    try:
        imap = imaplib.IMAP4_SSL(host, 993, timeout=30)
    except OSError as exc:
        sys.exit(f"Could not connect to {host}: {exc}. Check IMAP_HOST and your internet connection.")
    try:
        imap.login(user, password)
    except imaplib.IMAP4.error:
        sys.exit("Login failed. Check IMAP_USER and IMAP_PASSWORD (use the 16-character app password, no spaces).")
    return imap


def run_once(dry_run: bool, folder: Path | None = None) -> None:
    llm = build_llm()
    if folder is not None:
        if prepare_demo_folder(folder):
            print(f"Created {folder / 'inbox'} with the 5 sample emails. Add your own .txt or .eml files there.")
        imap, own_address, drafts_folder = LocalFolderMailbox(folder), LOCAL_ADDRESS, "drafts"
        state_path = folder / ".state.json"
    else:
        imap = connect()
        own_address = os.environ["IMAP_USER"]
        drafts_folder = os.getenv("IMAP_DRAFTS_FOLDER") or find_drafts_folder(imap)
        state_path = STATE_FILE
    processed = load_state(state_path)
    try:
        results = process_inbox(
            imap,
            llm,
            own_address=own_address,
            processed=processed,
            folder=os.getenv("IMAP_FOLDER", "INBOX"),
            drafts_folder=drafts_folder,
            since_days=int(os.getenv("IMAP_SINCE_DAYS", "7")),
            dry_run=dry_run,
            on_processed=lambda p: save_state(p, state_path),
        )
    finally:
        imap.logout()

    if not results:
        print("No new emails to process.")
    for r in results:
        print(f"\n--- {r['subject']!r} from {r['from']} -> {r['status']}")
        if r["status"] == "skipped":
            print(f"    no draft: {r['reason']}")
            continue
        if r["status"] == "error":
            print(f"    {r['error']}")
            continue
        print(f"    draft subject: {r['draft_subject']}")
        if dry_run:
            print("    draft body:\n" + "\n".join("      " + line for line in r["draft_body"].splitlines()))
    if dry_run:
        print("\n(dry run: no drafts were saved)")
    elif results:
        saved = sum(r["status"] not in ("error", "skipped") for r in results)
        where = f"the folder {folder / 'drafts'} (double-click a file to open it in Mail)" if folder else "your Drafts folder"
        print(f"\nSaved {saved} draft(s) to {where}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft replies to freight quote requests (never sends).")
    parser.add_argument("--dry-run", action="store_true", help="show what would be drafted, save nothing")
    parser.add_argument("--watch", type=int, metavar="SECONDS", help="keep checking every N seconds")
    parser.add_argument(
        "--folder", type=Path, metavar="PATH",
        help="folder mode: read emails from PATH/inbox and write drafts to PATH/drafts (no email account needed)",
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.watch:
            print(f"Checking every {args.watch}s. Press Ctrl+C to stop.")
            while True:
                run_once(args.dry_run, args.folder)
                time.sleep(args.watch)
        else:
            run_once(args.dry_run, args.folder)
    except LLMUnavailableError as exc:
        sys.exit(f"Stopped: {exc}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
