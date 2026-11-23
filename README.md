# Freight Quote Assistant

An AI assistant for freight forwarders. It reads a customer's quote-request email, extracts the shipment details into structured data, checks what is missing, flags risks for the team, and drafts the reply.

**The pitch to a forwarder:** "Your team replies to quote requests in minutes instead of hours, and never forgets to ask for the weight or the ready date."

## What it does

| Input | Output |
|---|---|
| A messy email: "need to send machine parts from pune to germany, 8 boxes, quite heavy, asap" | Status (`ready_to_quote` / `needs_info`), extracted fields (route, mode, cargo, weight in kg, volume, dates, Incoterm), missing-info list, internal flags (dangerous goods, reefer, no Incoterm), and a draft reply asking exactly for what is missing |

It never invents a price. Rates are set by a human, so the tool can't make costly promises.

## Run it on your Mac

```bash
cd freight-quote-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env          # default uses free local Ollama; no key needed
set -a; source .env; set +a

uvicorn app.main:app --reload
```

Open http://localhost:8000. Pick a sample email from the dropdown, then click **Process email**.
API docs are generated automatically at http://localhost:8000/docs.

## Choosing the AI model

| `LLM_PROVIDER` | Cost | Needs | Notes |
|---|---|---|---|
| `ollama` (default) | Free | Ollama running, `ollama pull qwen2.5:7b` | Runs on your Mac. Good for development; less accurate than large cloud models. |
| `anthropic` | Pay per use | `ANTHROPIC_API_KEY` | More accurate. Use for demos to clients. |

## Run with Docker

```bash
docker build -t freight-quote-assistant .
docker run --rm -p 8000:8000 --env-file .env -e OLLAMA_URL=http://host.docker.internal:11434 freight-quote-assistant
```

Inside Docker, `localhost` means the container itself, so `host.docker.internal` is used to reach Ollama on your Mac.

## Tests

```bash
pytest -q        # 73 tests, no API key needed (the LLM is replaced by a fake)
ruff check .     # linting
```

## How it works

```
email ──► FastAPI /api/quote-requests
            │
            ▼
        safety.py   remove lines that look like instructions to an AI
            ▼
        extractor.py ── LLM (Ollama or Claude) ──► JSON with RAW numbers + units
            │   parse + validate with Pydantic
            │   invalid? retry with the exact error as feedback (max 3)
            ▼
        grounding.py  every number must appear in the email, invented ones are dropped
            ▼
        normalize.py  unit conversion + multiplication in Python, no AI
            ▼
        safety.py   drop suspicious notes; any hit → status "needs_review"
            ▼
        models.py   missing_fields()   business rules, no AI
            ▼
        reply.py    build_warnings() + draft_reply()   templates, no AI
            ▼
        QuoteDraft (JSON) ──► demo page / your integration
```

| File | Role |
|---|---|
| `app/models.py` | Pydantic schemas and the "what is required to quote" rules |
| `app/extractor.py` | Prompt, LLM clients (Ollama, Anthropic), JSON parsing, validation, retry loop |
| `app/normalize.py` | Converts lb/t/cuft/inches and multiplies per-piece values, in Python |
| `app/grounding.py` | Drops any number the LLM returned that is not in the email text |
| `app/safety.py` | Prompt-injection defences: line removal, suspicious-notes check |
| `app/reply.py` | Internal warnings and customer reply templates |
| `app/main.py` | FastAPI app, dependency injection, error mapping (502/503/422) |
| `app/static/index.html` | One-page demo UI for client calls |
| `app/samples/` | 5 realistic test emails, including a prompt-injection attempt |
| `tests/` | Unit and API tests with a fake LLM |

## Design decisions (be ready to explain these in interviews and sales calls)

0. **The LLM reads; Python calculates.** Testing showed both qwen2.5:7b and 14b got per-piece weight and volume wrong (e.g. 8,148 kg instead of 3,538 kg). The LLM now copies numbers and units exactly as written, and `normalize.py` does the math. A bigger model did not fix this, and it was *more* vulnerable to prompt injection, because it follows instructions better.

0b. **Grounding check.** qwen2.5:7b once returned a volume that appeared nowhere in the email (shown as 40,776 cbm for 12 pallets). Every number must now literally appear in the email; otherwise it is dropped, recomputed from grounded numbers where possible, and the email is flagged for review.
1. **The LLM only does what LLMs are good at: reading messy text.** Missing-field checks, warnings and the reply are plain Python. That makes them predictable, testable and free to run.
2. **Never trust LLM output.** Every response is validated against a strict schema (`extra="forbid"`, ranges on weight and volume). Invalid output is retried with the specific error message, a "self-correction" loop.
3. **The LLM sits behind an interface (`LLMClient`).** Tests use a fake, and switching providers later means writing one small class.
4. **The email is untrusted input (defence in depth).** Lines that look like instructions to an AI are removed before the LLM sees them; the email is wrapped in `<email>` tags; suspicious notes are dropped afterwards; notes never go into the customer reply; and any hit sets status `needs_review` so a human checks it. Pattern matching only catches obvious attacks, which is why a human stays in the loop.
5. **Clear failure modes.** Bad input → 422, LLM keeps returning garbage → 502, provider down or key missing → 503. Logs record attempts and latency but never the customer's email content.
6. **The UI builds HTML with `textContent` only**, so a malicious email can't inject scripts into the page.

## Verification status

Verified in the build environment:
- All 73 tests pass; `ruff` passes.
- The server starts; `/health`, `/`, `/api/samples` respond; a missing API key returns a clear 503.

**Not yet verified, so do this first:**
- A real call to Ollama or the Anthropic API (neither was available while building). Run all 5 samples and check the extracted fields by hand.
- `docker build` (no Docker daemon was available). Build it on your Mac.
- The default model name `claude-sonnet-5-5`. If the API rejects it, set `ANTHROPIC_MODEL` to a current model from the Anthropic docs.

## Next steps

1. **Evaluation set:** write 30+ emails with the correct answers and measure field-level accuracy. "95% field accuracy on 30 real-style emails" is a sales line *and* a portfolio line.
2. **Deploy** to a public URL (Render, Railway or Fly.io) so prospects can try it from a link.
3. **Record a 2-minute demo video** using samples 02 and 03.
4. **Gmail/Outlook integration:** read the inbox, save the draft reply automatically. This is what clients will pay a retainer for.
5. **Rate lookup:** connect the forwarder's rate sheet (Excel) so the draft includes an indicative price for a human to approve.
