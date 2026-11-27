"""Unit tests for the demo rate limiter, using a fake clock instead of sleeping."""

from app.ratelimit import RateLimiter, client_ip


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make(per_ip=3, window=60, daily=100):
    clock = FakeClock()
    return RateLimiter(per_ip_limit=per_ip, window_s=window, global_daily_limit=daily, clock=clock), clock


def test_allows_up_to_limit_then_blocks():
    limiter, _ = make(per_ip=3)
    assert [limiter.check("a").allowed for _ in range(4)] == [True, True, True, False]


def test_window_slides():
    limiter, clock = make(per_ip=2, window=60)
    limiter.check("a")
    clock.now += 30
    limiter.check("a")
    assert not limiter.check("a").allowed
    clock.now += 31  # the first request is now older than 60 s
    assert limiter.check("a").allowed


def test_retry_after_counts_down_to_oldest_request_expiring():
    limiter, clock = make(per_ip=1, window=60)
    limiter.check("a")
    clock.now += 20
    decision = limiter.check("a")
    assert not decision.allowed
    assert decision.retry_after_s == 40
    assert decision.reason == "per_ip"


def test_rejected_requests_do_not_extend_the_block():
    limiter, clock = make(per_ip=1, window=60)
    limiter.check("a")
    for _ in range(5):
        clock.now += 10
        limiter.check("a")  # blocked attempts are not recorded
    clock.now += 11
    assert limiter.check("a").allowed


def test_limits_are_per_key():
    limiter, _ = make(per_ip=1)
    assert limiter.check("a").allowed
    assert limiter.check("b").allowed
    assert not limiter.check("a").allowed


def test_global_daily_cap():
    limiter, _ = make(per_ip=100, daily=3)
    assert [limiter.check(f"ip{i}").allowed for i in range(4)] == [True, True, True, False]
    assert limiter.check("new").reason == "global"


def test_zero_disables_limits():
    limiter, _ = make(per_ip=0, daily=0)
    assert all(limiter.check("a").allowed for _ in range(50))


def test_client_ip_uses_last_forwarded_entry():
    # The visitor can fake the first entries; the proxy appends the real address last.
    assert client_ip("6.6.6.6, 203.0.113.9", "10.0.0.1") == "203.0.113.9"
    assert client_ip(None, "10.0.0.1") == "10.0.0.1"
    assert client_ip("", None) == "unknown"
