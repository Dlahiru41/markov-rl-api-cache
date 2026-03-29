"""Tests for session expiry and finalization behavior."""

from datetime import datetime, timedelta, timezone

from src.pipeline.session_tracker import SessionTracker


class DummyMarkov:
    def __init__(self):
        self.transitions = []

    def update(self, a, b):
        self.transitions.append((a, b))


def test_session_expiry_finalizes_and_updates_markov(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dummy = DummyMarkov()
    events = []
    tracker = SessionTracker(config={"session_ttl_minutes": 30, "session_max_calls": 200}, markov_chain=dummy, on_session_finalized=events.append)

    tracker.track("s1", "/a", {})
    tracker.track("s1", "/b", {})

    st = tracker._sessions["s1"]
    st.last_seen = datetime.now(timezone.utc) - timedelta(minutes=31)

    tracker.track("s2", "/x", {})

    assert ("/a", "/b") in dummy.transitions
    assert any(e["session_id"] == "s1" and e["reason"] == "ttl_expired" for e in events)
