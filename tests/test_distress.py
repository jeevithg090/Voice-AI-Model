import pytest

from src.realtime.distress import DistressTracker


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


def test_requires_min_hits_within_window(clock):
    t = DistressTracker(window_s=5, min_hits=3, threshold=0.7, clock=clock)
    assert t.observe(0.9) is False
    clock.now = 1
    assert t.observe(0.2) is False  # below threshold, ignored
    clock.now = 2
    assert t.observe(0.8) is False
    assert t.hit_count == 2
    clock.now = 3
    assert t.observe(0.95) is True
    assert t.is_sustained()
    assert t.hit_count == 0  # events cleared after a trigger


def test_old_hits_expire(clock):
    t = DistressTracker(window_s=5, min_hits=3, threshold=0.7, clock=clock)
    t.observe(0.9)
    clock.now = 1
    t.observe(0.9)
    clock.now = 7  # first two hits are now outside the window
    assert t.observe(0.9) is False
    assert t.hit_count == 1
    assert not t.is_sustained()


def test_threshold_is_strict(clock):
    t = DistressTracker(window_s=5, min_hits=1, threshold=0.7, clock=clock)
    assert t.observe(0.7) is False
    assert t.observe(0.71) is True


def test_latch_survives_until_consumed_or_expired(clock):
    t = DistressTracker(window_s=5, min_hits=1, threshold=0.5, latch_s=10, clock=clock)
    t.observe(0.9)
    clock.now = 4  # e.g. ASR finishes a few seconds after the spike
    assert t.is_sustained()
    assert t.consume() is True
    assert t.consume() is False

    t.observe(0.9)
    clock.now += 11
    assert t.is_sustained() is False


def test_reset_and_validation(clock):
    t = DistressTracker(min_hits=1, threshold=0.1, clock=clock)
    t.observe(0.9)
    t.reset()
    assert not t.is_sustained()
    with pytest.raises(ValueError):
        DistressTracker(min_hits=0)
