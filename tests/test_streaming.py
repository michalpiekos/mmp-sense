"""Live streams from a receiver's relay: decoding, timing, alignment and fragments."""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.io import wavfile

from fake_relay import RECEIVER_CLOCK, FakeRelay, clock, data, event, hello, listing, msg
from mmp_sense.core.session import Session
from mmp_sense.streaming import LiveStreams, RelayError, RelayStream, cut, list_streams
from mmp_sense.streaming.relay import PCM_FORMATS, _decode, encode

RATE = 48_000
BLOCK = 480
UTC0 = 1_794_729_600_000_000_000          # 2026-11-15T08:00:00Z in ns


def wait(condition, seconds=5.0):
    end = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def tone(first: int, n: int, channels: int = 4, impulse_at: int | None = None) -> np.ndarray:
    k = np.arange(first, first + n)[:, None]
    counts = (np.sin(k * 0.01 + np.arange(channels)) * 1000).astype(np.int64)
    if impulse_at is not None and first <= impulse_at < first + n:
        counts[impulse_at - first, :] = 30000
    return counts


def node(seconds: float, *, mono0_us: int, utc_at_mono0_ns: int, impulse_at: int | None = None,
         utc_clock: int | None = 2, receiver_late_ns: int = 0, skip_block: int | None = None) -> list[bytes]:
    """A node's relayed messages: blocks of 480 frames, and once a second its
    UTC against its timeline and the receiver's arrival time."""
    out = []
    blocks = int(seconds * RATE / BLOCK)
    for b in range(blocks):
        first = b * BLOCK
        t_us = mono0_us + first * 1_000_000 // RATE
        if b != skip_block:
            out.append(data(first, t_us, tone(first, BLOCK, impulse_at=impulse_at)))
        if b % (RATE // BLOCK) == 0:
            exact = utc_at_mono0_ns + (t_us - mono0_us) * 1000
            if utc_clock is not None:
                out.append(clock((utc_clock, exact, 20_000_000), (0, t_us, 0)))
            out.append(clock((RECEIVER_CLOCK, exact + receiver_late_ns, 100_000_000), (0, t_us, 0)))
    return out


@pytest.mark.parametrize("fmt", sorted(PCM_FORMATS))
def test_every_pcm_format_decodes_and_encodes_back(fmt):
    rng = np.random.default_rng(3)
    counts = rng.uniform(-1, 1, (50, 3)) if fmt.startswith("f") else \
        rng.integers(-(2 ** (PCM_FORMATS[fmt][0] * 8 - 1)) + 1, 2 ** (PCM_FORMATS[fmt][0] * 8 - 1) - 1, (50, 3))
    raw = encode(counts, fmt)
    assert len(raw) == 50 * 3 * PCM_FORMATS[fmt][0]
    assert encode(_decode(raw, fmt, 3), fmt) == raw


def opened(item):
    """A relayed session that has had its hello, as the relay sends it first."""
    relayed = RelayStream("unused", item)
    feed(relayed, [hello(item)])
    return relayed


def test_samples_are_timed_by_the_frame_counter_and_a_lost_block_is_a_gap():
    relayed = opened(listing("a" * 32))
    source = relayed.sources()[0]
    assert source.channels == ["S", "N", "W", "E"] and source.rate == RATE
    assert source.dtype == np.float32
    for m in node(1.0, mono0_us=5_000_000, utc_at_mono0_ns=UTC0, skip_block=10):
        relayed.on_message(chr(m[0]), m[1], int.from_bytes(m[2:4], "little"), m[8:])
    times, values = source.read()
    assert len(times) == RATE - BLOCK and source.pcm.lost_frames == BLOCK
    assert times[0] == pytest.approx(5.0)
    gap = np.argmax(np.diff(times))
    assert np.diff(times)[gap] == pytest.approx((BLOCK + 1) / RATE)
    assert times[gap + 1] == pytest.approx(5.0 + 11 * BLOCK / RATE)
    assert np.array_equal(source.counts(values[:BLOCK]), tone(0, BLOCK))
    utc = relayed.utc_offset()
    assert utc.source == "ntp" and utc.uncertainty_s == pytest.approx(0.02)
    assert utc.offset == pytest.approx(UTC0 / 1e9 - 5.0)


def test_without_its_own_utc_a_stream_is_timed_by_the_receiver():
    relayed = opened(listing("b" * 32, utc=False))
    for m in node(1.0, mono0_us=0, utc_at_mono0_ns=UTC0, utc_clock=None, receiver_late_ns=40_000_000):
        relayed.on_message(chr(m[0]), m[1], int.from_bytes(m[2:4], "little"), m[8:])
    utc = relayed.utc_offset()
    assert utc.source == "receiver" and utc.uncertainty_s == pytest.approx(0.1)
    assert utc.offset == pytest.approx(UTC0 / 1e9 + 0.04)


def test_nothing_is_handed_over_before_utc_is_known_unless_it_never_comes():
    relayed = opened(listing("c" * 32, utc=False))
    source = relayed.sources()[0]
    for b in range(100):                               # 1 s, no observation at all
        relayed.on_message("D", 0, 1, data(b * BLOCK, b * 10_000, tone(b * BLOCK, BLOCK))[8:])
    assert len(source.read()[0]) == 0
    for b in range(100, 400):
        relayed.on_message("D", 0, 1, data(b * BLOCK, b * 10_000, tone(b * BLOCK, BLOCK))[8:])
    assert len(source.read()[0]) == 400 * BLOCK, "after 3 s it is shown anyway"


def test_the_end_and_losses_are_told():
    relayed = RelayStream("unused", listing("d" * 32))
    relayed.on_message("N", 0, 0, event(0, {"kind": "relay_lost", "messages": 7})[8:])
    assert relayed.relay_lost == 7 and relayed.state == "connecting"
    assert relayed.on_message("N", 0, 0, event(0, {"kind": "stream_end", "end": "link_lost"})[8:]) is False
    assert relayed.on_message("X", 0, 0, b"") is True
    assert (relayed.state, relayed.end) == ("ended", "link_lost")


def test_a_scaled_stream_keeps_its_counts():
    relayed = opened(listing("e" * 32, channels=("x", "y", "z"), scale=0.002, unit="m/s^2", rate=400))
    source = relayed.sources()[0]
    assert source.units == "m/s^2" and source.dtype == np.float64
    counts = np.array([[1, -2, 300]] * 4)
    relayed.on_message("C", 0, 0, clock((2, UTC0, 1000), (0, 0, 0))[8:])
    relayed.on_message("D", 0, 1, data(0, 0, counts)[8:])
    _, values = source.read()
    assert np.allclose(values, counts * 0.002) and np.array_equal(source.counts(values), counts)


def test_listing_and_an_unreachable_relay():
    with FakeRelay() as relay:
        relay.add(listing("f" * 32))
        assert [s["session_id"] for s in list_streams(relay.address)] == ["f" * 32]
    with pytest.raises(RelayError, match="cannot be reached"):
        list_streams("127.0.0.1:9", timeout=1)


def two_nodes(relay):
    """Node a, timed by its NTP UTC; node b, 0.25 s later in UTC, on another
    timeline and timed by the receiver only. Both hear an impulse at UTC0 + 1 s."""
    a, b = relay.add(listing("a" * 32)), relay.add(listing("b" * 32, utc=False))
    relay.push(a["session_id"], *node(2.0, mono0_us=10_000_000, utc_at_mono0_ns=UTC0, impulse_at=RATE))
    relay.push(b["session_id"], *node(2.0, mono0_us=500_000_000, utc_at_mono0_ns=UTC0 + 250_000_000,
                                              impulse_at=int(0.75 * RATE), utc_clock=None,
                                              receiver_late_ns=30_000_000))
    return [a, b]


def held(session, name):
    return len(session.streams[name].buffer)


def test_two_nodes_land_on_one_utc_axis(tmp_path):
    with FakeRelay() as relay:
        listings = two_nodes(relay)
        session = Session()
        live = LiveStreams(session)
        names = live.open(relay.address, listings, history_s=30)
        assert names == ["tetrahedral-aaaa", "tetrahedral-bbbb"]
        assert list(session.streams) == names and session.buffer_s == 30
        wait(lambda: session.poll() is not None and all(held(session, n) == 2 * RATE for n in names))
        assert live.align()
        assert live.epoch == UTC0 / 1e9
        hits = []
        for name in names:
            t, v = session.streams[name].buffer.snapshot()
            hits.append(float(t[np.argmax(np.abs(v[:, 0]))]))
        assert hits[0] == pytest.approx(1.0, abs=1e-6), "node a by its own UTC"
        assert hits[1] == pytest.approx(1.0 + 0.03, abs=1e-3), "node b by the receiver, 30 ms late"

        live.set_trim(names[1], -0.03)
        t, v = session.streams[names[1]].buffer.snapshot()
        assert float(t[np.argmax(np.abs(v[:, 0]))]) == pytest.approx(1.0, abs=1e-3)

        fragments = cut(live, 0.5, 1.5, tmp_path)
        assert [f.path.name for f in fragments] == ["tetrahedral-aaaa-20261115T080000.wav",
                                                    "tetrahedral-bbbb-20261115T080000.wav"]
        a = fragments[0]
        rate, frames = wavfile.read(a.path)
        assert rate == RATE and frames.dtype == np.int16 and abs(len(frames) - RATE) <= 1
        starts = [k for k in (RATE // 2 - 1, RATE // 2, RATE // 2 + 1)
                  if np.array_equal(frames, tone(k, len(frames), impulse_at=RATE))]
        assert len(starts) == 1, "the node's own samples, unchanged"
        assert a.started_at_utc.startswith("2026-11-15T08:00:00.50")
        assert a.meta["channel_names"] == ["S", "N", "W", "E"] and a.meta["sample_format"] == "s16"
        assert a.meta["clock"] == {"source": "ntp", "uncertainty_ns": 20_000_000, "from": "node",
                                   "trim_s": 0.0, "by_hand": False}
        assert fragments[1].meta["clock"]["from"] == "receiver"
        assert fragments[1].meta["clock"]["trim_s"] == -0.03

        live.close(names[1])
        assert list(session.streams) == names[:1]
        live.close_all()


def test_an_ended_stream_keeps_what_it_held():
    with FakeRelay() as relay:
        item = relay.add(listing("a" * 32))
        relay.push(item["session_id"], *node(1.0, mono0_us=0, utc_at_mono0_ns=UTC0),
                   event(990_000, {"kind": "stream_end", "end": "clean"}), msg("X"))
        session = Session()
        live = LiveStreams(session)
        (name,) = live.open(relay.address, [item])
        source = live.streams[name].source
        wait(lambda: source.state == "ended")
        wait(lambda: session.poll() is not None and held(session, name) == RATE)
        assert source.relayed.end == "clean"


def feed(relayed, messages):
    for m in messages:
        relayed.on_message(chr(m[0]), m[1], int.from_bytes(m[2:4], "little"), m[8:])


def test_bundles_repeats_and_overlaps_are_taken_once():
    relayed = opened(listing("g" * 32))
    source = relayed.sources()[0]
    a, b = data(0, 0, tone(0, BLOCK)), data(BLOCK, 10_000, tone(BLOCK, BLOCK))
    feed(relayed, [clock((2, UTC0, 1000), (0, 0, 0)), msg("B", a + b), a,
                   data(BLOCK // 2, 5_000, tone(BLOCK // 2, BLOCK * 2))])
    times, values = source.read()
    assert len(times) == 2 * BLOCK + BLOCK // 2 and source.pcm.lost_frames == 0
    assert np.array_equal(source.counts(values), tone(0, 2 * BLOCK + BLOCK // 2))
    assert np.all(np.diff(times) > 0)


def test_a_sample_clock_off_its_nominal_rate_does_not_drift_from_the_timeline():
    relayed = opened(listing("h" * 32))
    source = relayed.sources()[0]
    true_rate = RATE * (1 + 100e-6)                    # 100 ppm fast
    blocks = []
    for b in range(2000):                              # 20 s
        first = b * BLOCK
        blocks.append(data(first, int(round(first * 1e6 / true_rate)), tone(first, BLOCK)))
    feed(relayed, [clock((2, UTC0, 1000), (0, 0, 0)), *blocks])
    times, _ = source.read()
    last_frame = 2000 * BLOCK - 1
    assert times[-1] == pytest.approx(last_frame / true_rate, abs=50e-6), "nominal would be 2 ms off"
    assert np.all(np.diff(times) > 0)


def test_a_timeline_that_is_utc_is_taken_as_utc():
    item = listing("i" * 32, utc=False)
    item["clocks"][0] = {"id": 0, "name": "utc", "kind": "utc", "hz": 1_000_000_000, "timeline": True,
                         "source": "gnss_pps", "domain": "gnss"}
    utc = opened(item).utc_offset()
    assert (utc.offset, utc.source, utc.domain) == (0.0, "gnss_pps", "gnss")


def test_the_surest_of_several_node_clocks_is_used():
    item = listing("j" * 32)
    item["clocks"].append({"id": 3, "name": "gps", "kind": "utc", "hz": 1_000_000_000, "source": "gnss"})
    relayed = opened(item)
    feed(relayed, [clock((2, UTC0 + 5_000_000, 20_000_000), (3, UTC0, 1_000), (0, 0, 0)),
                   clock((2, UTC0 + 7_000_000, 20_000_000), (0, 0, 0))])
    utc = relayed.utc_offset()
    assert (utc.source, utc.offset, utc.uncertainty_s) == ("gnss", UTC0 / 1e9, pytest.approx(1e-6))


def test_a_watch_of_a_closed_stream_ends_with_why():
    with FakeRelay() as relay:
        item = listing("k" * 32)
        relayed = RelayStream(relay.address, item)
        source = relayed.sources()[0].start()
        wait(lambda: relayed.state == "ended")
        assert relayed.end.startswith("404")
        source.stop()
    with pytest.raises(ValueError, match="no timeline"):
        RelayStream("unused", {**listing("l" * 32), "clocks": []})


def test_stop_and_start_intake_take_up_the_stream_again_with_one_reader():
    import threading

    with FakeRelay() as relay:
        item = relay.add(listing("m" * 32))
        relay.push(item["session_id"], *node(1.0, mono0_us=0, utc_at_mono0_ns=UTC0))
        session = Session()
        live = LiveStreams(session)
        (name,) = live.open(relay.address, [item])
        relayed = live.streams[name].relayed
        wait(lambda: session.poll() is not None and len(session.streams[name].buffer) == RATE)
        session.stop_intake()
        assert relayed.state == "stopped"
        session.start_intake()
        wait(lambda: relayed.state == "live")
        relay.push(item["session_id"], *node(2.0, mono0_us=0, utc_at_mono0_ns=UTC0)[-150:])
        wait(lambda: session.poll() is not None and len(session.streams[name].buffer) > 0)
        time.sleep(0.2)
        assert relayed.end == "" and relayed.state == "live"
        assert sum(t.name.startswith("relay-") and t.is_alive() for t in threading.enumerate()) == 1
        assert live.streams[name].source.pcm.lost_frames == 0, "the stretch stopped is not lost"
        live.close_all()


def test_a_correction_made_by_hand_is_kept_and_a_trim_takes_over_again():
    with FakeRelay() as relay:
        item = relay.add(listing("n" * 32))
        relay.push(item["session_id"], *node(1.0, mono0_us=0, utc_at_mono0_ns=UTC0))
        session = Session()
        live = LiveStreams(session)
        (name,) = live.open(relay.address, [item])
        wait(lambda: session.poll() is not None and len(session.streams[name].buffer) == RATE)
        live.align()
        session.set_clock(name, 0.123)
        assert not live.align() and live.streams[name].by_hand
        assert session.streams[name].buffer.clock.offset == 0.123
        live.set_trim(name, 0.01)
        assert not live.streams[name].by_hand
        assert session.streams[name].buffer.clock.offset == pytest.approx(live.epoch - UTC0 / 1e9 - 0.01)
        live.close(name)
        session.start_intake()
        assert live.streams == {} and session.streams[name].source.state != "live", "closed stays closed"


def test_a_fragment_across_a_lost_block_is_split_there(tmp_path):
    with FakeRelay() as relay:
        item = relay.add(listing("o" * 32))
        relay.push(item["session_id"], *node(1.0, mono0_us=0, utc_at_mono0_ns=UTC0, skip_block=50))
        session = Session()
        live = LiveStreams(session)
        (name,) = live.open(relay.address, [item])
        wait(lambda: session.poll() is not None and len(session.streams[name].buffer) == RATE - BLOCK)
        live.align()
        parts = cut(live, 0.0, 1.0, tmp_path)
        assert [round(p.duration_s * RATE) for p in parts] == [50 * BLOCK, RATE - 51 * BLOCK]
        second = wavfile.read(parts[1].path)[1]
        assert np.array_equal(second, tone(51 * BLOCK, RATE - 51 * BLOCK))
        assert parts[1].started_at_utc.startswith("2026-11-15T08:00:00.51")
        live.close_all()


def test_jittered_block_times_do_not_leave_the_samples_late():
    relayed = opened(listing("p" * 32))
    source = relayed.sources()[0]
    rng = np.random.default_rng(5)
    blocks = []
    for b in range(6000):                              # 60 s, each block's time ±1 ms off
        first = b * BLOCK
        blocks.append(data(first, int(round(first * 1e6 / RATE + rng.uniform(-1000, 1000))), tone(first, BLOCK)))
    feed(relayed, [clock((2, UTC0, 1000), (0, 0, 0)), *blocks])
    times, _ = source.read()
    frames = np.arange(len(times))
    error = times[-RATE:] - frames[-RATE:] / RATE
    assert abs(float(np.mean(error))) < 100e-6, "no lasting bias"
    assert np.all(np.diff(times) > 0.7 / RATE), "never overlapping, never jumping like a loss"
    assert np.all(np.diff(times) < 1.3 / RATE)
