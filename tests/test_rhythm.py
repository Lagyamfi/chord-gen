from chordsheet.guitar.rhythm import bar_starts, regular_pulse


def counts(pulse):
    return [(p["n"], p["of"]) for p in pulse]


def test_pickup_beats_count_as_end_of_bar():
    beats = [i * 0.5 for i in range(10)]
    pulse = regular_pulse(beats, [1.0, 3.0, 5.0 - 0.5 * 2])  # downbeats at beats 2, 6, 6
    assert counts(pulse)[:6] == [(3, 4), (4, 4), (1, 4), (2, 4), (3, 4), (4, 4)]


def test_stray_downbeat_does_not_break_the_count():
    beats = [i * 0.5 for i in range(24)]
    downbeats = [0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]  # 3.0 is a glitch (would make 2-beat bars)
    pulse = regular_pulse(beats, downbeats)
    assert all(p["of"] == 4 for p in pulse)
    assert bar_starts(pulse) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_phase_moves_when_two_downbeats_agree():
    beats = [i * 0.5 for i in range(20)]
    # bars of 4 from beat 0, then a 2-beat bar and a new phase from beat 10
    pulse = regular_pulse(beats, [0.0, 2.0, 4.0, 5.0, 7.0, 9.0])
    assert pulse[10]["n"] == 1 and pulse[14]["n"] == 1


def test_three_four_is_kept():
    beats = [i * 0.5 for i in range(12)]
    pulse = regular_pulse(beats, [0.0, 1.5, 3.0, 4.5])
    assert counts(pulse)[:4] == [(1, 3), (2, 3), (3, 3), (1, 3)]


def test_half_time_tracking_is_doubled_to_four():
    beats = [i * 1.0 for i in range(8)]  # tracker followed half notes
    pulse = regular_pulse(beats, [0.0, 2.0, 4.0, 6.0])
    assert len(pulse) == 15 and all(p["of"] == 4 for p in pulse)
    assert bar_starts(pulse) == [0.0, 2.0, 4.0, 6.0]


def test_no_downbeats():
    assert counts(regular_pulse([0.0, 0.5], [])) == [(1, 1), (1, 1)]
