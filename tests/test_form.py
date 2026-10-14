from chordsheet.guitar.form import bar_slots, form_text, song_form


def bar(*chords, start=0.0, beat=0.5):
    """One bar of four beats; chords given per beat (mir_eval labels)."""
    return [
        {"start": start + i * beat, "end": start + (i + 1) * beat, "mir_eval": c, "chord": c}
        for i, c in enumerate(chords)
    ]


def song(*bar_chords):
    out, t = [], 0.0
    for chords in bar_chords:
        if isinstance(chords, str):
            chords = (chords,) * 4
        out.append(bar(*chords, start=t))
        t += 2.0
    return out


A = ["G:maj", "G:maj", "B:min", "C:maj"]
B = ["E:min", "C:maj", "G:maj", "D:maj"]


def test_bar_slots_sample_each_quarter():
    assert bar_slots(bar("G:maj", "G:maj", "C:maj", "D:maj")) == ("G:maj", "G:maj", "C:maj", "D:maj")


def test_verse_chorus_song_folds_to_two_lines():
    s = song(*A, *A, *B, *B, *A, *B, *B)
    form = song_form(s)
    assert [l["label"] for l in form["letters"]] == ["A", "B"]
    assert form_text(form) == "A ×2 · B ×2 · A · B ×2"


def test_noisy_repeat_still_groups_and_typical_version_wins():
    noisy = list(A)
    noisy[2] = ("B:min", "B:min", "D:maj", "B:min")  # one quarter misread
    s = song(*A, *noisy, *A)
    form = song_form(s)
    assert len(form["letters"]) == 1 and form["letters"][0]["count"] == 3
    assert form["letters"][0]["rep"] in (0, 8)  # a clean repeat, not the noisy one


def test_offset_skips_a_pickup_bar():
    s = song("N", *A, *A, *A)  # one bar of silence before the 4-bar lines
    form = song_form(s)
    assert form["lines"][0] == {"label": "-", "start": 0, "length": 1}
    assert [l["label"] for l in form["lines"][1:]] == ["A", "A", "A"]


def test_one_bar_different_is_a_variant():
    variant = A[:3] + ["D:maj"]  # last bar differs: a whole bar = 25% of the line
    form = song_form(song(*A, *B, *variant))
    assert [(l["label"], l["variant_of"]) for l in form["letters"]] == [("A", None), ("B", None), ("A′", "A")]
    assert form_text(form) == "A · B · A′"
