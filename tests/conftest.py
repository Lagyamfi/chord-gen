import shutil
from pathlib import Path

import pytest

from chordsheet.evaluation.synthetic import find_soundfont

GENERATED = Path(__file__).parent / "fixtures" / "generated"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    if shutil.which("fluidsynth") is None:
        pytest.skip("fluidsynth not installed")
    try:
        find_soundfont()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    return GENERATED
