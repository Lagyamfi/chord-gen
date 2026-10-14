from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NoteEvent:
    pitch: int  # MIDI pitch
    onset: float  # seconds
    offset: float  # seconds
    velocity: float
    instrument: str
    is_drum: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> NoteEvent:
        return cls(**d)


@dataclass(frozen=True)
class BeatGrid:
    beats: list[float]  # beat times in seconds
    downbeats: list[float]  # subset of beats that start a bar
    tempo: float  # BPM
    end: float  # end of the last beat window (seconds)
    has_downbeats: bool = False  # True when downbeats were detected, not assumed

    def windows(self) -> list[tuple[float, float]]:
        edges = [*self.beats, self.end]
        return list(zip(edges[:-1], edges[1:]))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BeatGrid:
        return cls(**d)
