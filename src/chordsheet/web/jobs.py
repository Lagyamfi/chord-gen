"""Add a song from a link: a one-at-a-time queue that downloads and analyses in
the background while the player keeps playing.

One worker, so a second link waits its turn: MuScriptor-large already keeps the
laptop busy, and two at once would only make both slower."""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from chordsheet.cache import Cache
from chordsheet.ingest.youtube import YouTubeError, cache_key, fetch_info, video_id

log = logging.getLogger(__name__)

# stage -> what the player says; in order
STAGES = {
    "queued": "Waiting its turn",
    "info": "Looking up the video",
    "download": "Downloading",
    "transcribe": "Transcribing the notes",
    "beats": "Finding the beat",
    "chords": "Chords and sections",
    "ready": "Ready",
}
MAX_MINUTES = 15

# (url, cache_root, progress) -> the song's cache key once its sheet is written
Runner = Callable[[str, Path, Callable[[str], None]], str]


@dataclass
class Job:
    id: int
    url: str
    song: str  # cache key the song will have
    status: str = "queued"  # queued | running | done | failed
    stage: str = "queued"
    title: str = ""
    artist: str = ""
    error: str = ""
    existing: bool = False  # the link was already in the library
    started: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self) | {
            "stage_label": STAGES.get(self.stage, self.stage),
            "step": list(STAGES).index(self.stage) if self.stage in STAGES else 0,
            "steps": len(STAGES) - 1,
        }


def analyse_link(transcriber: str = "muscriptor", model_size: str = "large") -> Runner:
    """The real runner: look the video up, save its title, then the full pipeline."""

    def run(url: str, cache_root: Path, progress: Callable[[str], None]) -> str:
        from chordsheet.pipeline import analyse

        progress("info")
        info = fetch_info(url)
        if info["duration"] > MAX_MINUTES * 60:
            raise YouTubeError(
                f"This video is {info['duration'] // 60} minutes long; "
                f"songs up to {MAX_MINUTES} minutes can be added."
            )
        cache = Cache(cache_key(url), cache_root)
        cache.save_json("meta.json", info)
        progress("meta")  # lets the queue pick the title up before the long part
        analyse(
            url,
            transcriber=transcriber,
            model_size=model_size,
            cache_root=cache_root,
            progress=progress,
        )
        return cache.key

    return run


class Jobs:
    def __init__(
        self,
        cache_root: Path,
        runner: Runner | None = None,
        has_song: Callable[[str], bool] | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.runner = runner or analyse_link()
        self.has_song = has_song or (lambda key: False)
        self._ids = itertools.count(1)
        self._jobs: dict[int, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def submit(self, url: str) -> Job:
        """Queue a link. Raises ValueError for anything that isn't a YouTube video link."""
        url = url.strip()
        if not video_id(url):
            raise ValueError("That isn't a YouTube video link.")
        key = cache_key(url)
        with self._lock:
            for job in self._jobs.values():  # already on its way
                if job.song == key and job.status in ("queued", "running"):
                    return job
            job = Job(next(self._ids), url, key)
            if self.has_song(key):
                job.status, job.stage, job.existing = "done", "ready", True
                self._jobs[job.id] = job
                return job
            self._jobs[job.id] = job
        self._queue.put(job)
        self._start()
        return job

    def list(self) -> list[dict]:
        with self._lock:
            return [
                j.to_dict() for j in sorted(self._jobs.values(), key=lambda j: -j.id)
            ]

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def dismiss(self, job_id: int) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in ("done", "failed"):
                del self._jobs[job_id]

    def wait(self, timeout: float = 10.0) -> None:
        """Block until the queue is empty (tests)."""
        end = time.time() + timeout
        while time.time() < end and any(
            j.status in ("queued", "running") for j in self._jobs.values()
        ):
            time.sleep(0.01)

    def _start(self) -> None:
        with self._lock:
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._work, name="chordsheet-add", daemon=True
                )
                self._worker.start()

    def _work(self) -> None:
        while True:
            job = self._queue.get()  # the worker lives as long as the server
            job.status = "running"

            def progress(stage: str, job: Job = job) -> None:
                if stage == "meta":
                    meta = Cache(job.song, self.cache_root).load_json("meta.json")
                    job.title, job.artist = (
                        meta.get("title", ""),
                        meta.get("artist", ""),
                    )
                elif stage in STAGES:
                    job.stage = stage

            try:
                job.song = self.runner(job.url, self.cache_root, progress)
                job.status, job.stage = "done", "ready"
            except YouTubeError as e:
                job.status, job.error = "failed", str(e)
            except Exception as e:  # noqa: BLE001 - shown in the player, logged in full
                log.error("adding %s failed:\n%s", job.url, traceback.format_exc())
                job.status, job.error = (
                    "failed",
                    f"Something went wrong while {STAGES.get(job.stage, job.stage).lower()}: {e}",
                )
