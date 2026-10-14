from chordsheet.ingest.file import ingest_file
from chordsheet.ingest.youtube import ingest_youtube, is_youtube_url

__all__ = ["ingest", "ingest_file", "ingest_youtube", "is_youtube_url"]


def ingest(source: str, cache_root=None):
    """Any source -> (Cache, path to mono 44.1 kHz WAV)."""
    kwargs = {} if cache_root is None else {"cache_root": cache_root}
    if is_youtube_url(source):
        return ingest_youtube(source, **kwargs)
    return ingest_file(source, **kwargs)
