"""One content identity for generation, recovery, freezing and split auditing."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from data_tools.teacher import canonical_bytes, sha256


def content_fingerprint(episode):
    entries = [{key: value for key, value in entry.items() if key != "id"} for entry in episode["entries"]]
    entries.sort(key=canonical_bytes)
    return sha256(canonical_bytes({"context": episode["context"], "entries": entries}))


class ContentRegistry:
    """Atomically reserve accepted content across the Train/Dev worker processes.

    Reservations contain hashes and opaque episode IDs, never labels or text.
    Existing accepted slots keep their reservation, including frozen pilot rows.
    A crashed, unfinished slot may replace its own reservation upon regeneration.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=60) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS content (episode_id TEXT PRIMARY KEY, digest TEXT UNIQUE NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS exclusions (digest TEXT PRIMARY KEY, reason TEXT NOT NULL)")

    def exclude(self, digest, reason):
        with sqlite3.connect(self.path, timeout=60) as connection:
            connection.execute("INSERT OR REPLACE INTO exclusions(digest,reason) VALUES(?,?)", (digest, reason))

    def claim(self, episode):
        digest = content_fingerprint(episode)
        with sqlite3.connect(self.path, timeout=60) as connection:
            connection.execute("BEGIN IMMEDIATE")
            excluded = connection.execute("SELECT reason FROM exclusions WHERE digest=?", (digest,)).fetchone()
            if excluded:
                return {"excluded_content_sha256": digest, "reason": excluded[0]}
            owner = connection.execute("SELECT episode_id FROM content WHERE digest=?", (digest,)).fetchone()
            if owner is not None and owner[0] != episode["id"]:
                return {"owner_id": owner[0], "content_sha256": digest}
            connection.execute("INSERT INTO content(episode_id,digest) VALUES(?,?) ON CONFLICT(episode_id) DO UPDATE SET digest=excluded.digest", (episode["id"], digest))
        return None
