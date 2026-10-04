"""Fresh-install persistence checks; no model or Ollama downloads needed."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from backend import storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "data" / "canvas.sqlite3"
        self.patch = patch.object(storage, "DATABASE", self.database)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        storage.initialize_database()

    def test_fresh_install_and_restart_preserve_session(self):
        session = storage.open_session("FirstSpark")
        saved = storage.save_clues(session["id"], 0, [{"index": 17, "value": 0}])
        storage.initialize_database()
        resumed = storage.open_session("firstspark")
        self.assertEqual(resumed, saved)
        with storage.connection() as db:
            names = {r["name"] for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            self.assertEqual(names, {"users", "sessions", "sampling_runs", "explanations"})
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_stale_write_does_not_replace_clues(self):
        session = storage.open_session("demo")
        saved = storage.save_clues(session["id"], 0, [{"index": 3, "value": 1}])
        with self.assertRaises(storage.SessionConflict):
            storage.save_clues(session["id"], 0, [])
        self.assertEqual(storage.get_session(session["id"]), saved)


if __name__ == "__main__":
    unittest.main()
