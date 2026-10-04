import json
import socket
from threading import Lock
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend import storage
from backend.session_api import checked

MODEL = "qwen3.8:27b"
GENERATION_LOCK = Lock()

SYSTEM = """You are the local Singularity Canvas guide.
Treat the supplied experiment snapshot as the factual source.
The canvas has 256 visible binary cells: green=1, purple=0.
The current diagnostic model learned 16 symbols and has 16 active hidden units.
The backend enumerates 65,536 hidden configurations, computes conditional
weights given the clues, and uses Torx to sample hidden states and pixels.
Execution is a software simulation through JAX on the Mac CPU.
There is no physical thermodynamic hardware connected.
Playback replays eight saved draws; it does not generate new samples.
Clue-consistent samples are not necessarily recognizable or correct symbols.
No speed or energy advantage, generalization, or exhaustive image search
has been established. Do not infer image identities from summary statistics.
You explain; you cannot change clues, generate canvas samples, or operate tools.
Answer the user's question in at most three concise sentences.
If the facts cannot answer it, say what remains unknown.
"""


class GuideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(strict=True, ge=0)
    question: str = Field(
        default="Explain what this experiment is showing.",
        min_length=1,
        max_length=1000,
    )


def install(app):
    with storage.connection() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS explanations (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                session_revision INTEGER NOT NULL,
                run_id INTEGER NOT NULL REFERENCES sampling_runs(id),
                model TEXT NOT NULL,
                question TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                answer TEXT NOT NULL,
                elapsed_ms REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

    @app.post("/api/sessions/{session_id}/explain")
    def explain(session_id: str, request: GuideRequest):
        question = request.question.strip()
        if not question:
            raise HTTPException(422, "Enter a question.")
        current = checked(storage.get_session, session_id)
        if current["revision"] != request.expected_revision:
            raise HTTPException(409, "Session changed; reload before explaining.")
        batch = current["latest_batch"]
        if batch is None:
            raise HTTPException(409, "Save a sample batch before requesting an explanation.")

        clues = current["clues"]
        samples = batch["samples"]
        snapshot = {
            "session_id": session_id,
            "session_revision": current["revision"],
            "run_id": batch["run_id"],
            "locked_cells": len(clues),
            "green_clues": sum(c["value"] == 1 for c in clues),
            "purple_clues": sum(c["value"] == 0 for c in clues),
            "free_cells": 256 - len(clues),
            "sample_count": len(samples),
            "distinct_samples_in_this_batch": len(set(tuple(s) for s in samples)),
            "all_samples_preserve_clues": all(
                s[c["index"]] == c["value"] for s in samples for c in clues
            ),
            "seed": batch["seed"],
            "sampling_compute_ms": batch["compute_ms"],
            "sampling_timing_scope": batch["timing_scope"],
            "model_sha256": batch["model_sha256"],
            "engine": batch["engine"],
        }
        payload = {
            "model": MODEL,
            "stream": False,
            "think": False,
            "system": SYSTEM,
            "prompt": "Experiment snapshot:\n" + json.dumps(snapshot)
                + "\nUser question:\n" + question,
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
                "num_predict": 256,
            },
        }

        if not GENERATION_LOCK.acquire(blocking=False):
            raise HTTPException(429, "The local guide is busy. Try again shortly.")
        try:
            started = time.perf_counter()
            ollama_request = Request(
                "http://127.0.0.1:11434/api/generate",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(ollama_request, timeout=180) as response:
                    result = json.load(response)
            except (TimeoutError, socket.timeout):
                raise HTTPException(504, "The local guide timed out.") from None
            except (HTTPError, URLError, ValueError):
                raise HTTPException(
                    502, "Could not obtain an explanation from local Ollama."
                ) from None
            answer = result.get("response", "").strip()
            if not answer or result.get("done_reason") != "stop":
                raise HTTPException(502, "The guide did not return a complete answer.")
            elapsed = (time.perf_counter() - started) * 1000

            with storage.connection() as db:
                cursor = db.execute(
                    """
                    INSERT INTO explanations(
                        session_id, session_revision, run_id, model,
                        question, snapshot_json, answer, elapsed_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id, current["revision"], batch["run_id"],
                        MODEL, question, json.dumps(snapshot), answer, elapsed,
                    ),
                )
                explanation_id = cursor.lastrowid

            return {
                "id": explanation_id,
                "answer": answer,
                "model": MODEL,
                "session_id": session_id,
                "session_revision": current["revision"],
                "run_id": batch["run_id"],
                "elapsed_ms": elapsed,
                "timing_scope": "Local AI request including model loading; separate from sampling.",
                "snapshot": snapshot,
                "label": "AI-generated explanation of a saved experiment; may contain errors.",
            }
        finally:
            GENERATION_LOCK.release()
