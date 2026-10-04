"""Read-only browsing of records belonging to a shared username."""
from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/accounts", tags=["Account history"])
DATABASE = Path(__file__).resolve().parents[1] / "data/singularity_canvas.sqlite3"


@contextmanager
def connection():
    db = sqlite3.connect(
        DATABASE.as_uri() + "?mode=ro", uri=True, timeout=10
    )
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only = ON")
        db.execute("BEGIN")
        yield db
    finally:
        db.close()


def account(db, username):
    row = db.execute(
        "SELECT * FROM users WHERE username = ?",
        (username.strip().lower(),),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Username not found.")
    return dict(row)


def record(row):
    result = dict(row)
    # Keep original database fields and provide decoded JSON for the UI.
    decoded = {}
    for name, value in result.items():
        if name.endswith("_json"):
            decoded[name.removesuffix("_json")] = json.loads(value)
    if decoded:
        result["decoded"] = decoded
    return result


def owned_session(db, user_id, session_id):
    row = db.execute(
        "SELECT * FROM sessions WHERE id = ? AND user_id = ?",
        (session_id, user_id),
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Session not found for this username.")
    return row


@router.get("/{username}/history")
def account_history(
    username: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    with connection() as db:
        user = account(db, username)
        counts = {
            "sessions": db.execute(
                "SELECT COUNT(*) FROM sessions WHERE user_id = ?",
                (user["id"],),
            ).fetchone()[0],
            "sampling_runs": db.execute(
                """SELECT COUNT(*) FROM sampling_runs r
                   JOIN sessions s ON s.id = r.session_id
                   WHERE s.user_id = ?""", (user["id"],),
            ).fetchone()[0],
            "explanations": db.execute(
                """SELECT COUNT(*) FROM explanations e
                   JOIN sessions s ON s.id = e.session_id
                   WHERE s.user_id = ?""", (user["id"],),
            ).fetchone()[0],
        }
        rows = db.execute(
            """SELECT s.*,
                 (SELECT COUNT(*) FROM sampling_runs r
                  WHERE r.session_id = s.id) AS run_count,
                 (SELECT COUNT(*) FROM explanations e
                  WHERE e.session_id = s.id) AS explanation_count
               FROM sessions s WHERE s.user_id = ?
               ORDER BY julianday(s.updated_at) DESC, s.rowid DESC
               LIMIT ? OFFSET ?""",
            (user["id"], limit, offset),
        ).fetchall()
        return {
            "user": user,
            "counts": counts,
            "sessions": [record(row) for row in rows],
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(rows) < counts["sessions"],
        }


@router.get("/{username}/sessions/{session_id}/history")
def session_history(
    username: str,
    session_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    with connection() as db:
        user = account(db, username)
        session = owned_session(db, user["id"], session_id)
        total = db.execute(
            "SELECT COUNT(*) FROM sampling_runs WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        rows = db.execute(
            """SELECT r.id, r.session_id, r.session_revision, r.seed,
                      r.model_name, r.model_sha256, r.backend, r.created_at,
                      (SELECT COUNT(*) FROM explanations e
                       WHERE e.run_id = r.id AND e.session_id = r.session_id)
                      AS explanation_count
               FROM sampling_runs r WHERE r.session_id = ?
               ORDER BY r.id DESC LIMIT ? OFFSET ?""",
            (session_id, limit, offset),
        ).fetchall()
        return {
            "session": record(session),
            "runs": [dict(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(rows) < total,
        }


@router.get("/{username}/runs/{run_id}/history")
def run_history(
    username: str,
    run_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    with connection() as db:
        user = account(db, username)
        run = db.execute(
            """SELECT r.* FROM sampling_runs r
               JOIN sessions s ON s.id = r.session_id
               WHERE r.id = ? AND s.user_id = ?""",
            (run_id, user["id"]),
        ).fetchone()
        if run is None:
            raise HTTPException(404, "Run not found for this username.")
        total = db.execute(
            """SELECT COUNT(*) FROM explanations
               WHERE run_id = ? AND session_id = ?""",
            (run_id, run["session_id"]),
        ).fetchone()[0]
        explanations = db.execute(
            """SELECT * FROM explanations
               WHERE run_id = ? AND session_id = ?
               ORDER BY id DESC LIMIT ? OFFSET ?""",
            (run_id, run["session_id"], limit, offset),
        ).fetchall()
        return {
            "run": record(run),
            "explanations": [record(row) for row in explanations],
            "total_explanations": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(explanations) < total,
        }


@router.get("/{username}/export")
def export_account_history(username: str):
    from datetime import datetime, timezone
    from fastapi.responses import Response

    with connection() as db:
        user = account(db, username)
        sessions = db.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY rowid",
            (user["id"],),
        ).fetchall()
        runs = db.execute(
            """SELECT r.* FROM sampling_runs r
               JOIN sessions s ON s.id = r.session_id
               WHERE s.user_id = ? ORDER BY r.id""",
            (user["id"],),
        ).fetchall()
        explanations = db.execute(
            """SELECT e.* FROM explanations e
               JOIN sessions s ON s.id = e.session_id
               WHERE s.user_id = ? ORDER BY e.id""",
            (user["id"],),
        ).fetchall()

        payload = {
            "format": "singularity-canvas-account-history",
            "schema_version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "sessions": len(sessions),
                "sampling_runs": len(runs),
                "explanations": len(explanations),
            },
            "user": user,
            "sessions": [record(row) for row in sessions],
            "sampling_runs": [record(row) for row in runs],
            "explanations": [record(row) for row in explanations],
            "notes": [
                "Original database fields are retained alongside decoded JSON.",
                "Individual clue edits between runs are not a complete timeline.",
                "Model weights and application code are not included.",
                "Reproduction requires the matching model and software environment.",
            ],
        }

    filename = (
        f"singularity-canvas-account-{user['id']}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    )
    return Response(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )
