import json

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from backend import storage


class OpenSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=80)
    new: bool = Field(default=False, strict=True)


class StoredClue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(strict=True, ge=0, le=255)
    value: int = Field(strict=True, ge=0, le=1)


class SaveClues(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(strict=True, ge=0)
    clues: list[StoredClue] = Field(max_length=256)


class SessionSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(strict=True, ge=0)
    count: int = Field(default=8, strict=True, ge=1, le=64)
    seed: int | None = Field(default=None, strict=True, ge=0, le=4294967295)
    revision: int = Field(default=0, strict=True, ge=0)


def checked(operation, *args):
    try:
        return operation(*args)
    except KeyError:
        raise HTTPException(404, "Session not found.") from None
    except storage.SessionConflict as error:
        raise HTTPException(409, str(error)) from None
    except ValueError as error:
        raise HTTPException(422, str(error)) from None


def record_batch(session_id, expected_revision, request, result, fingerprint):
    with storage.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        current = storage.session_data(db, session_id)
        if current["revision"] != expected_revision:
            raise storage.SessionConflict(
                "Session changed during sampling; this batch was not saved."
            )
        revision = expected_revision + 1
        saved = {
            **result,
            "session_id": session_id,
            "session_revision": revision,
            "model_sha256": fingerprint,
        }
        db.execute(
            """
            UPDATE sessions SET revision = ?,
                updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
            WHERE id = ?
            """,
            (revision, session_id),
        )
        cursor = db.execute(
            """
            INSERT INTO sampling_runs(
                session_id, session_revision, seed, model_name,
                model_sha256, backend, request_json, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, revision, saved["seed"], saved["model"],
                fingerprint, saved["backend"],
                json.dumps({
                    **request,
                    "clues": current["clues"],
                    "seed": saved["seed"],
                }),
                json.dumps(saved),
            ),
        )
        saved["run_id"] = cursor.lastrowid
        db.execute(
            "UPDATE sampling_runs SET result_json = ? WHERE id = ?",
            (json.dumps(saved), saved["run_id"]),
        )
        return saved


def install(app, sample_function, sample_request_class):
    storage.initialize_database()

    @app.post("/api/sessions/open")
    def open_session(request: OpenSession):
        return checked(storage.open_session, request.username, request.new)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        return checked(storage.get_session, session_id)

    @app.put("/api/sessions/{session_id}/clues")
    def save_clues(session_id: str, request: SaveClues):
        return checked(
            storage.save_clues,
            session_id,
            request.expected_revision,
            [clue.model_dump() for clue in request.clues],
        )

    @app.post("/api/sessions/{session_id}/sample")
    def sample_session(session_id: str, request: SessionSample):
        current = checked(storage.get_session, session_id)
        if current["revision"] != request.expected_revision:
            raise HTTPException(409, "Session changed; reload before sampling.")
        result = sample_function(sample_request_class(
            clues=current["clues"],
            count=request.count,
            seed=request.seed,
            revision=request.revision,
        ))
        return checked(
            record_batch,
            session_id,
            request.expected_revision,
            request.model_dump(),
            result,
            app.state.model_sha256,
        )
