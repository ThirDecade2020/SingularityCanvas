import json
from pathlib import Path

from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/symbols"


def build_reference(model_path, model_sha256):
    metadata = json.loads(model_path.with_suffix(".json").read_text())
    catalogue = json.loads((FOLDER / "catalogue.json").read_text())
    patterns = {pattern["id"]: pattern for pattern in catalogue["patterns"]}
    training_ids = metadata["training_ids"]

    if not training_ids or len(set(training_ids)) != len(training_ids):
        raise ValueError("Training IDs must be present and unique.")
    if catalogue["width"] != 16 or catalogue["height"] != 16:
        raise ValueError("Expected a 16 by 16 catalogue.")
    if not (FOLDER / "LICENSE").is_file():
        raise ValueError("Catalogue license is missing.")

    symbols = []
    for symbol_id in training_ids:
        pattern = patterns[symbol_id]
        bits = pattern["bits"]
        names = pattern["names"]
        if len(bits) != 256 or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
            raise ValueError("Invalid training pattern.")
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Missing symbol names.")
        symbols.append({
            "id": symbol_id,
            "names": names,
            "bits": bits,
        })

    return {
        "model": model_path.name,
        "model_sha256": model_sha256,
        "source": catalogue["source"],
        "source_version": catalogue["sourceVersion"],
        "license_url": "/api/model/symbols/license",
        "width": 16,
        "height": 16,
        "symbol_count": len(symbols),
        "symbols": symbols,
        "role": "Training references; these are not generated completions.",
    }


def install(app):
    @app.get("/api/model/symbols")
    def symbols():
        return app.state.symbol_reference

    @app.get("/api/model/symbols/license")
    def license_text():
        return FileResponse(FOLDER / "LICENSE", media_type="text/plain")
