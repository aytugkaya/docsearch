from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
from openai import OpenAIError

from app.embeddings import EMBEDDING_MODEL, get_embeddings_for_texts

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
INDEX_PATH = DATA_ROOT / "sop_index.sqlite"
FORMAT_VERSION = "2"
VECTOR_DTYPE = np.dtype("<f4")


class SopIndexError(RuntimeError):
    pass


def retrieve_sop_chunks(
    question: str,
    sop_path: str | Path,
    *,
    n: int = 4,
) -> list[tuple[int, str]]:
    if n < 1:
        raise ValueError("n must be at least 1")

    try:
        key = Path(sop_path).resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise SopIndexError("Selected SOP is outside the data directory.") from exc

    try:
        with closing(sqlite3.connect(
            f"{INDEX_PATH.as_uri()}?mode=ro", uri=True
        )) as db:
            meta = dict(db.execute("SELECT key, value FROM metadata"))
            if (
                meta.get("format_version") != FORMAT_VERSION
                or meta.get("model_id") != EMBEDDING_MODEL
            ):
                raise SopIndexError("SOP index is incompatible; rebuild it.")

            rows = db.execute(
                """
                SELECT chunk_no, text, embedding
                FROM sop_chunks
                WHERE sop_path = ?
                ORDER BY chunk_no
                """,
                (key,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise SopIndexError("Could not read the SOP index.") from exc

    if not rows:
        raise SopIndexError(f"No indexed chunks for {key}.")

    try:
        matrix = np.stack([
            np.frombuffer(row[2], dtype=VECTOR_DTYPE) for row in rows
        ])
        query = get_embeddings_for_texts([question])[0]
        denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        scores = (matrix @ query) / np.maximum(denom, 1e-12)
    except (OpenAIError, ValueError) as exc:
        raise SopIndexError("Could not score SOP chunks.") from exc

    best = np.argsort(-scores, kind="stable")[:n]
    return [(rows[i][0], rows[i][1]) for i in sorted(best)]