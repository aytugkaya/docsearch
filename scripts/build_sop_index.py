from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from docx import Document

from app.embeddings import EMBEDDING_MODEL, get_embeddings_for_texts
from app.sop_retrieval import DATA_ROOT, FORMAT_VERSION, INDEX_PATH, VECTOR_DTYPE

BATCH_SIZE = 64


def read_chunks(path: Path) -> list[str]:
    """Use one nonempty body paragraph per chunk, with its heading if present."""
    heading = ""
    chunks = []

    for paragraph in Document(path).paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        style = paragraph.style.name if paragraph.style else ""
        if style.startswith("Heading") or style == "Title":
            heading = text
            continue

        chunks.append(f"{heading}\n{text}" if heading else text)

    return chunks


def main() -> None:
    root = DATA_ROOT.resolve()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".docx"
        and not path.name.startswith("~$")
    )
    if not paths:
        raise SystemExit(f"No DOCX files found under {root}")

    records = []
    for path in paths:
        sop_key = path.resolve().relative_to(root).as_posix()
        chunks = read_chunks(path)
        if not chunks:
            raise SystemExit(f"No indexable paragraphs in {sop_key}")

        records.extend(
            (sop_key, chunk_no, text)
            for chunk_no, text in enumerate(chunks, start=1)
        )

    # Build beside the final database so a failed build leaves it untouched.
    with NamedTemporaryFile(
        dir=INDEX_PATH.parent, suffix=".sqlite", delete=False
    ) as file:
        temp_path = Path(file.name)

    try:
        with closing(sqlite3.connect(temp_path)) as db, db:
            db.executescript("""
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE sop_chunks (
                    sop_path TEXT NOT NULL,
                    chunk_no INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (sop_path, chunk_no)
                );
            """)

            db.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                [
                    ("format_version", FORMAT_VERSION),
                    ("model_id", EMBEDDING_MODEL),
                ],
            )

            for start in range(0, len(records), BATCH_SIZE):
                batch = records[start:start + BATCH_SIZE]
                vectors = get_embeddings_for_texts(
                    [text for _, _, text in batch]
                )

                db.executemany(
                    """
                    INSERT INTO sop_chunks
                        (sop_path, chunk_no, text, embedding)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (*record, np.asarray(vector, dtype=VECTOR_DTYPE).tobytes())
                        for record, vector in zip(batch, vectors, strict=True)
                    ],
                )

        os.replace(temp_path, INDEX_PATH)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Indexed {len(records)} chunks from {len(paths)} SOPs.")
    print(f"SQLite database: {INDEX_PATH}")


if __name__ == "__main__":
    main()