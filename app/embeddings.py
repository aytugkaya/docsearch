from functools import lru_cache

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


@lru_cache(maxsize=1)
def embedding_client() -> OpenAI:
    return OpenAI()


def get_embeddings_for_texts(texts: list[str]) -> np.ndarray:
    response = embedding_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return np.asarray(
        [item.embedding for item in sorted(response.data, key=lambda x: x.index)],
        dtype=np.float32,
    )