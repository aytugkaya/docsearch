from __future__ import annotations

from pathlib import Path
from typing import Literal, Any
import logging
import math
import re

import numpy as np
import pandas as pd
from docx import Document
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent

logger = logging.getLogger(__name__)

ROOT_INDEX_PATH = "/data/docs/index.xlsx"
DOCS_ROOT_DIR = Path("/data/docs")

EMBEDDING_MODEL = "text-embedding-3-small"

_openai_client = OpenAI()
_embedding_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}


# -----------------------------
# HELPERS
# -----------------------------
def normalize_doc_path(raw_path: str) -> str:
    raw_path = str(raw_path).strip()
    filename = Path(raw_path.replace("\\", "/")).name
    return str(DOCS_ROOT_DIR / filename)


def load_small_table(index_path: str) -> pd.DataFrame:
    path = Path(index_path)
    logger.info("Loading index table from %s", path)

    if not path.exists():
        logger.error("Index file not found: %s", path)
        raise FileNotFoundError(f"Index file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported table type: {suffix}")

    df = df.fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df.reset_index(drop=True)


def read_docx_text(path: str) -> str:
    logger.info("Reading SOP document from %s", path)
    doc = Document(path)
    return "\n".join(
        p.text.strip()
        for p in doc.paragraphs
        if p.text and p.text.strip()
    )


def extract_link_from_row(row: pd.Series) -> str | None:
    if "link" in row.index:
        val = str(row["link"]).strip()
        if val:
            return val

    for value in row.values:
        if isinstance(value, str):
            v = value.strip()
            if v.lower().endswith((".docx", ".xlsx", ".csv")):
                return v
    return None


def wants_new_search(text: str) -> bool:
    t = text.lower().strip()
    triggers = [
        "switch company",
        "new company",
        "different company",
        "change company",
        "search again",
        "look up another",
        "reset",
        "start over",
    ]
    return any(x in t for x in triggers)


def normalize_text(v: Any) -> str:
    return str(v).strip().lower()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def build_row_search_text(row: pd.Series, columns: list[str]) -> str:
    parts = []
    for c in columns:
        if c in row.index:
            value = str(row[c]).strip()
            if value:
                parts.append(f"{c}: {value}")
    return " | ".join(parts)


def get_search_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["company name", "alternate name", "notes"]
    cols = [c for c in preferred if c in df.columns]
    return cols if cols else list(df.columns)


def build_search_texts(df: pd.DataFrame) -> list[str]:
    cols = get_search_columns(df)
    return [build_row_search_text(row, cols) for _, row in df.iterrows()]


def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = np.clip(matrix_norms * query_norm, 1e-12, None)
    return np.dot(matrix, query_vec) / denom


def get_embeddings_for_texts(texts: list[str]) -> np.ndarray:
    response = _openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def get_query_embedding(text: str) -> np.ndarray:
    response = _openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def prepare_table_index(index_path: str) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    cache_key = str(Path(index_path).resolve())
    if cache_key in _embedding_cache:
        df, embeddings = _embedding_cache[cache_key]
        texts = build_search_texts(df)
        return df, texts, embeddings

    df = load_small_table(index_path)
    texts = build_search_texts(df)
    embeddings = get_embeddings_for_texts(texts)
    _embedding_cache[cache_key] = (df, embeddings)
    logger.info("Prepared embedding index for %s rows", len(df))
    return df, texts, embeddings


def exact_search(df: pd.DataFrame, query: str) -> list[int]:
    q = query.strip().lower()
    if not q:
        return []

    candidate_cols = [c for c in ["company name", "alternate name"] if c in df.columns]
    if not candidate_cols:
        candidate_cols = list(df.columns)

    hits = []
    for idx, row in df.iterrows():
        for c in candidate_cols:
            val = str(row[c]).strip().lower()
            if val == q:
                hits.append(idx)
                break
    return hits


def keyword_search(df: pd.DataFrame, query: str) -> list[int]:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []

    cols = get_search_columns(df)
    scored = []

    for idx, row in df.iterrows():
        text = " ".join(str(row[c]) for c in cols if c in row.index).lower()
        row_tokens = set(tokenize(text))
        overlap = len(query_tokens & row_tokens)

        if overlap > 0:
            phrase_bonus = 2 if query.lower() in text else 0
            score = overlap + phrase_bonus
            scored.append((idx, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:50]]


def semantic_search(query: str, df: pd.DataFrame, embeddings: np.ndarray) -> list[int]:
    query_vec = get_query_embedding(query)
    sims = cosine_similarity_matrix(query_vec, embeddings)
    ranked = np.argsort(-sims)
    return ranked[:50].tolist()


def reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}

    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked]


def candidate_rows_for_prompt(df: pd.DataFrame, row_ids: list[int], max_rows: int = 10) -> str:
    lines = []
    cols = get_search_columns(df)
    if "link" in df.columns and "link" not in cols:
        cols = cols + ["link"]

    for row_id in row_ids[:max_rows]:
        row = df.iloc[row_id]
        cells = [f"{col}={str(row[col]).strip()}" for col in cols if col in row.index]
        lines.append(f"row_id={row_id} | " + " | ".join(cells))
    return "\n".join(lines)


# -----------------------------
# MODELS
# -----------------------------
class WorkflowState(BaseModel):
    conversation_id: str
    original_user_query: str
    current_index_path: str

    awaiting_user_clarification: bool = False
    clarification_question: str | None = None

    selected_link: str | None = None
    active_sop_path: str | None = None
    locked_company_name: str | None = None
    locked_on_sop: bool = False

    status: str = "searching_index"


class RowSelectionDecision(BaseModel):
    action: Literal["select_row", "ask_user", "no_match"]
    selected_row_id: int | None = None
    clarification_question: str | None = None
    reason: str | None = None


class SopAnswer(BaseModel):
    answer: str


class ChatTurnResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_type: Literal["ask_user", "final_answer", "error"]
    assistant_text: str
    state: WorkflowState
    raw_result: Any = None


# -----------------------------
# AGENTS
# -----------------------------
row_selector_agent = Agent(
    "openai-chat:gpt-4.1-mini",
    output_type=RowSelectionDecision,
    system_prompt=(
        "You are selecting the best matching row from candidate table rows. "
        "You may only choose a row_id that appears in the candidate list. "
        "Return action='select_row' only if one row is clearly best. "
        "Return action='ask_user' with a short clarification question if several rows are plausible. "
        "Return action='no_match' if none fit."
    ),
)

qa_agent = Agent(
    "openai-chat:gpt-4.1-mini",
    output_type=SopAnswer,
    system_prompt=(
        "You answer questions only from the SOP text provided. "
        "Be concise. If the answer is not in the SOP, say that clearly. "
        "Do not switch companies or speculate. "
        "If the answer is not in the SOP, say: 'That information is not in the selected SOP.'"
    ),
)


# -----------------------------
# SOP QA
# -----------------------------
async def answer_from_sop(user_question: str, sop_path: str) -> SopAnswer:
    try:
        sop_text = read_docx_text(sop_path)
        prompt = f"""
User question:
{user_question}

SOP text:
{sop_text}
"""
        result = await qa_agent.run(prompt)
        return result.output
    except Exception:
        logger.exception("Failed answering from SOP path=%s", sop_path)
        raise


# -----------------------------
# HYBRID RETRIEVAL
# -----------------------------
async def hybrid_retrieve_candidates(
    query: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> list[int]:
    exact_ranked = exact_search(df, query)
    keyword_ranked = keyword_search(df, query)
    semantic_ranked = semantic_search(query, df, embeddings)

    logger.info(
        "Hybrid retrieval sizes exact=%s keyword=%s semantic=%s",
        len(exact_ranked), len(keyword_ranked), len(semantic_ranked)
    )

    fused = reciprocal_rank_fusion([exact_ranked, keyword_ranked, semantic_ranked], k=60)
    return fused[:15]


async def select_candidate_row(
    query: str,
    df: pd.DataFrame,
    candidate_ids: list[int],
    message_history=None,
):
    if message_history is None:
        message_history = []

    if not candidate_ids:
        return None, None

    if len(candidate_ids) == 1:
        return candidate_ids[0], None

    candidate_block = candidate_rows_for_prompt(df, candidate_ids, max_rows=10)
    prompt = f"""
User request:
{query}

Candidate rows:
{candidate_block}

Choose the best row_id if one row is clearly best.
If several are plausible, ask the user a clarification question.
"""

    result = await row_selector_agent.run(prompt, message_history=message_history)
    decision = result.output

    if decision.action == "select_row" and decision.selected_row_id in candidate_ids:
        return decision.selected_row_id, result

    if decision.action == "ask_user":
        return decision, result

    return None, result


# -----------------------------
# MAIN WORKFLOW
# -----------------------------
async def handle_turn(
    user_input: str,
    state: WorkflowState | None,
    message_history=None,
) -> ChatTurnResult:
    if message_history is None:
        message_history = []

    if state is None:
        state = WorkflowState(
            conversation_id="demo-1",
            original_user_query=user_input,
            current_index_path=ROOT_INDEX_PATH,
            status="searching_index",
        )
        logger.info("Starting new workflow")

    elif state.locked_on_sop and state.active_sop_path and not wants_new_search(user_input):
        logger.info("Locked on SOP, answering follow-up from %s", state.active_sop_path)
        sop_result = await answer_from_sop(user_input, state.active_sop_path)
        return ChatTurnResult(
            event_type="final_answer",
            assistant_text=sop_result.answer,
            state=state,
            raw_result=None,
        )

    elif state.locked_on_sop and wants_new_search(user_input):
        logger.info("Resetting locked SOP state due to new search")
        state.locked_on_sop = False
        state.active_sop_path = None
        state.locked_company_name = None
        state.selected_link = None
        state.awaiting_user_clarification = False
        state.clarification_question = None
        state.current_index_path = ROOT_INDEX_PATH
        state.status = "searching_index"
        state.original_user_query = user_input

    elif state.awaiting_user_clarification:
        logger.info("Received user clarification")
        state.original_user_query += f"\nUser clarification: {user_input}"
        state.awaiting_user_clarification = False
        state.clarification_question = None

    else:
        state.original_user_query = user_input

    while True:
        df, _, embeddings = prepare_table_index(state.current_index_path)

        candidate_ids = await hybrid_retrieve_candidates(
            state.original_user_query,
            df,
            embeddings,
        )

        if not candidate_ids:
            state.status = "error"
            return ChatTurnResult(
                event_type="error",
                assistant_text="I could not identify any likely matching rows in the table.",
                state=state,
                raw_result=None,
            )

        selection, raw_result = await select_candidate_row(
            state.original_user_query,
            df,
            candidate_ids,
            message_history=message_history,
        )

        if isinstance(selection, RowSelectionDecision) and selection.action == "ask_user":
            question = selection.clarification_question or "I found multiple plausible matches. Which one did you mean?"
            state.awaiting_user_clarification = True
            state.clarification_question = question
            state.status = "needs_clarification"

            return ChatTurnResult(
                event_type="ask_user",
                assistant_text=question,
                state=state,
                raw_result=raw_result,
            )

        if selection is None:
            state.status = "error"
            return ChatTurnResult(
                event_type="error",
                assistant_text="I could not confidently select the right row.",
                state=state,
                raw_result=raw_result,
            )

        selected_row = df.iloc[selection]
        link = extract_link_from_row(selected_row)

        if not link:
            logger.warning("Selected row has no usable link")
            state.status = "error"
            return ChatTurnResult(
                event_type="error",
                assistant_text="The selected row did not contain a usable file link.",
                state=state,
                raw_result=raw_result,
            )

        resolved_link = normalize_doc_path(link)
        logger.info("Resolved selected link to %s", resolved_link)

        if resolved_link.lower().endswith((".xlsx", ".csv")):
            logger.info("Selected link is another table, descending into nested index")
            state.current_index_path = resolved_link
            state.status = "searching_index"
            message_history = raw_result.all_messages() if raw_result is not None else message_history
            continue

        if resolved_link.lower().endswith(".docx"):
            state.selected_link = resolved_link
            state.active_sop_path = resolved_link
            state.locked_company_name = str(selected_row.get("company name", ""))
            state.locked_on_sop = True
            state.status = "locked_on_sop"

            sop_result = await answer_from_sop(
                state.original_user_query,
                state.active_sop_path,
            )

            locked_note = "\n\n[Locked to current SOP. Say 'switch company' to search another company.]"

            return ChatTurnResult(
                event_type="final_answer",
                assistant_text=sop_result.answer + locked_note,
                state=state,
                raw_result=raw_result,
            )

        state.status = "error"
        return ChatTurnResult(
            event_type="error",
            assistant_text="Selected row did not contain a usable DOCX, XLSX, or CSV link.",
            state=state,
            raw_result=raw_result,
        )