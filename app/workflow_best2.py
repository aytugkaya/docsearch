from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any
import logging
import re

import numpy as np
import pandas as pd
from docx import Document
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)

ROOT_INDEX_PATH = "/data/docs/index.xlsx"
DOCS_ROOT_DIR = Path("/data/docs")
EMBEDDING_MODEL = "text-embedding-3-small"

_openai_client = OpenAI()
_embedding_cache: dict[str, tuple[pd.DataFrame, np.ndarray, list[str]]] = {}


# -----------------------------
# DEPENDENCIES
# -----------------------------
@dataclass
class SearchDeps:
    index_path: str


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

    in_secondary_index: bool = False
    has_asked_secondary_clarification: bool = False

    status: str = "searching_index"


class RowSelectionDecision(BaseModel):
    action: Literal["select_row", "ask_user", "no_match", "respond"]
    selected_row_id: int | None = None
    clarification_question: str | None = None
    response_text: str | None = None
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
# BASE HELPERS
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


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def get_search_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["company name", "alternate name", "notes"]
    cols = [c for c in preferred if c in df.columns]
    return cols if cols else list(df.columns)


def build_row_search_text(row: pd.Series, columns: list[str]) -> str:
    parts = []
    for c in columns:
        if c in row.index:
            value = str(row[c]).strip()
            if value:
                parts.append(f"{c}: {value}")
    return " | ".join(parts)


def build_search_texts(df: pd.DataFrame) -> list[str]:
    cols = get_search_columns(df)
    return [build_row_search_text(row, cols) for _, row in df.iterrows()]


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


def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = np.clip(matrix_norms * query_norm, 1e-12, None)
    return np.dot(matrix, query_vec) / denom


def prepare_table_index(index_path: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    cache_key = str(Path(index_path).resolve())

    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    df = load_small_table(index_path)
    texts = build_search_texts(df)
    embeddings = get_embeddings_for_texts(texts)

    _embedding_cache[cache_key] = (df, embeddings, texts)
    logger.info("Prepared cached table index for %s rows", len(df))
    return df, embeddings, texts


def reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for rank_list in rank_lists:
        for rank, row_id in enumerate(rank_list, start=1):
            scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [row_id for row_id, _ in ranked]


# -----------------------------
# RETRIEVAL AGENT
# -----------------------------
retrieval_agent = Agent(
    "openai-chat:gpt-4.1-mini",
    deps_type=SearchDeps,
    output_type=RowSelectionDecision,
    system_prompt=(
    "You are a table retrieval assistant for SOP lookup. "
    "First determine the user's intent. "
    "Possible intents are: "
    "1) search for a company/SOP row, "
    "2) ask the user for clarification, "
    "3) respond conversationally when the input is a greeting, help request, or other non-search message, "
    "4) no_match when the message is a search request but no plausible row exists. "
    "Use tools only when the user is actually trying to find a row. "
    "If the user greets you or asks what you do, return action='respond' with a short helpful reply. "
    "If the user provides too little information to search, return action='ask_user'. "
    "Only return action='select_row' if you can choose a row_id returned by tools. "
    "Never invent a row_id."
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


@retrieval_agent.tool
def exact_search_tool(ctx: RunContext[SearchDeps], query: str, limit: int = 20) -> list[int]:
    """Find rows whose company name or alternate name exactly matches the user's query."""
    df, _, _ = prepare_table_index(ctx.deps.index_path)

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
                hits.append(int(idx))
                break

    return hits[:limit]


@retrieval_agent.tool
def keyword_search_tool(ctx: RunContext[SearchDeps], query: str, limit: int = 50) -> list[int]:
    """Find rows by keyword overlap across company name, alternate name, and notes."""
    df, _, _ = prepare_table_index(ctx.deps.index_path)
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
            scored.append((int(idx), score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:limit]]


@retrieval_agent.tool
def semantic_search_tool(ctx: RunContext[SearchDeps], query: str, limit: int = 50) -> list[int]:
    """Find rows by semantic similarity between the query and row text using embeddings."""
    df, embeddings, _ = prepare_table_index(ctx.deps.index_path)
    if len(df) == 0:
        return []

    query_vec = get_query_embedding(query)
    sims = cosine_similarity_matrix(query_vec, embeddings)
    ranked = np.argsort(-sims)
    return [int(i) for i in ranked[:limit].tolist()]


@retrieval_agent.tool_plain
def fuse_rankings_tool(rank_lists: list[list[int]], k: int = 60, limit: int = 15) -> list[int]:
    """Combine multiple ranked row-id lists using Reciprocal Rank Fusion."""
    fused = reciprocal_rank_fusion(rank_lists, k=k)
    return fused[:limit]


@retrieval_agent.tool
def get_rows_tool(ctx: RunContext[SearchDeps], row_ids: list[int]) -> list[dict[str, Any]]:
    """Return full row data for the provided row ids so you can inspect candidates before selecting one."""
    df, _, _ = prepare_table_index(ctx.deps.index_path)
    rows = []

    for row_id in row_ids:
        if 0 <= row_id < len(df):
            row = df.iloc[row_id].to_dict()
            row["row_id"] = int(row_id)
            rows.append(row)

    return rows


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
        if state.in_secondary_index:
            state.has_asked_secondary_clarification = True

    else:
        state.original_user_query = user_input

    while True:
        deps = SearchDeps(index_path=state.current_index_path)

        result = await retrieval_agent.run(
            f"User request: {state.original_user_query}",
            deps=deps,
            message_history=message_history,
        )

        decision = result.output

        if decision.action == "ask_user":
            question = decision.clarification_question or "I found multiple plausible matches. Which one did you mean?"
            state.awaiting_user_clarification = True
            state.clarification_question = question
            state.status = "needs_clarification"

            return ChatTurnResult(
                event_type="ask_user",
                assistant_text=question,
                state=state,
                raw_result=result,
            )
        if decision.action == "respond":
            return ChatTurnResult(
                event_type="ask_user",
                assistant_text=decision.response_text or "Hi — I can help you find the right SOP. Tell me the company name or alternate name.",
                state=state,
                raw_result=result,
            )

        if decision.action == "no_match" or decision.selected_row_id is None:
            state.status = "error"
            return ChatTurnResult(
                event_type="error",
                assistant_text="I could not confidently identify the right row.",
                state=state,
                raw_result=result,
            )

        df, _, _ = prepare_table_index(state.current_index_path)

        if not (0 <= decision.selected_row_id < len(df)):
            state.status = "error"
            return ChatTurnResult(
                event_type="error",
                assistant_text="The selected row id was invalid.",
                state=state,
                raw_result=result,
            )

        selected_row = df.iloc[decision.selected_row_id]
        link = extract_link_from_row(selected_row)

        if not link:
            logger.warning("Selected row has no usable link")
            state.status = "error"
            return ChatTurnResult(
                event_type="error",
                assistant_text="The selected row did not contain a usable file link.",
                state=state,
                raw_result=result,
            )

        resolved_link = normalize_doc_path(link)
        logger.info("Resolved selected link to %s", resolved_link)


        # NEW: if we are in a secondary index and have not yet asked
        # a clarifying question, do NOT allow direct SOP selection.
        if state.in_secondary_index and not state.has_asked_secondary_clarification:
            logger.info("Secondary index requires at least one disambiguation question")
            # Build a clarification prompt from the top candidate rows + their fields,
            # or just delegate to the agent with a strong instruction.
            state.awaiting_user_clarification = True
            state.clarification_question = (
                "I have multiple SOP variants for this company. "
                "Please specify which one you need (for example by address, variant, or other detail)."
            )
            state.status = "needs_clarification"
            return ChatTurnResult(
                event_type="ask_user",
                assistant_text=state.clarification_question,
                state=state,
                raw_result=result,
            )

        if resolved_link.lower().endswith((".xlsx", ".csv")):
            logger.info("Selected link is another table, descending into nested index")
            state.current_index_path = resolved_link
            state.status = "searching_index"
            state.in_secondary_index = True
            state.has_asked_secondary_clarification = False
            message_history = result.all_messages()
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
                raw_result=result,
            )

        state.status = "error"
        return ChatTurnResult(
            event_type="error",
            assistant_text="Selected row did not contain a usable DOCX, XLSX, or CSV link.",
            state=state,
            raw_result=result,
        )