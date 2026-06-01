from __future__ import annotations

from pathlib import Path
from typing import Literal, Any
import logging
import re

import numpy as np
import pandas as pd
from docx import Document
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
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
def normalize_col_name(col: Any) -> str:
    return str(col).replace("\n", " ").replace("\t", " ").strip().lower()


def normalize_doc_path(raw_path: str) -> str:
    raw = str(raw_path).strip().replace("\\", "/")
    p = Path(raw)
    resolved = p if p.is_absolute() else (DOCS_ROOT_DIR / p)
    return str(resolved.resolve())


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
    df.columns = [normalize_col_name(c) for c in df.columns]
    return df.reset_index(drop=True)


def read_docx_text(path: str) -> str:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"SOP file not found: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"SOP path is not a file: {p}")
    if p.suffix.lower() != ".docx":
        raise ValueError(f"Expected .docx file, got: {p.suffix}")

    logger.info("Reading SOP document from %s", p)
    doc = Document(str(p))
    return "\n".join(
        para.text.strip()
        for para in doc.paragraphs
        if para.text and para.text.strip()
    )


def extract_link_from_row(row: pd.Series) -> str | None:
    alias_cols = [
        "link", "file", "filepath", "file path", "path",
        "document", "document path", "doc", "doc path",
        "sop", "sop link", "sop path"
    ]

    normalized = {normalize_col_name(k): v for k, v in row.items()}

    for col in alias_cols:
        if col in normalized:
            val = str(normalized[col]).strip()
            if val and any(ext in val.lower() for ext in [".docx", ".xlsx", ".csv"]):
                return val

    for value in row.values:
        v = str(value).strip()
        if v and any(ext in v.lower() for ext in [".docx", ".xlsx", ".csv"]):
            return v

    return None


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
    preferred = [
        "company name", "alternate name", "variant", "criteria",
        "notes", "description", "subcategory", "type"
    ]
    cols = [c for c in preferred if c in df.columns]
    if cols:
        return cols

    fallback_exclude = {
        "link", "file", "filepath", "file path", "path",
        "document", "document path", "sop", "sop link", "sop path"
    }
    cols = [c for c in df.columns if c not in fallback_exclude]
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

    candidate_cols = [c for c in ["company name", "alternate name", "variant", "criteria"] if c in df.columns]
    if not candidate_cols:
        candidate_cols = get_search_columns(df)

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
        cells = [f"{col}={str(row[col]).strip()}" for col in cols if col in row.index and str(row[col]).strip()]
        lines.append(f"row_id={row_id} | " + " | ".join(cells))

    return "\n".join(lines)


# -----------------------------
# MODELS
# -----------------------------
class WorkflowState(BaseModel):
    conversation_id: str
    current_index_path: str = ROOT_INDEX_PATH

    awaiting_user_clarification: bool = False
    clarification_question: str | None = None

    selected_link: str | None = None
    active_sop_path: str | None = None
    locked_company_name: str | None = None
    locked_on_sop: bool = False

    status: str = "idle"

    last_user_query: str | None = None
    last_sop_search_query: str | None = None

    root_company_name: str | None = None
    in_secondary_index: bool = False
    secondary_index_path: str | None = None


class RouterDecision(BaseModel):
    route: Literal[
        "general_chat",
        "sop_search",
        "sop_followup",
        "clarification_reply",
        "switch_sop",
    ]
    user_message_for_handler: str = Field(...)
    reason: str | None = None


# class RowSelectionDecision(BaseModel):
#     action: Literal["select_row", "ask_user", "no_match"]
#     selected_row_id: int | None = None
#     clarification_question: str | None = None
#     reason: str | None = None
class RowSelectionDecision(BaseModel):
    action: Literal["select_row", "ask_user", "no_match"]
    selected_row_id: int | None = None
    clarification_question: str | None = None
    distinguishing_fields: list[str] = []
    reason: str | None = None



class SopAnswer(BaseModel):
    answer: str


class GeneralChatAnswer(BaseModel):
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
router_agent = Agent(
    "openai-chat:gpt-4.1-mini",
    output_type=RouterDecision,
    system_prompt=(
        "You are a conversation router for an SOP assistant. "
        "Classify the user's latest message into exactly one route.\n"
        "Routes:\n"
        "- general_chat: greetings, small talk, generic discussion, non-SOP questions.\n"
        "- sop_search: user wants to find/select a company or SOP.\n"
        "- sop_followup: user asks about the currently selected SOP.\n"
        "- clarification_reply: user is answering a disambiguation question.\n"
        "- switch_sop: user wants to change/reset/search another company/SOP.\n\n"
        "Use the workflow state in the prompt.\n"
        "If awaiting clarification and the user gives a short variant/option-like reply, prefer clarification_reply.\n"
        "If an SOP is already locked and the user asks a content/process question, prefer sop_followup.\n"
        "Do not force SOP routes for small talk."
    ),
)

# row_selector_agent = Agent(
#     "openai-chat:gpt-4.1-mini",
#     output_type=RowSelectionDecision,
#     system_prompt=(
#         "You are selecting the best matching row from candidate table rows. "
#         "You may only choose a row_id that appears in the candidate list. "
#         "Return action='select_row' only if one row is clearly best. "
#         "Return action='ask_user' with a short clarification question if several rows are plausible. "
#         "Return action='no_match' if none fit."
#     ),
# )

row_selector_agent = Agent(
    "openai-chat:gpt-4.1-mini",
    output_type=RowSelectionDecision,
    system_prompt=(
        "You are selecting the best matching row from candidate table rows. "
        "You may only choose a row_id that appears in the candidate list. "
        "If one row is clearly best, return action='select_row'. "
        "If multiple rows remain plausible, return action='ask_user' and ask a clarification question. **NEVER use link field** as a distinguishing field."
        "using the most distinguishing fields visible in the candidate rows. **NEVER use link field** as a distinguishing field."
        "Do not ask generic questions like 'first or second document'. "
        "Prefer meaningful field names and values from the rows themselves."
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

general_chat_agent = Agent(
    "openai-chat:gpt-4.1-mini",
    output_type=GeneralChatAnswer,
    system_prompt=(
        "You are a concise, friendly enterprise assistant. "
        "Handle greetings, small talk, and general non-SOP questions. "
        "Do not invent access to SOP content unless it is explicitly provided."
    ),
)


# -----------------------------
# ROUTER
# -----------------------------
async def route_turn(user_input: str, state: WorkflowState) -> RouterDecision:
    prompt = f"""
Workflow state:
- locked_on_sop: {state.locked_on_sop}
- locked_company_name: {state.locked_company_name or ""}
- active_sop_path: {state.active_sop_path or ""}
- awaiting_user_clarification: {state.awaiting_user_clarification}
- clarification_question: {state.clarification_question or ""}
- current_index_path: {state.current_index_path}
- in_secondary_index: {state.in_secondary_index}
- root_company_name: {state.root_company_name or ""}
- last_sop_search_query: {state.last_sop_search_query or ""}

Latest user message:
{user_input}
"""
    result = await router_agent.run(prompt)
    return result.output


# -----------------------------
# ANSWERS
# -----------------------------
async def answer_from_sop(user_question: str, sop_path: str) -> SopAnswer:
    try:
        sop_text = read_docx_text(sop_path)
    except FileNotFoundError:
        logger.exception("SOP file missing: %s", sop_path)
        return SopAnswer(
            answer="I found a matching SOP entry, but the referenced DOCX file is missing from storage."
        )
    except Exception:
        logger.exception("Failed reading SOP file: %s", sop_path)
        return SopAnswer(
            answer="I found a matching SOP entry, but I could not read the SOP document."
        )

    prompt = f"""
User question:
{user_question}

SOP text:
{sop_text}
"""
    result = await qa_agent.run(prompt)
    return result.output


async def answer_general_chat(user_input: str, state: WorkflowState) -> GeneralChatAnswer:
    prompt = f"""
Workflow state:
- locked_on_sop: {state.locked_on_sop}
- locked_company_name: {state.locked_company_name or ""}
- awaiting_user_clarification: {state.awaiting_user_clarification}
- in_secondary_index: {state.in_secondary_index}
- root_company_name: {state.root_company_name or ""}

User message:
{user_input}
"""
    result = await general_chat_agent.run(prompt)
    return result.output


# -----------------------------
# HYBRID RETRIEVAL
# -----------------------------
async def hybrid_retrieve_candidates(query: str, df: pd.DataFrame, embeddings: np.ndarray) -> list[int]:
    exact_ranked = exact_search(df, query)
    keyword_ranked = keyword_search(df, query)
    semantic_ranked = semantic_search(query, df, embeddings)

    logger.info(
        "Hybrid retrieval sizes exact=%s keyword=%s semantic=%s",
        len(exact_ranked), len(keyword_ranked), len(semantic_ranked)
    )

    fused = reciprocal_rank_fusion([exact_ranked, keyword_ranked, semantic_ranked], k=60)
    return fused[:15]


async def select_candidate_row(query: str, df: pd.DataFrame, candidate_ids: list[int], message_history=None):
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
# WORKFLOW HELPERS
# -----------------------------
def reset_search_state(state: WorkflowState, user_input: str) -> None:
    state.current_index_path = ROOT_INDEX_PATH
    state.awaiting_user_clarification = False
    state.clarification_question = None
    state.selected_link = None
    state.active_sop_path = None
    state.locked_company_name = None
    state.locked_on_sop = False
    state.status = "idle"
    state.last_user_query = user_input
    state.last_sop_search_query = user_input
    state.root_company_name = None
    state.in_secondary_index = False
    state.secondary_index_path = None


# -----------------------------
# SOP SEARCH / LOCK FLOW
# -----------------------------
async def handle_sop_search(
    user_input: str,
    state: WorkflowState,
    message_history=None,
) -> ChatTurnResult:
    if message_history is None:
        message_history = []

    state.last_user_query = user_input
    state.last_sop_search_query = user_input

    while True:
        df, _, embeddings = prepare_table_index(state.current_index_path)

        candidate_ids = await hybrid_retrieve_candidates(
            state.last_sop_search_query,
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
            state.last_sop_search_query,
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
        logger.info("Selected row columns: %s", list(df.columns))
        logger.info("Selected row values: %s", selected_row.to_dict())

        link = extract_link_from_row(selected_row)

        if not link:
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
            state.current_index_path = resolved_link
            state.secondary_index_path = resolved_link
            state.in_secondary_index = True
            state.status = "searching_index"

            if not state.root_company_name:
                state.root_company_name = (
                    str(selected_row.get("company name", "")).strip()
                    or str(selected_row.get("alternate name", "")).strip()
                    or None
                )

            message_history = raw_result.all_messages() if raw_result is not None else message_history
            continue

        if resolved_link.lower().endswith(".docx"):
            state.selected_link = resolved_link
            state.active_sop_path = resolved_link
            state.locked_company_name = (
                str(selected_row.get("company name", "")).strip()
                or state.root_company_name
                or None
            )
            state.locked_on_sop = True
            state.awaiting_user_clarification = False
            state.clarification_question = None
            state.status = "locked_on_sop"
            state.in_secondary_index = False
            state.secondary_index_path = None

            matched_name = state.locked_company_name or "the selected company"
            assistant_text = (
                f"I found and opened the SOP for {matched_name}. "
                "This SOP is now selected for the session.\n\n"
                "You can now ask me questions about this SOP, ask general chat questions, "
                "or ask for a different company."
            )

            return ChatTurnResult(
                event_type="final_answer",
                assistant_text=assistant_text,
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


async def handle_sop_followup(user_input: str, state: WorkflowState) -> ChatTurnResult:
    if not state.locked_on_sop or not state.active_sop_path:
        return ChatTurnResult(
            event_type="error",
            assistant_text="No SOP is currently selected. Ask me to search for a company or SOP first.",
            state=state,
            raw_result=None,
        )

    state.last_user_query = user_input
    sop_result = await answer_from_sop(user_input, state.active_sop_path)

    return ChatTurnResult(
        event_type="final_answer",
        assistant_text=sop_result.answer,
        state=state,
        raw_result=None,
    )


async def handle_clarification_reply(
    user_input: str,
    state: WorkflowState,
    message_history=None,
) -> ChatTurnResult:
    if message_history is None:
        message_history = []

    state.last_user_query = user_input

    if state.last_sop_search_query:
        merged_query = f"{state.last_sop_search_query}\nUser clarification: {user_input}"
    else:
        merged_query = user_input

    state.awaiting_user_clarification = False
    state.clarification_question = None
    state.last_sop_search_query = merged_query

    return await handle_sop_search(
        user_input=merged_query,
        state=state,
        message_history=message_history,
    )


async def handle_switch_sop(
    user_input: str,
    state: WorkflowState,
    message_history=None,
) -> ChatTurnResult:
    if message_history is None:
        message_history = []

    reset_search_state(state, user_input)
    return await handle_sop_search(
        user_input=user_input.strip(),
        state=state,
        message_history=message_history,
    )


async def handle_general_turn(user_input: str, state: WorkflowState) -> ChatTurnResult:
    state.last_user_query = user_input
    chat_result = await answer_general_chat(user_input, state)

    return ChatTurnResult(
        event_type="final_answer",
        assistant_text=chat_result.answer,
        state=state,
        raw_result=None,
    )


# -----------------------------
# MAIN ENTRYPOINT
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
            current_index_path=ROOT_INDEX_PATH,
            status="idle",
        )

    decision = await route_turn(user_input, state)
    routed_text = decision.user_message_for_handler.strip() or user_input.strip()

    logger.info(
        "Router chose route=%s reason=%s conversation_id=%s",
        decision.route,
        decision.reason,
        state.conversation_id,
    )

    if decision.route == "general_chat":
        return await handle_general_turn(routed_text, state)

    if decision.route == "sop_followup":
        return await handle_sop_followup(routed_text, state)

    if decision.route == "clarification_reply":
        return await handle_clarification_reply(routed_text, state, message_history)

    if decision.route == "switch_sop":
        return await handle_switch_sop(routed_text, state, message_history)

    return await handle_sop_search(routed_text, state, message_history)