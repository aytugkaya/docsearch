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
_embedding_cache: dict[str, tuple[pd.DataFrame, np.ndarray, list[str]]] = {}


# -----------------------------
# MODELS
# -----------------------------
class EntityMention(BaseModel):
    surface_text: str
    normalized_text: str | None = None
    role: Literal[
        "company_name",
        "alternate_name",
        "identifier",
        "constraint",
        "unknown",
    ] = "unknown"
    can_lookup_sop: bool = False


class ParsedTurn(BaseModel):
    intent: Literal["search_and_answer", "search_row", "chat"]
    retrieval_status: Literal["ready", "ambiguous", "missing_reference"]
    mentions: list[EntityMention] = Field(default_factory=list)
    retrieval_query_text: str | None = None
    user_question_for_sop: str | None = None
    clarification_question: str | None = None
    reason: str | None = None


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

    pending_user_question: str | None = None
    parsed_mentions: list[EntityMention] = Field(default_factory=list)

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
    next_message_history: list[Any] = Field(default_factory=list)
    raw_result: Any = None


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


def canonicalize_lookup_text(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[_\-\s]+", "", s)
    return s


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_-]+", text.lower())


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
    return df, embeddings, texts


def reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for rank_list in rank_lists:
        for rank, row_id in enumerate(rank_list, start=1):
            scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [row_id for row_id, _ in ranked]


# -----------------------------
# AGENTS
# -----------------------------
turn_parser_agent = Agent(
    "openai-chat:gpt-4.1-mini",
    output_type=ParsedTurn,
    system_prompt=(
        "You parse the user's turn into a structured object for SOP search. "
        "Do not answer the SOP question. "
        "A company name, alternate name, or identifier counts as a usable lookup reference. "
        "If the user already provided a concrete company name, alternate name, or identifier, set retrieval_status='ready'. "
        "Do not ask for the company again in that case. "
        "Use intent='search_and_answer' when the user asks an SOP-content question and also provides a lookup reference. "
        "Use intent='search_row' when the user is primarily asking to find the SOP/company row. "
        "Use intent='chat' only for greetings, help requests, or non-search conversation. "
        "Set retrieval_status='missing_reference' only when no concrete lookup reference is present. "
        "Set retrieval_status='ambiguous' only when the reference itself is genuinely ambiguous. "
        "For each mention, set can_lookup_sop=True only for usable company_name, alternate_name, or identifier mentions. "
        "retrieval_query_text should contain only the shortest useful lookup string for row retrieval. "
        "user_question_for_sop should contain only the SOP question to ask after the SOP is locked. "
        "If the user asks an SOP question but no lookup reference is present, provide a brief clarification_question asking for the company name or alternate name."
    ),
)

row_judge_agent = Agent(
    "openai-chat:gpt-4.1-mini",
    output_type=RowSelectionDecision,
    system_prompt=(
        "You decide which table row best matches a row-lookup request after hybrid retrieval has already produced candidates. "
        "You will receive the lookup query and a small list of candidate rows. "
        "Prefer exact identifier and alternate-name matches when present. "
        "If one row is clearly best, return action='select_row' with its row_id. "
        "If multiple rows are still genuinely plausible, return action='ask_user' and ask a short differentiating question grounded in the candidate rows. "
        "If none are plausible, return action='no_match'. "
        "Never invent a row_id not present in the candidate list. "
        "Do not answer the SOP question."
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
# HYBRID RETRIEVAL
# -----------------------------
def exact_search_ids(df: pd.DataFrame, query: str, limit: int = 20) -> list[int]:
    q = query.strip().lower()
    if not q:
        return []

    q_canon = canonicalize_lookup_text(q)
    candidate_cols = [c for c in ["company name", "alternate name"] if c in df.columns]
    if not candidate_cols:
        candidate_cols = list(df.columns)

    hits = []
    for idx, row in df.iterrows():
        for c in candidate_cols:
            val = str(row[c]).strip().lower()
            if val == q or canonicalize_lookup_text(val) == q_canon:
                hits.append(int(idx))
                break

    return hits[:limit]


def keyword_search_ids(df: pd.DataFrame, query: str, limit: int = 50) -> list[int]:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []

    cols = get_search_columns(df)
    scored = []
    q_lower = query.lower()
    q_canon = canonicalize_lookup_text(query)

    for idx, row in df.iterrows():
        text_parts = [str(row[c]) for c in cols if c in row.index]
        text = " ".join(text_parts).lower()
        row_tokens = set(tokenize(text))

        overlap = len(query_tokens & row_tokens)
        phrase_bonus = 2 if q_lower in text else 0

        canon_bonus = 0
        for c in cols:
            if c in row.index:
                val = str(row[c]).strip().lower()
                if canonicalize_lookup_text(val) == q_canon:
                    canon_bonus = 5
                    break

        score = overlap + phrase_bonus + canon_bonus
        if score > 0:
            scored.append((int(idx), score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored[:limit]]


def semantic_search_ids(index_path: str, query: str, limit: int = 50) -> list[int]:
    df, embeddings, _ = prepare_table_index(index_path)
    if len(df) == 0:
        return []

    query_vec = get_query_embedding(query)
    sims = cosine_similarity_matrix(query_vec, embeddings)
    ranked = np.argsort(-sims)
    return [int(i) for i in ranked[:limit].tolist()]


def get_candidate_rows(df: pd.DataFrame, row_ids: list[int]) -> list[dict[str, Any]]:
    rows = []
    for row_id in row_ids:
        if 0 <= row_id < len(df):
            row = df.iloc[row_id].to_dict()
            row["row_id"] = int(row_id)
            rows.append(row)
    return rows


async def hybrid_select_row(index_path: str, retrieval_query: str) -> tuple[RowSelectionDecision, list[dict[str, Any]], Any]:
    df, _, _ = prepare_table_index(index_path)

    exact_ids = exact_search_ids(df, retrieval_query, limit=20)
    keyword_ids = keyword_search_ids(df, retrieval_query, limit=50)
    semantic_ids = semantic_search_ids(index_path, retrieval_query, limit=50)

    fused_ids = reciprocal_rank_fusion(
        [exact_ids, keyword_ids, semantic_ids],
        k=60,
    )[:15]

    logger.info(
        "Hybrid retrieval query=%s exact=%s keyword=%s semantic_top=%s fused=%s",
        retrieval_query,
        exact_ids,
        keyword_ids,
        semantic_ids[:10],
        fused_ids,
    )

    candidates = get_candidate_rows(df, fused_ids)
    if not candidates:
        return (
            RowSelectionDecision(
                action="no_match",
                reason=f"No rows matched the hybrid retrieval query '{retrieval_query}'.",
            ),
            [],
            None,
        )

    judge_prompt = f"""
Lookup query:
{retrieval_query}

Candidate rows:
{candidates}
"""
    judge_run = await row_judge_agent.run(judge_prompt)
    return judge_run.output, candidates, judge_run


# -----------------------------
# SOP QA
# -----------------------------
async def answer_from_sop(user_question: str, sop_path: str) -> SopAnswer:
    sop_text = read_docx_text(sop_path)
    prompt = f"""
User question:
{user_question}

SOP text:
{sop_text}
"""
    result = await qa_agent.run(prompt)
    return result.output


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

    if state.locked_on_sop and state.active_sop_path and not wants_new_search(user_input):
        sop_result = await answer_from_sop(user_input, state.active_sop_path)
        return ChatTurnResult(
            event_type="final_answer",
            assistant_text=sop_result.answer,
            state=state,
            next_message_history=[],
            raw_result=None,
        )

    if state.locked_on_sop and wants_new_search(user_input):
        state.locked_on_sop = False
        state.active_sop_path = None
        state.locked_company_name = None
        state.selected_link = None
        state.awaiting_user_clarification = False
        state.clarification_question = None
        state.current_index_path = ROOT_INDEX_PATH
        state.status = "searching_index"
        state.original_user_query = user_input
        state.in_secondary_index = False
        state.has_asked_secondary_clarification = False
        state.pending_user_question = None
        state.parsed_mentions = []

    parse_run = None
    search_request = None

    if state.awaiting_user_clarification:
        state.original_user_query = f"{state.original_user_query}\nClarification: {user_input}".strip()
        state.awaiting_user_clarification = False
        state.clarification_question = None
        state.status = "searching_index"
        search_request = user_input.strip()

    else:
        state.original_user_query = user_input

        parse_run = await turn_parser_agent.run(
            f"User input: {state.original_user_query}",
            message_history=message_history,
        )
        parsed = parse_run.output

        state.parsed_mentions = parsed.mentions
        if parsed.user_question_for_sop:
            state.pending_user_question = parsed.user_question_for_sop

        if parsed.intent == "chat":
            return ChatTurnResult(
                event_type="final_answer",
                assistant_text="Hi — I can help you find the right SOP. Tell me the company name or alternate name.",
                state=state,
                next_message_history=parse_run.all_messages(),
                raw_result=parse_run,
            )

        if not state.locked_on_sop and parsed.retrieval_status == "missing_reference":
            question = parsed.clarification_question or (
                "I can answer that once I know which SOP to use. Please provide the company name or alternate name first."
            )
            state.awaiting_user_clarification = True
            state.clarification_question = question
            state.status = "needs_clarification"

            return ChatTurnResult(
                event_type="ask_user",
                assistant_text=question,
                state=state,
                next_message_history=parse_run.all_messages(),
                raw_result=parse_run,
            )

        if not state.locked_on_sop and parsed.retrieval_status == "ambiguous":
            question = parsed.clarification_question or (
                "I found more than one plausible reference. Please clarify which company or alternate name you mean."
            )
            state.awaiting_user_clarification = True
            state.clarification_question = question
            state.status = "needs_clarification"

            return ChatTurnResult(
                event_type="ask_user",
                assistant_text=question,
                state=state,
                next_message_history=parse_run.all_messages(),
                raw_result=parse_run,
            )

        search_request = (parsed.retrieval_query_text or state.original_user_query).strip()

    while True:
        decision, candidates, judge_run = await hybrid_select_row(
            state.current_index_path,
            search_request,
        )

        if decision.action == "ask_user":
            question = decision.clarification_question or "I found multiple plausible matches. Which one did you mean?"
            state.awaiting_user_clarification = True
            state.clarification_question = question
            state.status = "needs_clarification"

            if state.in_secondary_index:
                state.has_asked_secondary_clarification = True

            next_history = judge_run.all_messages() if judge_run is not None else []

            return ChatTurnResult(
                event_type="ask_user",
                assistant_text=question,
                state=state,
                next_message_history=next_history,
                raw_result=judge_run,
            )

        if decision.action == "respond":
            next_history = judge_run.all_messages() if judge_run is not None else []
            return ChatTurnResult(
                event_type="final_answer",
                assistant_text=decision.response_text or "Hi — I can help you find the right SOP. Tell me the company name or alternate name.",
                state=state,
                next_message_history=next_history,
                raw_result=judge_run,
            )

        if decision.action == "no_match" or decision.selected_row_id is None:
            state.status = "error"
            next_history = judge_run.all_messages() if judge_run is not None else []
            return ChatTurnResult(
                event_type="error",
                assistant_text="I could not confidently identify the right row.",
                state=state,
                next_message_history=next_history,
                raw_result=judge_run,
            )

        df, _, _ = prepare_table_index(state.current_index_path)

        if not (0 <= decision.selected_row_id < len(df)):
            state.status = "error"
            next_history = judge_run.all_messages() if judge_run is not None else []
            return ChatTurnResult(
                event_type="error",
                assistant_text="The selected row id was invalid.",
                state=state,
                next_message_history=next_history,
                raw_result=judge_run,
            )

        selected_row = df.iloc[decision.selected_row_id]
        link = extract_link_from_row(selected_row)

        if not link:
            state.status = "error"
            next_history = judge_run.all_messages() if judge_run is not None else []
            return ChatTurnResult(
                event_type="error",
                assistant_text="The selected row did not contain a usable file link.",
                state=state,
                next_message_history=next_history,
                raw_result=judge_run,
            )

        resolved_link = normalize_doc_path(link)
        logger.info("Resolved selected link to %s", resolved_link)

        if resolved_link.lower().endswith((".xlsx", ".csv")):
            state.current_index_path = resolved_link
            state.status = "searching_index"
            state.in_secondary_index = True
            continue

        if resolved_link.lower().endswith(".docx"):
            state.selected_link = resolved_link
            state.active_sop_path = resolved_link
            state.locked_company_name = str(selected_row.get("company name", "")).strip() or None
            state.locked_on_sop = True
            state.status = "locked_on_sop"

            if state.pending_user_question:
                sop_result = await answer_from_sop(
                    state.pending_user_question,
                    state.active_sop_path,
                )
                assistant_text = (
                    sop_result.answer
                    + "\n\n[Locked to current SOP. Say 'switch company' to search another company.]"
                )
            else:
                label = state.locked_company_name or "the selected company"
                assistant_text = (
                    f"I found the SOP for {label} and locked onto it."
                    "\n\n[Locked to current SOP. Ask a question about this SOP or say 'switch company' to search another company.]"
                )

            next_history = judge_run.all_messages() if judge_run is not None else []

            return ChatTurnResult(
                event_type="final_answer",
                assistant_text=assistant_text,
                state=state,
                next_message_history=next_history,
                raw_result=judge_run,
            )

        state.status = "error"
        next_history = judge_run.all_messages() if judge_run is not None else []
        return ChatTurnResult(
            event_type="error",
            assistant_text="Selected row did not contain a usable DOCX, XLSX, or CSV link.",
            state=state,
            next_message_history=next_history,
            raw_result=judge_run,
        )