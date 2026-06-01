from __future__ import annotations

from pathlib import Path
from typing import Literal, Any
import re
import pandas as pd
from docx import Document
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent


# -----------------------------
# CONFIG
# -----------------------------
# ROOT_INDEX_WINDOWS = r"C:\Users\maytu\OneDrive\Desktop\docs\index.xlsx"
ROOT_INDEX_PATH = "/data/docs/index.xlsx"
CONTAINER_DOCS_DIR = Path("/data/docs")

# -----------------------------
# HELPERS
# -----------------------------
def normalize_windows_path(raw_path: str) -> str:
    raw_path = str(raw_path).strip()

    filename = Path(raw_path.replace("\\", "/")).name
    if filename:
        return str(CONTAINER_DOCS_DIR / filename)

    return raw_path


# # ROOT_INDEX_PATH = normalize_windows_path(ROOT_INDEX_WINDOWS)
# ROOT_INDEX_WINDOWS = r"C:\Users\maytu\OneDrive\Desktop\docs\index.xlsx"
# ROOT_INDEX_PATH = normalize_windows_path(ROOT_INDEX_WINDOWS)

def load_small_table(index_path: str) -> pd.DataFrame:
    path = Path(index_path)
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


def dataframe_to_row_text(df: pd.DataFrame) -> str:
    lines = []
    for i, row in df.iterrows():
        cells = [f"{col}={str(row[col]).strip()}" for col in df.columns]
        lines.append(f"row_id={i} | " + " | ".join(cells))
    return "\n".join(lines)


def extract_link_from_row(row: pd.Series) -> str | None:
    for value in row.values:
        if isinstance(value, str):
            v = value.strip()
            if v.lower().endswith(".docx") or v.lower().endswith(".xlsx") or v.lower().endswith(".csv"):
                return v
    return None


def read_docx_text(path: str) -> str:
    doc = Document(path)
    return "\n".join(
        p.text.strip()
        for p in doc.paragraphs
        if p.text and p.text.strip()
    )


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


class TableDecision(BaseModel):
    action: Literal["select_row", "ask_user", "no_match"]
    selected_row_id: int | None = None
    question: str | None = None


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
match_agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=TableDecision,
    system_prompt=(
        "You are selecting a row from a small company table. "
        "Return select_row if one row clearly matches the user's request. "
        "Return ask_user with exactly one short clarification question if there is ambiguity. "
        "Return no_match if nothing fits."
    ),
)

qa_agent = Agent(
    "openai:gpt-4.1-mini",
    output_type=SopAnswer,
    system_prompt=(
        "You answer questions only from the SOP text provided. "
        "Be concise. If the answer is not in the SOP, say that clearly. "
        "Do not switch companies or speculate. Follow these rules strictly:"
        """
        Rules:
- The SOP has already been selected by the application.
- Answer only from the SOP text below.
- Do not say the SOP is missing for another company name mentioned earlier.
- Do not re-evaluate which company was intended.
- If the answer is in the SOP, give it directly.
- If the answer is not in the SOP, say: "That information is not in the selected SOP."

        """
    ),
)


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

    elif state.locked_on_sop and state.active_sop_path and not wants_new_search(user_input):
        sop_result = await answer_from_sop(user_input, state.active_sop_path)
        return ChatTurnResult(
            event_type="final_answer",
            assistant_text=sop_result.answer,
            state=state,
            raw_result=None,
        )

    elif state.locked_on_sop and wants_new_search(user_input):
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
        state.original_user_query += f"\nUser clarification: {user_input}"
        state.awaiting_user_clarification = False
        state.clarification_question = None

    else:
        state.original_user_query = user_input

    while True:
        df = load_small_table(state.current_index_path)
        table_text = dataframe_to_row_text(df)

        prompt = f"""
User query:
{state.original_user_query}

Current table:
{table_text}

Your task:
- Choose the best matching row if clear.
- If ambiguous, ask one short clarification question.
- If no row matches, return no_match.
"""

        agent_result = await match_agent.run(
            prompt,
            message_history=message_history,
        )
        decision = agent_result.output

        if decision.action == "ask_user":
            state.awaiting_user_clarification = True
            state.clarification_question = decision.question
            state.status = "needs_clarification"

            return ChatTurnResult(
                event_type="ask_user",
                assistant_text=decision.question,
                state=state,
                raw_result=agent_result,
            )

        if decision.action == "no_match":
            state.status = "error"
            return ChatTurnResult(
                event_type="error",
                assistant_text="I could not identify the right row in the table.",
                state=state,
                raw_result=agent_result,
            )

        if decision.selected_row_id is None or decision.selected_row_id >= len(df):
            state.status = "error"
            return ChatTurnResult(
                event_type="error",
                assistant_text="The model selected an invalid row.",
                state=state,
                raw_result=agent_result,
            )

        selected_row = df.iloc[decision.selected_row_id]
        link = extract_link_from_row(selected_row)

        if not link:
            state.status = "error"
            return ChatTurnResult(
                event_type="error",
                assistant_text="The selected row did not contain a usable file link.",
                state=state,
                raw_result=agent_result,
            )

        resolved_link = normalize_windows_path(link)

        if resolved_link.lower().endswith((".xlsx", ".csv")):
            state.current_index_path = resolved_link
            state.status = "searching_index"
            message_history = agent_result.all_messages()
            continue

        if resolved_link.lower().endswith(".docx"):
            state.selected_link = resolved_link
            state.active_sop_path = resolved_link
            state.locked_company_name = str(selected_row.to_dict())
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
                raw_result=agent_result,
            )

        state.status = "error"
        return ChatTurnResult(
            event_type="error",
            assistant_text="Selected row did not contain a usable DOCX, XLSX, or CSV link.",
            state=state,
            raw_result=agent_result,
        )