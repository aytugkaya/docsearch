# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.workflow import handle_turn

app = FastAPI(title="SOP Bot")

SESSIONS = {}

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    event_type: str
    workflow_state: dict

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid4())

    session = SESSIONS.get(session_id, {
        "workflow_state": None,
        "message_history": [],
    })

    result = await handle_turn(
        user_input=req.message,
        state=session["workflow_state"],
        message_history=session["message_history"],
    )

    session["workflow_state"] = result.state

    if result.raw_result is not None:
        try:
            session["message_history"] = result.raw_result.all_messages()
        except Exception:
            session["message_history"] = []

    SESSIONS[session_id] = session

    return ChatResponse(
        session_id=session_id,
        reply=result.assistant_text,
        event_type=result.event_type,
        workflow_state=result.state.model_dump(),
    )

@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"ok": True}

@app.get("/health")
async def health():
    return {"status": "ok"}