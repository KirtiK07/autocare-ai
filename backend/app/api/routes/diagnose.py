from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.api.deps import get_container, get_session
from app.api.routes._convert import to_prediction
from app.core.container import Container
from app.core.errors import NotFoundError, RateLimitedError
from app.db.repositories import ChatRepository, PredictionRepository
from app.schemas import Citation, DiagnoseRequest, DiagnoseResponse, SentenceCheck, Snippet
from app.services.llm import ChatTurn

router = APIRouter(tags=["assistant"])


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(
    body: DiagnoseRequest,
    request: Request,
    container: Container = Depends(get_container),
    session: Session = Depends(get_session),
) -> DiagnoseResponse:
    """Answer a question from the manuals, grounded in the given prediction."""
    client = request.client.host if request.client else "unknown"
    if not request.app.state.diagnose_limiter.allow(client):
        raise RateLimitedError("Too many questions; please wait a minute.")

    prediction, history = None, []
    chats = ChatRepository(session)
    if body.prediction_id is not None:
        row = PredictionRepository(session).get(body.prediction_id)
        if row is None:
            raise NotFoundError(f"Prediction {body.prediction_id} not found")
        prediction = to_prediction(row)
        history = [ChatTurn(m.role, m.content) for m in chats.recent(body.prediction_id)]

    result = await run_in_threadpool(container.agent.answer, body.question, prediction, history)

    message_id = None
    if body.prediction_id is not None:
        chats.add(body.prediction_id, "user", body.question)
        msg = chats.add(body.prediction_id, "assistant", result.answer, result.citations, result.supported)
        message_id = msg.id

    return DiagnoseResponse(
        answer=result.answer,
        citations=[Citation.model_validate(c) for c in result.citations],
        snippets=[
            Snippet(id=c.id, source=c.source, page=c.page, text=c.text, score=round(c.score, 3))
            for c in result.snippets
        ],
        sentences=[
            SentenceCheck(text=s.text, supported=s.supported, score=s.score) for s in result.sentences
        ],
        supported=result.supported,
        covered=result.covered,
        mode=result.mode,
        message_id=message_id,
    )
