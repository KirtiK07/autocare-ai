from app.services.diagnostic_agent import NOT_COVERED, DiagnosticAgent
from app.services.guardrail import LexicalGuardrail, NoGuardrail, split_sentences
from app.services.llm import ExtractiveClient, build_llm
from tests.conftest import OIL_CHUNK, FakeLLM, FakeRetriever


def test_split_sentences_strips_bullets() -> None:
    assert split_sentences("- One thing. Two things.\n2. Three.") == ["One thing.", "Two things.", "Three."]


def test_agent_answers_with_citations() -> None:
    agent = DiagnosticAgent(FakeRetriever(), FakeLLM(), LexicalGuardrail())
    ans = agent.answer("Why is my oil light on?")
    assert ans.covered
    assert ans.citations == [{"source": "Owner's Manual", "page": 42}]
    assert ans.snippets[0].id == "c1"


def test_agent_not_covered_without_chunks() -> None:
    agent = DiagnosticAgent(FakeRetriever([]), FakeLLM(), NoGuardrail())
    ans = agent.answer("How do I fix my brakes?")
    assert not ans.covered and ans.answer == NOT_COVERED


def test_guardrail_replaces_unsupported_answer() -> None:
    llm = FakeLLM(
        "Replace the turbocharger wastegate actuator immediately. Recalibrate the transmission solenoids."
    )
    agent = DiagnosticAgent(FakeRetriever(), llm, LexicalGuardrail())
    ans = agent.answer("What now?")
    assert ans.answer == NOT_COVERED and not ans.supported


def test_extractive_client_quotes_context() -> None:
    ctx = DiagnosticAgent.format_context([OIL_CHUNK])
    out = ExtractiveClient().complete("sys", f"QUESTION: q\n\nCONTEXT:\n{ctx}")
    assert "[Owner's Manual, page 42]" in out and "oil pressure warning light" in out


def test_build_llm_without_key_is_extractive() -> None:
    assert build_llm("groq", "", "", 5).name == "extractive"


class _GatedRetriever(FakeRetriever):
    """Returns chunks only when the query mentions oil (mimics a relevance floor)."""

    def search(self, query: str, k: int = 5) -> list:  # type: ignore[type-arg]
        self.queries.append(query)
        return self.chunks if "oil" in query.lower() else []


def test_off_topic_question_not_covered_despite_prediction_context() -> None:
    from app.schemas import FeatureContribution, Prediction

    pred = Prediction(
        probability=0.9,
        label="faulty",
        risk_band="high",
        model_version="t",
        top_features=[
            FeatureContribution(
                feature="lub_oil_pressure",
                label="Lubricant oil pressure",
                value=1.0,
                deviation=-2.0,
                direction="low",
            )
        ],
    )
    agent = DiagnosticAgent(_GatedRetriever(), FakeLLM(), NoGuardrail())
    assert not agent.answer("What is the capital of France?", pred).covered
    assert agent.answer("Why is my oil light on?", pred).covered


def test_split_keeps_trailing_citation() -> None:
    assert split_sentences("A is low. [S, page 1] B is high [S, page 2]. C.") == [
        "A is low. [S, page 1]",
        "B is high [S, page 2].",
        "C.",
    ]
