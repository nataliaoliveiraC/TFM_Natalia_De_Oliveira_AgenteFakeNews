from src.language_detector import detect_language
from src.claim_analyzer import extract_claims
from src.research_agent import run_research_agent
from src.live_retrieval import run_live_retrieval
from src.evidence_verifier import verify_evidence
from src.verdict_synthesizer import (
    ClaimVerificationSummary,
    FinalVerdict,
    synthesize_verdict,
)


from typing import Optional, Literal
from pydantic import BaseModel

from src.claim_analyzer import ClaimItem
from src.inference import predict_fake_news

class VerifiedEvidence(BaseModel):
    """
    Evidencia recuperada por el RAG junto con la valoración realizada por el Evidence Verifier.
    """

    evidence_id: int
    title: Optional[str]
    url: Optional[str]
    text: str
    source_type: Optional[str]
    score: float

    relation: Literal[
        "SUPPORTS",
        "REFUTES",
        "NEUTRAL",
    ]

    reason: str


class ClaimAnalysisDetail(BaseModel):
    """
    Resultado completo del análisis factual de una claim.
    """

    claim: ClaimItem
    verification: ClaimVerificationSummary
    evidences: list[VerifiedEvidence]

class NewsAnalysisResult(BaseModel):
    """
    Resultado completo del análisis end-to-end de una noticia.
    """

    language: Literal["en", "es"]
    claims: list[ClaimItem]
    claim_verifications: list[ClaimVerificationSummary]
    claim_details: list[ClaimAnalysisDetail]
    ml_signal: Optional[dict]
    final_verdict: FinalVerdict

def analyze_news(
    title,
    body,
    client,
    tavily_client,
    embedding_model,
    ):
    """
    Ejecuta el pipeline end-to-end de análisis de una noticia.
    """

    # 1. Detectar idioma
    language = detect_language(
        title,
        body,
    )

    # 2. Extraer claims
    claim_analysis = extract_claims(
        title=title,
        body=body,
        language=language,
        client=client,
    )

    claim_summaries = []
    claim_details = []

    # 3. Verificar cada claim
    for claim_item in claim_analysis.claims:

        candidate_documents = run_research_agent(
            claim=claim_item.claim,
            entities=claim_item.entities,
            date_reference=claim_item.date_reference,
            client=client,
            tavily_client=tavily_client,
        )

        evidences = run_live_retrieval(
            claim=claim_item.claim,
            candidate_documents=candidate_documents,
            embedding_model=embedding_model,
        )

        verification_result = verify_evidence(
            claim=claim_item.claim,
            evidences=evidences,
            client=client,
        )

        claim_summary = ClaimVerificationSummary(
            claim=claim_item.claim,
            verdict=verification_result.verdict,
            evidence_sufficient=verification_result.evidence_sufficient,
            explanation=verification_result.explanation,
        )

        claim_summaries.append(
            claim_summary
        )

        # 3.1. Asociar cada evidencia recuperada con
        # la valoración realizada por el Evidence Verifier
        verified_evidences = []

        for assessment in verification_result.evidence_assessments:

            evidence = evidences[
                assessment.evidence_id - 1
            ]

            verified_evidence = VerifiedEvidence(
                evidence_id=assessment.evidence_id,
                title=evidence.get("title"),
                url=evidence.get("url"),
                text=evidence.get("text", ""),
                source_type=evidence.get("source_type"),
                score=evidence.get("score", 0.0),
                relation=assessment.relation,
                reason=assessment.reason,
            )

            verified_evidences.append(
                verified_evidence
            )

        # 3.2. Guardar el detalle completo de la claim
        claim_detail = ClaimAnalysisDetail(
            claim=claim_item,
            verification=claim_summary,
            evidences=verified_evidences,
        )

        claim_details.append(
            claim_detail
        )

    # 4. Señal auxiliar ML
    ml_signal = None

    if language == "en":
        ml_signal = predict_fake_news(
            title=title,
            text=body,
        )

    # 5. Veredicto final
    final_verdict = synthesize_verdict(
        claim_summaries=claim_summaries,
        client=client,
        language=language,
        ml_signal=ml_signal,
    )

    # 6. Resultado estructurado
    return NewsAnalysisResult(
        language=language,
        claims=claim_analysis.claims,
        claim_verifications=claim_summaries,
        claim_details=claim_details,
        ml_signal=ml_signal,
        final_verdict=final_verdict,
    )