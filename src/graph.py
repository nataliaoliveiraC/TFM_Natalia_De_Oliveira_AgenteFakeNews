from typing import TypedDict, Optional

from src.claim_analyzer import ClaimItem, extract_claims
from src.verdict_synthesizer import (
    synthesize_verdict,
    ClaimVerificationSummary,
    FinalVerdict,
)
from src.backend import (ClaimAnalysisDetail,
                         VerifiedEvidence,
                         )
                         
from src.language_detector import detect_language

from functools import partial


from src.research_agent import run_research_agent
from src.live_retrieval import run_live_retrieval
from src.evidence_verifier import verify_evidence

from src.inference import predict_fake_news
from langgraph.graph import StateGraph, START, END

class NewsGraphState(TypedDict, total=False):
    title: str
    body: str

    language: str

    claims: list[ClaimItem]

    candidate_documents: dict[int, list[dict]]

    retrieved_evidences: dict[int, list[dict]]

    claim_verifications: list[ClaimVerificationSummary]

    claim_details: list[ClaimAnalysisDetail]

    ml_signal: Optional[dict]

    final_verdict: FinalVerdict

def detect_language_node(state: NewsGraphState):
    language = detect_language(
        state["title"],
        state["body"],
    )

    return {
        "language": language
    }

def claim_analyzer_node(
    state: NewsGraphState,
    client,
):
    claim_analysis = extract_claims(
        title=state["title"],
        body=state["body"],
        language=state["language"],
        client=client,
    )

    return {
        "claims": claim_analysis.claims
    }


# def factual_verification_node(
#     state: NewsGraphState,
#     client,
#     tavily_client,
#     embedding_model,
# ):
#     """
#     Ejecuta el flujo de verificación factual para todas las claims:
#     Research Agent -> Live RAG -> Evidence Verifier.
#     """

#     claim_summaries = []
#     claim_details = []

#     for claim_item in state["claims"]:

#         # 1. Investigación web
#         candidate_documents = run_research_agent(
#             claim=claim_item.claim,
#             entities=claim_item.entities,
#             date_reference=claim_item.date_reference,
#             client=client,
#             tavily_client=tavily_client,
#         )

#         # 2. Recuperación de evidencias
#         evidences = run_live_retrieval(
#             claim=claim_item.claim,
#             candidate_documents=candidate_documents,
#             embedding_model=embedding_model,
#         )

#         # 3. Verificación de evidencias
#         verification_result = verify_evidence(
#             claim=claim_item.claim,
#             evidences=evidences,
#             client=client,
#         )

#         # 4. Resumen de verificación de la claim
#         claim_summary = ClaimVerificationSummary(
#             claim=claim_item.claim,
#             verdict=verification_result.verdict,
#             evidence_sufficient=verification_result.evidence_sufficient,
#             explanation=verification_result.explanation,
#         )

#         claim_summaries.append(
#             claim_summary
#         )

#         # 5. Asociar las evidencias con la valoración del verifier
#         verified_evidences = []

#         for assessment in verification_result.evidence_assessments:

#             evidence = evidences[
#                 assessment.evidence_id - 1
#             ]

#             verified_evidence = VerifiedEvidence(
#                 evidence_id=assessment.evidence_id,
#                 title=evidence.get("title"),
#                 url=evidence.get("url"),
#                 text=evidence.get("text", ""),
#                 source_type=evidence.get("source_type"),
#                 score=evidence.get("score", 0.0),
#                 relation=assessment.relation,
#                 reason=assessment.reason,
#             )

#             verified_evidences.append(
#                 verified_evidence
#             )

#         # 6. Detalle completo de la claim
#         claim_detail = ClaimAnalysisDetail(
#             claim=claim_item,
#             verification=claim_summary,
#             evidences=verified_evidences,
#         )

#         claim_details.append(
#             claim_detail
#         )

#     return {
#         "claim_verifications": claim_summaries,
#         "claim_details": claim_details,
#     }

def research_agent_node(
    state: NewsGraphState,
    client,
    tavily_client,
):
    """
    Ejecuta el Research Agent para cada claim y recupera
    documentos candidatos.
    """

    candidate_documents = {}

    for claim_item in state["claims"]:

        documents = run_research_agent(
            claim=claim_item.claim,
            entities=claim_item.entities,
            date_reference=claim_item.date_reference,
            client=client,
            tavily_client=tavily_client,
        )

        candidate_documents[
            claim_item.id
        ] = documents

    return {
        "candidate_documents": candidate_documents
    }

def live_rag_node(
    state: NewsGraphState,
    embedding_model,
):
    """
    Ejecuta el pipeline Live RAG para los documentos
    recuperados de cada claim.
    """

    retrieved_evidences = {}

    for claim_item in state["claims"]:

        evidences = run_live_retrieval(
            claim=claim_item.claim,
            candidate_documents=state[
                "candidate_documents"
            ][claim_item.id],
            embedding_model=embedding_model,
        )

        retrieved_evidences[
            claim_item.id
        ] = evidences

    return {
        "retrieved_evidences": retrieved_evidences
    }

def evidence_verifier_node(
    state: NewsGraphState,
    client,
):
    """
    Verifica las evidencias recuperadas para cada claim
    y genera sus resultados de verificación.
    """

    claim_summaries = []
    claim_details = []

    for claim_item in state["claims"]:

        evidences = state[
            "retrieved_evidences"
        ][claim_item.id]

        verification_result = verify_evidence(
            claim=claim_item.claim,
            evidences=evidences,
            client=client,
            language=state["language"],
        )

        claim_summary = ClaimVerificationSummary(
            claim=claim_item.claim,
            verdict=verification_result.verdict,
            evidence_sufficient=(
                verification_result.evidence_sufficient
            ),
            explanation=verification_result.explanation,
        )

        claim_summaries.append(
            claim_summary
        )

        verified_evidences = []

        for assessment in (
            verification_result.evidence_assessments
        ):

            evidence = evidences[
                assessment.evidence_id - 1
            ]

            verified_evidence = VerifiedEvidence(
                evidence_id=assessment.evidence_id,
                title=evidence.get("title"),
                url=evidence.get("url"),
                text=evidence.get("text", ""),
                source_type=evidence.get(
                    "source_type"
                ),
                score=evidence.get(
                    "score",
                    0.0,
                ),
                relation=assessment.relation,
                reason=assessment.reason,
            )

            verified_evidences.append(
                verified_evidence
            )

        claim_detail = ClaimAnalysisDetail(
            claim=claim_item,
            verification=claim_summary,
            evidences=verified_evidences,
        )

        claim_details.append(
            claim_detail
        )

    return {
        "claim_verifications": claim_summaries,
        "claim_details": claim_details,
    }



def ml_node(
    state: NewsGraphState,
):
    """
    Ejecuta el clasificador ML para noticias en inglés.
    """

    ml_signal = predict_fake_news(
        title=state["title"],
        text=state["body"],
    )

    return {
        "ml_signal": ml_signal
    }

def verdict_synthesizer_node(
    state: NewsGraphState,
    client,
):
    """
    Genera el veredicto final de la noticia a partir de las
    verificaciones de las claims y, cuando exista, de la señal ML.
    """

    final_verdict = synthesize_verdict(
        claim_summaries=state["claim_verifications"],
        client=client,
        language=state["language"],
        ml_signal=state.get("ml_signal"),
    )

    return {
        "final_verdict": final_verdict
    }

def route_by_language(
    state: NewsGraphState,
):
    """
    Decide si debe ejecutarse la rama ML.
    """

    if state["language"] == "en":
        return "ml"

    return "no_ml"


def build_news_graph(
    client,
    tavily_client,
    embedding_model,
):
    """
    Construye y compila el grafo completo de análisis de noticias.
    """

    graph = StateGraph(
        NewsGraphState
    )

    # Nodos
    graph.add_node(
        "detect_language",
        detect_language_node,
    )

    graph.add_node(
        "claim_analyzer",
        partial(
            claim_analyzer_node,
            client=client,
        ),
    )

    # graph.add_node(
    #     "factual_verification",
    #     partial(
    #         factual_verification_node,
    #         client=client,
    #         tavily_client=tavily_client,
    #         embedding_model=embedding_model,
    #     ),
    # )

    graph.add_node(
    "research_agent",
        partial(
            research_agent_node,
            client=client,
            tavily_client=tavily_client,
        ),
    )

    graph.add_node(
        "live_rag",
        partial(
            live_rag_node,
            embedding_model=embedding_model,
        ),
    )

    graph.add_node(
        "evidence_verifier",
        partial(
            evidence_verifier_node,
            client=client,
        ),
    )

    graph.add_node(
        "ml",
        ml_node,
    )

    graph.add_node(
        "verdict_synthesizer",
        partial(
            verdict_synthesizer_node,
            client=client,
        ),
    )

    # Flujo inicial
    graph.add_edge(
        START,
        "detect_language",
    )

    graph.add_edge(
        "detect_language",
        "claim_analyzer",
    )

    graph.add_edge(
        "claim_analyzer",
        "research_agent",
    )

    graph.add_edge(
        "research_agent",
        "live_rag",
    )

    graph.add_edge(
        "live_rag",
        "evidence_verifier",
    )
    # Ruta condicional según idioma
    graph.add_conditional_edges(
        "evidence_verifier",
        route_by_language,
        {
            "ml": "ml",
            "no_ml": "verdict_synthesizer",
        },
    )

    # La rama ML continúa hacia el sintetizador
    graph.add_edge(
        "ml",
        "verdict_synthesizer",
    )

    # Fin del flujo
    graph.add_edge(
        "verdict_synthesizer",
        END,
    )

    return graph.compile()