from typing import Literal
from pydantic import BaseModel

class EvidenceAssessment(BaseModel):
    """
    Evaluación de una evidencia concreta respecto a la claim.
    """

    evidence_id: int

    relation: Literal[
        "SUPPORTS",
        "REFUTES",
        "NEUTRAL"
    ]

    reason: str


class VerificationResult(BaseModel):
    """
    Resultado global de la verificación factual de una claim utilizando exclusivamente las evidencias 
    recuperadas.
    """

    verdict: Literal[
        "SUPPORTED",
        "REFUTED",
        "NOT_ENOUGH_EVIDENCE",
        "CONFLICTING_EVIDENCE"
    ]

    evidence_sufficient: bool

    evidence_assessments: list[EvidenceAssessment]

    explanation: str

def format_evidences_for_llm(evidences):
    """
    Convierte las evidencias recuperadas por el RAG en un texto structurado que pueda ser utilizado por el 
    Evidence Verifier.

    No se incluyen los scores de similitud para evitar que el LLM interprete relevancia semántica como grado
    de veracidad.

    Parameters
    ----------
    evidences : list[dict]
        Lista de evidencias devuelta por retrieve_evidence().

    Returns
    -------
    str
        Evidencias numeradas con su URL y texto.
    """

    formatted_evidences = []

    for evidence_id, evidence in enumerate(evidences, start=1):

        formatted_evidences.append(
            f"""
            EVIDENCE {evidence_id}
            URL: {evidence["url"]}
            TEXT:
            {evidence["text"]}
            """.strip()
            )

    return "\n\n".join(formatted_evidences)

def verify_evidence(
    claim,
    evidences,
    client,
    model="gpt-5.6-terra",
    language=None,
):
    """
    Verifica una claim utilizando exclusivamente las evidencias recuperadas por el sistema RAG.

    Parameters
    ----------
    claim : str
        Afirmación que se quiere verificar.

    evidences : list[dict]
        Evidencias recuperadas por retrieve_evidence().

    client
        Cliente de OpenAI.

    model : str
        Modelo LLM utilizado para realizar la verificación.

    Returns
    -------
    VerificationResult
        Resultado estructurado de la verificación factual.
    """

    # Preparamos las evidencias recuperadas en un formato comprensible para el LLM
    evidence_text = format_evidences_for_llm(evidences)

    if language == "es":
        language_instruction = (
            "Write the explanation and reasons in Spanish."
        )
    elif language == "en":
        language_instruction = (
            "Write the explanation and reasons in English."
        )

    instructions = f"""
    You are an Evidence Verifier for a fact-checking system.

    Your task is to verify a claim using ONLY the evidence provided to you.

    Do not use external knowledge.
    Do not assume facts that are not present in the evidence.

    A claim may contain multiple factual components.
    When this happens, consider the different components of the claim before assigning evidence relationships and the 
    final verdict.

    For each evidence item, determine its relationship to the claim:

    - SUPPORTS:
    The evidence provides factual information that supports the claim as a whole, or supports an important component without
    contradicting or leaving unresolved another essential component of the claim.

    - REFUTES:
    The evidence provides factual information that contradicts the claim as a whole, or directly contradicts an essential 
    component in a way that makes the overall claim false.

    - NEUTRAL:
    The evidence is related to the claim but does not provide enough information to support or refute it as a whole. 
    Use NEUTRAL when the evidence addresses only one component of a multi-part claim and leaves other essential
    components unresolved.

    Then determine the overall verdict:

    - SUPPORTED:
    The available evidence is sufficient to support the claim as a whole.

    - REFUTED:
    The available evidence is sufficient to refute the claim as a whole.

    - NOT_ENOUGH_EVIDENCE:
    The available evidence does not contain enough information to reach a justified conclusion about the claim.

    - CONFLICTING_EVIDENCE:
    The available evidence supports some important factual aspects of the claim while refuting, contradicting, or 
    substantially qualifying other important aspects, so that labeling the whole claim simply SUPPORTED or REFUTED would
    lose relevant information.

    Use this verdict also for cherry-picking cases, where the claim relies on factually supported information but 
    selectively combines, omits, or frames evidence in a way that produces a misleading overall conclusion.

    Do not automatically label a multi-part claim REFUTED simply because one component is contradicted if other important 
    components are supported.
    In such cases, consider whether CONFLICTING_EVIDENCE is more appropriate.

    Set evidence_sufficient to true when the provided evidence contains enough factual information to justify the selected
    verdict, including when the evidence is sufficient to establish that the case is CONFLICTING_EVIDENCE.

    Set evidence_sufficient to false when the appropriate verdict is
    NOT_ENOUGH_EVIDENCE.

    Assess every evidence item provided.

    Base the final verdict on the content of the evidence, not on retrieval similarity scores.

    {language_instruction}
    """

    response = client.responses.parse(
        model=model,
        instructions=instructions,
        input=f"""
    CLAIM:
    {claim}

    EVIDENCE:
    {evidence_text}
    """,
            text_format=VerificationResult,
        )

    return response.output_parsed