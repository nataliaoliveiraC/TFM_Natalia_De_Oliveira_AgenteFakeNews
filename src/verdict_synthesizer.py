from pydantic import BaseModel
from typing import Literal


class FinalVerdict(BaseModel):
    verdict: Literal[
        "SUPPORTED",
        "REFUTED",
        "NOT_ENOUGH_EVIDENCE",
        "CONFLICTING_EVIDENCE",
    ]
    explanation: str

class ClaimVerificationSummary(BaseModel):
    """
    Resumen del resultado de verificación de una claim.
    """

    claim: str
    verdict: Literal[
        "SUPPORTED",
        "REFUTED",
        "NOT_ENOUGH_EVIDENCE",
        "CONFLICTING_EVIDENCE",
    ]
    evidence_sufficient: bool
    explanation: str

def synthesize_verdict(
    claim_summaries,
    client,
    language,
    ml_signal=None,
    model="gpt-5.6-terra",
):
    """
    Genera un veredicto final a nivel de noticia a partir de los resultados
    de verificación de sus claims y, opcionalmente, de una señal auxiliar
    proporcionada por el modelo de Machine Learning.

    La explicación final se genera en el idioma original de la noticia.
    """

    if language == "es":
        output_language_instruction = (
            "Write the final explanation in Spanish."
        )

        if ml_signal is not None:
            raise ValueError(
                "ml_signal debe ser None para noticias en español."
            )

    elif language == "en":
        output_language_instruction = (
            "Write the final explanation in English."
        )

    else:
        raise ValueError(
            f"Idioma no soportado: {language}"
        )

    instructions = f"""
    You are a Verdict Synthesizer for a fact-checking system.

    Your task is to produce a final verdict for a news article based on the verification results of its
    individual claims.

    Each claim includes:
    - the claim text
    - its factual verdict
    - whether the available evidence was sufficient
    - an explanation from the Evidence Verifier

    An optional machine learning signal may also be provided for English-language articles.

    The machine learning signal is an auxiliary prediction based on the textual characteristics
    of the full article.

    Use the factual verification results as the primary basis for the final verdict.

    Important rules:
    - Do not perform new fact-checking.
    - Do not invent new evidence.
    - Do not reinterpret external sources.
    - Base the synthesis on the provided claim verification results and, when available, the auxiliary ML signal.
    - Claims with insufficient evidence must not be treated as proven true or false.
    - If different claims have different verdicts, explain the relevant distinction.
    - The final verdict should represent the overall factual status of the article.
    - The machine learning signal is secondary to factual verification.
    - Do not override sufficient factual evidence with the machine learning prediction.
    - If factual evidence clearly supports or refutes the article, prioritize the factual evidence.
    - Use the machine learning signal only as complementary information.
    - The ML decision score is not a probability and must not be interpreted as one.
    - {output_language_instruction}

    Available final verdicts:
    - SUPPORTED
    - REFUTED
    - NOT_ENOUGH_EVIDENCE
    - CONFLICTING_EVIDENCE
    """

    claims_text = "\n\n".join(
        [
            f"""
CLAIM {index}:
{summary.claim}

VERDICT:
{summary.verdict}

EVIDENCE SUFFICIENT:
{summary.evidence_sufficient}

EXPLANATION:
{summary.explanation}
"""
            for index, summary in enumerate(
                claim_summaries,
                start=1,
            )
        ]
    )

    if ml_signal is not None:
        ml_text = f"""
MACHINE LEARNING SIGNAL:

PREDICTION:
{ml_signal.get("prediction")}

LABEL:
{ml_signal.get("label")}

DECISION SCORE:
{ml_signal.get("decision_score")}
"""
    else:
        ml_text = """
MACHINE LEARNING SIGNAL:
Not available.
"""

    response = client.responses.parse(
        model=model,
        instructions=instructions,
        input=f"""
CLAIM VERIFICATION RESULTS:

{claims_text}

{ml_text}
""",
        text_format=FinalVerdict,
    )

    return response.output_parsed