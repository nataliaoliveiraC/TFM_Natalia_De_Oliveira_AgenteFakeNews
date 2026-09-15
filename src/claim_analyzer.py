from typing import Optional
from pydantic import BaseModel


#Representacion de una Claim
class ClaimItem(BaseModel):
    """
    Representa una afirmación verificable extraída de una noticia.
    """

    id: int
    claim: str
    entities: list[str]
    date_reference: Optional[str]

#Agrupacion de todas las claims para una noticia
class ClaimAnalysisResult(BaseModel):
    """
    Resultado estructurado del Claim Analyzer.
    """

    claims: list[ClaimItem]


def extract_claims(
    title,
    body,
    language,
    client,
    model="gpt-5.6-terra",
):
    """
    Extrae afirmaciones verificables de una noticia.

    La función utiliza un LLM para identificar las principales claims
    factuales presentes en el título y el contenido de la noticia.
    """

    instructions = """
    You are a Claim Analyzer for a fact-checking system.

    Your task is to extract the main factual and verifiable claims
    from a news article.

    Extract only statements that could, in principle, be verified
    using external evidence.

    Do not include:
    - opinions
    - subjective evaluations
    - rhetorical statements
    - vague claims that cannot be fact-checked

    Each extracted claim must:
    - be understandable on its own
    - preserve the original meaning
    - avoid unnecessary context
    - include the relevant named entities
    - include an explicit date reference when present
    - separate independent factual assertions into distinct claims, even when they appear in the same sentence.
    - avoid combining multiple verifiable propositions into a single claim.
    - each claim should contain one main factual proposition whenever possible.
    
    Do not determine whether the claims are true or false.
    Do not search for evidence.
    """

    response = client.responses.parse(
        model=model,
        instructions=instructions,
        input=f"""
TITLE:
{title}

BODY:
{body}

LANGUAGE:
{language}
""",
        text_format=ClaimAnalysisResult,
    )

    return response.output_parsed