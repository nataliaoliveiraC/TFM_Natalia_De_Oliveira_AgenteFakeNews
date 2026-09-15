from tavily import TavilyClient
from pydantic import BaseModel

def search_web(
    query,
    tavily_client,
    max_results=5,
):
    """
    Realiza una búsqueda general en la web.

    Parameters
    ----------
    query : str
        Consulta que se enviará al motor de búsqueda.

    tavily_client : TavilyClient
        Cliente de Tavily previamente inicializado.

    max_results : int
        Número máximo de resultados devueltos.

    Returns
    -------
    list[dict]
        Resultados normalizados con título, URL, snippet y tipo de fuente.
    """

    response = tavily_client.search(
        query=query,
        max_results=max_results,
    )

    results = []

    for result in response.get("results", []):
        results.append(
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "snippet": result.get("content"),
                "source_type": "web",
            }
        )

    return results


FACT_CHECK_DOMAINS = [
    "reuters.com",
    "snopes.com",
    "politifact.com",
    "fullfact.org",
    "factcheck.org",
    "maldita.es",
    "newtral.es",
]

def search_fact_checks(
    query,
    tavily_client,
    max_results=5,
):
    """
    Busca verificaciones previas relacionadas con una consulta en dominios especializados en fact-checking.
    """

    response = tavily_client.search(
        query=query,
        max_results=max_results,
        include_domains=FACT_CHECK_DOMAINS,
    )

    results = []

    for result in response.get("results", []):
        results.append(
            {
                "title": result.get("title"),
                "url": result.get("url"),
                "snippet": result.get("content"),
                "source_type": "fact_check",
            }
        )

    return results


def fetch_url(
    url,
    tavily_client,
    source_type="web",
):
    """
    Recupera el contenido textual de una URL.
    """

    response = tavily_client.extract(
        urls=[url]
    )

    results = response.get("results", [])

    if not results:
        return None

    result = results[0]

    return {
        "url": result.get("url"),
        "title": result.get("title"),
        "text": result.get("raw_content"),
        "source_type": source_type,
    }

#Definiciones necesarias para el Research Agent

class ResearchPlan(BaseModel):
    """
    Plan de investigación generado para una claim.
    """

    web_queries: list[str]
    fact_check_queries: list[str]

def create_research_plan(
    claim,
    entities,
    date_reference,
    client,
    model="gpt-5.6-terra",
):
    """
    Genera un plan de investigación para una claim.
    """

    instructions = """
    You are a Research Agent for a fact-checking system.

    Your task is to generate a concise research plan for verifying a claim.

    Produce:
    - web_queries: general web search queries aimed at finding reliable and relevant information.
    - fact_check_queries: queries aimed at finding previous fact-checks related to the claim.

    The queries should:
    - preserve the meaning of the original claim
    - include important entities
    - include relevant dates when useful
    - be concise and suitable for a search engine
    - avoid unnecessary duplication

    Generate at most 3 web queries and at most 2 fact-check queries.

    Fact-check queries should describe the claim to be verified.
    Do not include the name of a specific fact-checking organization unless it is explicitly relevant to the claim.

    Do not determine whether the claim is true or false.
    Do not provide evidence.
    Do not answer the claim.
    """

    response = client.responses.parse(
        model=model,
        instructions=instructions,
        input=f"""
CLAIM:
{claim}

ENTITIES:
{entities}

DATE REFERENCE:
{date_reference}
""",
        text_format=ResearchPlan,
    )

    return response.output_parsed

def execute_research_plan(
    research_plan,
    tavily_client,
    max_results_per_query=5,
):
    """
    Ejecuta las consultas contenidas en un ResearchPlan.

    Realiza búsquedas web generales y búsquedas orientadas a fact-checking, conservando la query que originó
    cada resultado.
    """

    web_results = []
    fact_check_results = []

    for query in research_plan.web_queries:

        results = search_web(
            query=query,
            tavily_client=tavily_client,
            max_results=max_results_per_query,
        )

        for result in results:
            result = result.copy()
            result["search_query"] = query #Guardamos la query para analizar el comportamiento del agente.
            web_results.append(result)

    for query in research_plan.fact_check_queries:

        results = search_fact_checks(
            query=query,
            tavily_client=tavily_client,
            max_results=max_results_per_query,
        )

        for result in results:
            result = result.copy()
            result["search_query"] = query
            fact_check_results.append(result)

    return {
        "web_results": web_results,
        "fact_check_results": fact_check_results,
    }
#Deduplicados y seleccion de resultados relevantes.
def deduplicate_search_results(search_results):
    """
    Elimina resultados duplicados a partir de la URL.

    Conserva la primera aparición de cada URL.
    """

    unique_results = []
    seen_urls = set()

    for result in search_results:
        url = result.get("url")

        if not url:
            continue

        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    return unique_results

#Para la selección de resultados relevantes.
class SelectedSearchResult(BaseModel):
    """
    Resultado de búsqueda seleccionado como relevante para la claim.
    """

    result_id: int
    reason: str


class SearchResultSelection(BaseModel):
    """
    Selección de resultados de búsqueda relevantes.
    """

    selected_results: list[SelectedSearchResult]

def select_relevant_results(
    claim,
    web_results,
    fact_check_results,
    client,
    model="gpt-5.6-terra",
):
    """
    Selecciona los resultados de búsqueda más relevantes para verificar una claim.
    """

    candidates = web_results + fact_check_results

    formatted_results = []

    for result_id, result in enumerate(candidates):
        formatted_results.append(
            f"""
RESULT {result_id}

TITLE:
{result.get("title")}

URL:
{result.get("url")}

SNIPPET:
{result.get("snippet")}

SOURCE TYPE:
{result.get("source_type")}
"""
        )

    candidates_text = "\n".join(formatted_results)

    instructions = """
    You are part of a Research Agent for a fact-checking system.

    Your task is to select the search results that are genuinely useful for verifying the given claim.

    Select results that:
    - are directly related to the claim
    - may provide factual evidence relevant to the claim
    - contain information about the key entities, events, dates or regulations involved
    - may help support, refute or clarify the claim

    Do not select results that:
    - are only loosely related to the topic
    - refer to unrelated events or claims
    - are generic pages with little useful factual information
    - are clearly irrelevant

    You may select results from both general web search and fact-check search.

    Do not determine whether the claim is true or false.
    Do not invent URLs or sources.
    Select only from the provided result IDs.

    Select at most 6 results.
    """

    response = client.responses.parse(
        model=model,
        instructions=instructions,
        input=f"""
CLAIM:
{claim}

SEARCH RESULTS:
{candidates_text}
""",
        text_format=SearchResultSelection,
    )

    return response.output_parsed

def get_selected_results(
    selection,
    web_results,
    fact_check_results,
):
    """
    Recupera los resultados originales seleccionados por el LLM.
    """

    candidates = web_results + fact_check_results

    selected_results = []

    for selected in selection.selected_results:
        result = candidates[selected.result_id].copy()
        result["selection_reason"] = selected.reason
        selected_results.append(result)

    return selected_results


def fetch_selected_results(
    selected_results,
    tavily_client,
):
    """
    Recupera el contenido completo de los resultados seleccionados.
    """

    documents = []

    for result in selected_results:
        document = fetch_url(
            url=result["url"],
            tavily_client=tavily_client,
            source_type=result["source_type"],
        )

        if document is None:
            continue

        document["search_query"] = result.get("search_query")
        document["selection_reason"] = result.get("selection_reason")

        documents.append(document)

    return documents

#Función final orquestadora

def run_research_agent(
    claim,
    entities,
    date_reference,
    client,
    tavily_client,
    max_results_per_query=5,
):
    """
    Ejecuta el flujo completo del Research Agent para una claim.

    Genera un plan de investigación, ejecuta las búsquedas,
    elimina resultados duplicados, selecciona las fuentes
    relevantes y recupera su contenido completo.

    Devuelve los documentos candidatos que serán utilizados
    posteriormente por el componente RAG.
    """

    # 1. Generar el plan de investigación
    research_plan = create_research_plan(
        claim=claim,
        entities=entities,
        date_reference=date_reference,
        client=client,
    )

    # 2. Ejecutar las búsquedas
    research_results = execute_research_plan(
        research_plan=research_plan,
        tavily_client=tavily_client,
        max_results_per_query=max_results_per_query,
    )

    # 3. Eliminar URLs duplicadas
    unique_web_results = deduplicate_search_results(
        research_results["web_results"]
    )

    unique_fact_check_results = deduplicate_search_results(
        research_results["fact_check_results"]
    )

    # 4. Seleccionar los resultados relevantes
    selection = select_relevant_results(
        claim=claim,
        web_results=unique_web_results,
        fact_check_results=unique_fact_check_results,
        client=client,
    )

    # 5. Recuperar los resultados originales seleccionados
    selected_results = get_selected_results(
        selection=selection,
        web_results=unique_web_results,
        fact_check_results=unique_fact_check_results,
    )

    # 6. Recuperar el contenido completo de las URLs
    candidate_documents = fetch_selected_results(
        selected_results=selected_results,
        tavily_client=tavily_client,
    )

    return candidate_documents