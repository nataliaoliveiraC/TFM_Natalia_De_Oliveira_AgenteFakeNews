from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import build_news_graph
from src.live_retrieval import load_embedding_model

from openai import OpenAI
from tavily import TavilyClient

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT/".env")

import streamlit as st


st.set_page_config(
    page_title="Fake News Verification System - LangGraph",
    page_icon="app/assets/lupa_icon.png",
    layout="wide",
)


@st.cache_resource
def get_resources():
    """
    Carga una única vez los recursos necesarios para ejecutar el sistema de análisis.
    """

    client = OpenAI()

    tavily_client = TavilyClient()

    embedding_model = load_embedding_model()

    news_graph = build_news_graph(
        client=client,
        tavily_client=tavily_client,
        embedding_model=embedding_model,
    )

    return news_graph


news_graph = get_resources()


st.title("Fake News Verification System")

st.write(
    """
    Sistema de verificación de noticias basado en extracción de claims,
    búsqueda de evidencias en la web, RAG y verificación factual. Para noticias en inglés, 
    incorpora además una señal auxiliar de Machine Learning.
    """
)


# title = st.text_input(
#     "Título de la noticia",
#     placeholder="Introduce el título de la noticia...",
# )


# body = st.text_area(
#     "Contenido de la noticia",
#     placeholder="Introduce el contenido completo de la noticia...",
#     height=300,
# )


# analyze_button = st.button(
#     "Analizar noticia",
#     type="primary",
# )

with st.form("news_form"):

    title = st.text_input(
        "Título de la noticia",
        placeholder="Introduce el título de la noticia...",
    )

    body = st.text_area(
        "Contenido de la noticia",
        placeholder="Introduce el contenido completo de la noticia...",
        height=250,
    )

    analyze_button = st.form_submit_button(
        "Analizar noticia",
        type="primary",
    )

NODE_LABELS = {
    "detect_language": "Detectando idioma...",
    "claim_analyzer": "Extrayendo afirmaciones verificables (claims)...",
    "research_agent": "Buscando fuentes y evidencias...",
    "live_rag": "Recuperando las evidencias más relevantes...",
    "evidence_verifier": "Verificando las evidencias...",
    "ml": "Ejecutando el clasificador de Machine Learning...",
    "verdict_synthesizer": "Generando el veredicto final...",
}


if analyze_button:

    if not title.strip() or not body.strip():

        st.warning(
            "Introduce tanto el título como el contenido de la noticia."
        )

    else:

        input_state = {
            "title": title,
            "body": body,
        }

        result = {}

        with st.status(
            "Analizando noticia...",
            expanded=True
        ) as status:

            for event in news_graph.stream(
                input_state,
                stream_mode="updates",
            ):

                for node_name, update in event.items():

                    message = NODE_LABELS.get(
                        node_name,
                        f"Ejecutando: {node_name}"
                    )

                    st.write(message)

                    result.update(update)

            status.update(
                label="Análisis completado.",
                state="complete",
                expanded=False,
            )
        verdict = result["final_verdict"].verdict

        if verdict == "SUPPORTED":
            st.success("Veredicto final: SUPPORTED")

        elif verdict == "REFUTED":
            st.error("Veredicto final: REFUTED")

        elif verdict == "NOT_ENOUGH_EVIDENCE":
            st.warning("Veredicto final: NOT ENOUGH EVIDENCE")

        elif verdict == "CONFLICTING_EVIDENCE":
            st.warning("Veredicto final: CONFLICTING EVIDENCE")
            
        col1, col2 = st.columns(2)

        col1.metric(
            "Idioma",
            result["language"].upper(),
        )

        col2.metric(
            "Claims analizados",
            len(result["claims"]),
        )

        # col3.metric(
        #     "Veredicto",
        #     result["final_verdict"].verdict,
        # )

        st.subheader("Explicación final")

        st.write(
            result["final_verdict"].explanation
        )


        # Señal auxiliar de Machine Learning
        if result.get("ml_signal") is not None:

            st.subheader("Señal auxiliar de Machine Learning")

            ml_signal = result["ml_signal"]

            ml_col1, ml_col2 = st.columns(2)

            ml_col1.metric(
                "Predicción ML",
                ml_signal["prediction"],
            )

            ml_col2.metric(
                "Decision score",
                f"{ml_signal['decision_score']:.3f}",
            )

            st.caption(
                "Señal auxiliar basada en características textuales. "
                "El decision score indica la posición del ejemplo "
                "respecto a la frontera de decisión del clasificador SVM. "
                "No representa una probabilidad."
            )

        st.info(
            "A continuación se muestra el detalle del análisis realizado: "
            "las afirmaciones extraídas, las evidencias recuperadas y las fuentes "
            "utilizadas para justificar el veredicto obtenido por el sistema."
        )
        st.subheader("Claims extraídos")

        # for claim in result["claims"]:
        #     st.write(f"- {claim.claim}")

        for i, claim in enumerate(
            result["claims"],
            start=1
        ):
            st.write(
                f"**Claim {i}.** {claim.claim}"
            )

        st.subheader("Verificación por claim")

        for i, detail in enumerate(
            result["claim_details"],
            start=1
        ):

            claim_text = detail.claim.claim
            claim_verdict = detail.verification.verdict

            with st.expander(
                f"Claim {i} — {claim_verdict}"
            ):

                st.markdown("**Afirmación**")
                st.write(
                    claim_text
                )

                st.markdown("**Explicación**")
                st.write(
                    detail.verification.explanation
                )

                st.markdown(
                    "**Fuentes utilizadas para la verificación**"
                )

                for j, evidence in enumerate(
                    detail.evidences,
                    start=1
                ):

                    st.markdown(
                        f"**Fuente {j}: "
                        f"{evidence.title or 'Sin título'}**"
                    )

                    st.write(
                        f"Relación con el claim: "
                        f"{evidence.relation}"
                    )

                    st.write(
                        f"Motivo: {evidence.reason}"
                    )

                    if evidence.url:
                        st.markdown(
                            f"[Abrir fuente original]"
                            f"({evidence.url})"
                        )

                    with st.expander(
                        "Ver fragmento recuperado"
                    ):
                        st.text(
                            evidence.text
                        )

                    st.divider()