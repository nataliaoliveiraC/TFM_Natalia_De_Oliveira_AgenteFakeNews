from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

def load_embedding_model(
    model_name=EMBEDDING_MODEL_NAME,
):
    """
    Carga el modelo multilingüe utilizado para generar embeddings de claims y fragmentos documentales.
    """

    return SentenceTransformer(model_name)


def clean_document_text(text):
    """
    Realiza una limpieza básica y conservadora del texto
    recuperado desde una fuente web.
    """

    if not text:
        return ""

    # Eliminar imágenes en formato Markdown
    text = re.sub(
        r"!\[[^\]]*\]\([^)]+\)",
        " ",
        text,
    )

    # Conservar el texto de los enlaces Markdown,
    # eliminando únicamente la URL
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Normalizar espacios y saltos de línea
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def prepare_live_documents(candidate_documents):
    """
    Prepara los documentos recuperados por el Research Agent para su posterior procesamiento mediante RAG.
    """

    documents = []

    for document in candidate_documents:
        cleaned_text = clean_document_text(
            document.get("text")
        )

        if not cleaned_text:
            continue

        documents.append(
            {
                "title": document.get("title"),
                "url": document.get("url"),
                "text": cleaned_text,
                "source_type": document.get("source_type"),
                "search_query": document.get("search_query"),
            }
        )

    return documents


def chunk_live_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=150,
):
    """
    Divide los documentos preparados en fragmentos manteniendo
    los metadatos de la fuente original.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for document in documents:
        text_chunks = text_splitter.split_text(
            document["text"]
        )

        for chunk_index, text in enumerate(text_chunks):
            chunks.append(
                {
                    "text": text,
                    "title": document.get("title"),
                    "url": document.get("url"),
                    "source_type": document.get("source_type"),
                    "search_query": document.get("search_query"),
                    "chunk_index": chunk_index,
                }
            )

    return chunks


def embed_chunks(
    chunks,
    embedding_model,
):
    """
    Genera embeddings normalizados para los chunks documentales.
    """

    passages = [
        f"passage: {chunk['text']}"
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        passages,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings


def build_faiss_index(chunk_embeddings):
    """
    Construye un índice FAISS a partir de embeddings normalizados.
    """

    embeddings = np.asarray(
        chunk_embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index

def retrieve_live_evidence(
    claim,
    chunks,
    faiss_index,
    embedding_model,
    top_k=10,
    candidate_k=30,
    max_chunks_per_document=2,
):
    """
    Recupera los chunks más relevantes para una claim aplicando
    una restricción de diversidad documental.
    """

    query_embedding = embedding_model.encode(
        [f"query: {claim}"],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    candidate_k = min(
        candidate_k,
        faiss_index.ntotal,
    )

    scores, indices = faiss_index.search(
        query_embedding,
        candidate_k,
    )

    retrieved_evidence = []
    chunks_per_document = {}

    for score, index in zip(scores[0], indices[0]):

        chunk = chunks[index]
        url = chunk.get("url")

        current_count = chunks_per_document.get(
            url,
            0,
        )

        if current_count >= max_chunks_per_document:
            continue

        evidence = chunk.copy()
        evidence["score"] = float(score)

        retrieved_evidence.append(evidence)

        chunks_per_document[url] = current_count + 1

        if len(retrieved_evidence) >= top_k:
            break

    return retrieved_evidence

def run_live_retrieval(
    claim,
    candidate_documents,
    embedding_model,
    top_k=10,
    candidate_k=30,
    max_chunks_per_document=2,
):
    """
    Ejecuta el pipeline completo de recuperación de evidencias para una claim en modo live.

    Prepara los documentos, genera los chunks y sus embeddings, construye el índice FAISS y recupera las 
    evidencias más relevantes aplicando diversidad documental.
    """

    # 1. Preparar y limpiar documentos
    prepared_documents = prepare_live_documents(
        candidate_documents
    )

    # 2. Dividir los documentos en chunks
    chunks = chunk_live_documents(
        prepared_documents
    )

    # 3. Generar embeddings de los chunks
    chunk_embeddings = embed_chunks(
        chunks=chunks,
        embedding_model=embedding_model,
    )

    # 4. Construir el índice FAISS
    faiss_index = build_faiss_index(
        chunk_embeddings
    )

    # 5. Recuperar las evidencias más relevantes
    retrieved_evidence = retrieve_live_evidence(
        claim=claim,
        chunks=chunks,
        faiss_index=faiss_index,
        embedding_model=embedding_model,
        top_k=top_k,
        candidate_k=candidate_k,
        max_chunks_per_document=max_chunks_per_document,
    )

    return retrieved_evidence