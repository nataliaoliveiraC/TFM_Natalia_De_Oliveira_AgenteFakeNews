import json
import zipfile

import faiss
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text_parts(text_parts):
    """
    Limpia los fragmentos de texto extraídos de una fuente.

    Elimina valores no textuales, strings vacíos y fragmentos exactamente duplicados, manteniendo el orden 
    original.

    Parameters
    ----------
    text_parts : list
        Lista de fragmentos de texto.

    Returns
    -------
    list
        Lista de fragmentos limpios y sin duplicados exactos.
    """
    cleaned_parts = []
    seen = set()

    for part in text_parts:
        if not isinstance(part, str):
            continue

        part = part.strip()

        if not part:
            continue

        if part in seen:
            continue

        seen.add(part)
        cleaned_parts.append(part)

    return cleaned_parts


def build_document(record):
    """
    Convierte un registro del Knowledge Store de AVeriTeC en una estructura documental utilizable por el 
    pipeline.

    Parameters
    ----------
    record : dict
        Registro original del Knowledge Store.

    Returns
    -------
    dict
        Documento normalizado con texto y metadatos.
    """
    text_parts = clean_text_parts(
        record.get("url2text", [])
    )

    text = " ".join(text_parts)

    return {
        "claim_id": record.get("claim_id"),
        "url": record.get("url"),
        "source_type": record.get("type"),
        "query": record.get("query"),
        "text_parts": text_parts,
        "text": text,
    }


def load_knowledge_records(knowledge_zip, claim_id):
    """
    Carga los registros del Knowledge Store asociados a una claim.

    La función busca el archivo correspondiente a la claim independientemente de la carpeta interna del ZIP, de
    modo que puede utilizarse tanto con los Knowledge Stores de train como con el de dev.

    Parameters
    ----------
    knowledge_zip : str or Path
        Ruta al archivo ZIP del Knowledge Store.

    claim_id : int
        Identificador de la claim.

    Returns
    -------
    list
        Lista de registros asociados a la claim.
    """
    target_file = f"{claim_id}.json"

    knowledge_records = []

    with zipfile.ZipFile(knowledge_zip, "r") as zip_ref:

        matching_files = [
            name
            for name in zip_ref.namelist()
            if name.rsplit("/", 1)[-1] == target_file
        ]

        if not matching_files:
            raise FileNotFoundError(
                f"No se encontró {target_file} dentro de {knowledge_zip}"
            )

        file_name = matching_files[0]

        with zip_ref.open(file_name) as f:
            for line in f:
                line = line.decode("utf-8").strip()

                if line:
                    knowledge_records.append(
                        json.loads(line)
                    )

    return knowledge_records

def deduplicate_documents_by_url(documents):
    """
    Elimina documentos duplicados que apuntan a la misma URL.

    Se conserva únicamente la primera aparición de cada URL.

    Parameters
    ----------
    documents : list
        Lista de documentos normalizados.

    Returns
    -------
    list
        Lista de documentos sin URLs duplicadas.
    """
    documents_by_url = {}
  
    for document in documents:
        url = document["url"]

        if url not in documents_by_url:
            documents_by_url[url] = document
            continue

        current_document = documents_by_url[url]

        if len(document["text"]) > len(current_document["text"]):
            documents_by_url[url] = document

    return list(documents_by_url.values())


def select_relevant_documents(
    claim,
    documents,
    top_n=100,
):
    """
    Preselecciona los documentos más relacionados con una afirmación mediante similitud TF-IDF.

    Esta etapa se utiliza como filtro previo al chunking y a la generación de embeddings, con el objetivo de 
    reducir el coste computacional del retrieval semántico.

    Parameters
    ----------
    claim : str
        Afirmación para la que se quieren recuperar evidencias.

    documents : list[dict]
        Lista completa de documentos asociados al knowledge store de la claim.

    top_n : int, default=100
        Número máximo de documentos que se conservarán después
        de la preselección.

    Returns
    -------
    list[dict]
        Documentos seleccionados y ordenados de mayor a menor
        similitud TF-IDF con la afirmación.
    """
    document_texts = [
        document["text"]
        for document in documents
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
    )
    # Representamos la claim y los documentos en el mismo
    # espacio TF-IDF
    tfidf_matrix = vectorizer.fit_transform(
        [claim] + document_texts
    )

    claim_vector = tfidf_matrix[0]
    document_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(
        claim_vector,
        document_vectors,
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    selected_documents = []

    for idx in ranked_indices[:top_n]:
        document = documents[int(idx)].copy()

        document["document_score"] = float(
            similarities[idx]
        )

        selected_documents.append(document)

    return selected_documents

def build_claim_index(
    claim,
    knowledge_records,
    embedding_model,
    text_splitter,
    max_chunks_before_filter=40000,
    top_n_documents=100,
):
    """
    Construye el índice vectorial de una claim de AVeriTeC.

    El proceso:
    1. transforma los registros del Knowledge Store en documentos;
    2. elimina documentos vacíos;
    3. elimina URLs duplicadas;
    4. realiza un chunking inicial;
    5. aplica una preselección TF-IDF si el número de chunks
       supera el umbral definido;
    6. genera los chunks definitivos;
    7. calcula embeddings;
    8. construye un índice FAISS.

    Parameters
    ----------
    claim : str
        Afirmación para la que se quiere construir el índice.

    knowledge_records : list[dict]
        Registros del knowledge store asociados a la claim.

    embedding_model
        Modelo SentenceTransformer utilizado para generar embeddings.

    text_splitter
        Objeto utilizado para dividir los documentos en chunks.

    max_chunks_before_filter : int, default=40000
        Número máximo de chunks que se permite procesar directamente.
        Si se supera este valor, se aplica una preselección TF-IDF de documentos.

    top_n_documents : int, default=100
        Número de documentos que se conservarán cuando sea necesario aplicar la preselección TF-IDF.

    Returns
    -------
    chunks : list[dict]
        Lista final de chunks utilizados para construir el índice.

    index
        Índice FAISS que contiene los embeddings de los chunks.

    info : dict
        Información sobre el proceso de construcción del índice, incluyendo número de documentos, número de 
        chunks y si se utilizó preselección TF-IDF.
    """

    # 1. Conservamos únicamente los registros que contienen texto
    retrieval_records = [
        record
        for record in knowledge_records
        if record.get("url2text")
    ]

    # 2. Construimos los documentos
    documents = [
        build_document(record)
        for record in retrieval_records
    ]

    original_documents = len(documents)

    # 3. Eliminamos URLs duplicadas
    documents = deduplicate_documents_by_url(
        documents
    )

    unique_documents = len(documents)

    # 4. Generamos inicialmente los chunks para conocer el tamaño real del corpus
    chunks = []

    for document_id, document in enumerate(documents):

        split_texts = text_splitter.split_text(
            document["text"]
        )

        for chunk_id, chunk_text in enumerate(split_texts):

            chunks.append({
                "claim_id": document["claim_id"],
                "document_id": document_id,
                "chunk_id": chunk_id,
                "url": document["url"],
                "source_type": document["source_type"],
                "query": document["query"],
                "text": chunk_text,
            })

    initial_chunks = len(chunks)

    # 5. Comprobamos si el corpus es demasiado grande
    use_tfidf_filter = (
        initial_chunks > max_chunks_before_filter
    )

    if use_tfidf_filter:

        print(
            f"Corpus grande detectado: {initial_chunks} chunks."
        )

        print(
            f"Se aplicará TF-IDF para seleccionar "
            f"{top_n_documents} documentos."
        )

        # Seleccionamos los documentos más relacionados
        documents = select_relevant_documents(
            claim=claim,
            documents=documents,
            top_n=top_n_documents
        )

        # Volvemos a crear los chunks,
        # esta vez únicamente con los documentos seleccionados
        chunks = []

        for document_id, document in enumerate(documents):

            split_texts = text_splitter.split_text(
                document["text"]
            )

            for chunk_id, chunk_text in enumerate(split_texts):

                chunks.append({
                    "claim_id": document["claim_id"],
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "url": document["url"],
                    "source_type": document["source_type"],
                    "query": document["query"],
                    "text": chunk_text,
                })

    else:

        print(
            f"Corpus manejable: {initial_chunks} chunks."
        )

        print(
            "No es necesario aplicar preselección TF-IDF."
        )

    final_chunks = len(chunks)

    print("Documentos originales:", original_documents)
    print("Documentos únicos:", unique_documents)
    print("Documentos finales:", len(documents))
    print("Chunks iniciales:", initial_chunks)
    print("Chunks finales:", final_chunks)

    # 6. Extraemos únicamente el texto de los chunks
    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # 7. Generamos los embeddings SOLO del corpus final
    chunk_embeddings = embedding_model.encode(
        chunk_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    chunk_embeddings = chunk_embeddings.astype(
        "float32"
    )

    # 8. Construimos el índice FAISS
    embedding_dim = chunk_embeddings.shape[1]

    index = faiss.IndexFlatIP(
        embedding_dim
    )

    index.add(
        chunk_embeddings
    )

    # Información útil para analizar posteriormente el proceso
    info = {
        "original_documents": original_documents,
        "unique_documents": unique_documents,
        "final_documents": len(documents),
        "initial_chunks": initial_chunks,
        "final_chunks": final_chunks,
        "tfidf_filter_used": use_tfidf_filter,
    }

    return chunks, index, info


def retrieve_evidence(
    claim,
    embedding_model,
    index,
    chunks,
    candidate_k=50,
    final_k=10,
    max_chunks_per_document=2,
):
    """
    Recupera evidencias semánticamente relevantes para una afirmación, aplicando además una restricción de
    diversidad por documento.

    Parameters
    ----------
    claim : str
        Afirmación que se quiere verificar.

    embedding_model
        Modelo de SentenceTransformer utilizado para generar el embedding de la afirmación.

    index
        Índice FAISS que contiene los embeddings de los chunks.

    chunks : list[dict]
        Lista de chunks con su texto y metadatos.
        El orden de esta lista debe coincidir con el orden de los vectores almacenados en el índice FAISS.

    candidate_k : int, default=50
        Número de candidatos iniciales recuperados mediante FAISS antes de aplicar la restricción de 
        diversidad.

    final_k : int, default=10
        Número máximo de chunks de evidencia que se devolverán finalmente.

    max_chunks_per_document : int, default=2
        Número máximo de chunks que se permite seleccionar procedentes del mismo documento.

    Returns
    -------
    list[dict]
        Lista de evidencias recuperadas. Para cada chunk se devuelve su score de similitud, identificador 
        del documento, identificador del chunk, URL, tipo de fuente y texto.
    """

    # 1. Generamos el embedding de la afirmación
    query_embedding = embedding_model.encode(
        claim,
        normalize_embeddings=True
    )

    # FAISS espera una matriz bidimensional:
    # (número de consultas, dimensión del embedding)
    query_embedding = (
        query_embedding
        .astype("float32")
        .reshape(1, -1)
    )

    # 2. Recuperamos un conjunto inicial amplio de candidatos
    # ordenados por similitud semántica con la afirmación
    scores, indices = index.search(
        query_embedding,
        candidate_k
    )

    # 3. Seleccionamos las evidencias finales aplicando
    # diversidad por documento
    selected = []

    # Guarda cuántos chunks hemos seleccionado de cada documento
    document_counts = {}

    for score, idx in zip(scores[0], indices[0]):

        # FAISS puede devolver -1 si no encuentra suficientes resultados
        if idx == -1:
            continue

        chunk = chunks[int(idx)]

        document_id = chunk["document_id"]

        # Número de chunks ya seleccionados de este documento
        current_count = document_counts.get(
            document_id,
            0
        )

        # Si ya hemos alcanzado el máximo permitido para este documento, ignoramos este chunk
        if current_count >= max_chunks_per_document:
            continue

        # Añadimos el chunk al conjunto final de evidencias
        selected.append({
            "score": float(score),
            "document_id": document_id,
            "chunk_id": chunk["chunk_id"],
            "url": chunk["url"],
            "source_type": chunk["source_type"],
            "text": chunk["text"],
        })

        # Actualizamos el contador de chunks seleccionados para este documento
        document_counts[document_id] = (
            current_count + 1
        )

        # Terminamos cuando alcanzamos el número deseado de evidencias finales
        if len(selected) == final_k:
            break

    return selected                