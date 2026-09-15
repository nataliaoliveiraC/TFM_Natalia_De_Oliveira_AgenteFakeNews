import sys
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.retrieval import (
    load_knowledge_records,
    build_claim_index,
    retrieve_evidence,
)

from src.evidence_verifier import (
    verify_evidence,
)

DATA_DIR = PROJECT_ROOT / "Data" / "averitec"
RAW_DIR = DATA_DIR / "raw"
CLAIMS_DIR = RAW_DIR / "claims"
KNOWLEDGE_DIR = RAW_DIR / "knowledge_store"
PROCESSED_DIR = DATA_DIR / "processed"

DEV_CLAIMS_PATH = CLAIMS_DIR / "dev.json"

DEV_KNOWLEDGE_ZIP = (
    KNOWLEDGE_DIR
    / "data_store"
    / "knowledge_store"
    / "dev_knowledge_store.zip"
)

DEV_RESULTS_PATH = (
    PROCESSED_DIR
    / "dev_evaluation_results.csv"
)

load_dotenv(PROJECT_ROOT/".env")

client = OpenAI()

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

def verify_averitec_claim(
    claim_id,
    claim,
    knowledge_zip,
    embedding_model,
    text_splitter,
    client,
):
    """
    Ejecuta el pipeline completo de verificación factual para una claim de AVeriTeC.

    Parameters
    ----------
    claim_id : int
        Identificador de la claim en AVeriTeC.

    claim : str
        Texto de la afirmación que se quiere verificar.

    knowledge_zip : str or Path
        ZIP que contiene el knowledge store correspondiente.

    embedding_model
        Modelo utilizado para generar embeddings.

    text_splitter
        Objeto utilizado para dividir los documentos en chunks.

    client
        Cliente de OpenAI utilizado por el Evidence Verifier.

    Returns
    -------
    dict
        Resultado completo del pipeline, incluyendo información del retrieval, evidencias recuperadas y
        verificación factual.
    """

    # 1. Cargamos el knowledge store asociado a la claim
    knowledge_records = load_knowledge_records(
        knowledge_zip=knowledge_zip,
        claim_id=claim_id
    )

    # 2. Construimos el índice vectorial
    chunks, index, retrieval_info = build_claim_index(
        claim=claim,
        knowledge_records=knowledge_records,
        embedding_model=embedding_model,
        text_splitter=text_splitter
    )

    # 3. Recuperamos las evidencias más relevantes
    evidences = retrieve_evidence(
        claim=claim,
        embedding_model=embedding_model,
        index=index,
        chunks=chunks
    )

    # 4. El LLM verifica la claim utilizando esas evidencias
    verification = verify_evidence(
        claim=claim,
        evidences=evidences,
        client=client
    )

    return {
        "claim_id": claim_id,
        "claim": claim,
        "retrieval_info": retrieval_info,
        "evidences": evidences,
        "verification": verification,
    }

def normalize_averitec_label(label):
    """
    Convierte las etiquetas originales de AVeriTeC al formato
    utilizado por el Evidence Verifier.
    """

    label_mapping = {
        "Supported": "SUPPORTED",
        "Refuted": "REFUTED",
        "Not Enough Evidence": "NOT_ENOUGH_EVIDENCE",
        "Conflicting Evidence/Cherrypicking": "CONFLICTING_EVIDENCE",
    }

    return label_mapping[label]

def evaluate_averitec_sample(
    evaluation_sample,
    knowledge_zip,
    embedding_model,
    text_splitter,
    client,
    output_path,
):
    """
    Evalúa una muestra de claims de AVeriTeC utilizando el pipeline completo de verificación factual.

    Los resultados se guardan después de evaluar cada claim.
    Si existe un archivo previo de resultados, la evaluación continúa únicamente con las claims pendientes.

    Parameters
    ----------
    evaluation_sample : pd.DataFrame
        Muestra de claims que se quiere evaluar.

    knowledge_zip : str or Path
        ZIP que contiene los knowledge stores.

    embedding_model
        Modelo utilizado para generar embeddings.

    text_splitter
        Divisor utilizado para generar chunks.

    client
        Cliente de OpenAI utilizado por el Evidence Verifier.

    output_path : str or Path
        Ruta donde se guardarán progresivamente los resultados.

    Returns
    -------
    pd.DataFrame
        Tabla con los resultados de evaluación.
    """

    output_path = Path(output_path)

    # 1. Comprobamos si ya existen resultados anteriores
    if output_path.exists():

        existing_results = pd.read_csv(
            output_path
        )

        evaluation_results = (
            existing_results.to_dict("records")
        )

        completed_claim_ids = set(
            existing_results["claim_id"]
            .astype(int)
            .tolist()
        )

        print(
            f"Resultados anteriores encontrados: "
            f"{len(completed_claim_ids)} claims."
        )

    else:

        evaluation_results = []
        completed_claim_ids = set()

    # 2. Recorremos la muestra de evaluación
    for claim_id, row in evaluation_sample.iterrows():

        claim_id = int(claim_id)

        # Si ya está evaluada, no repetimos el cálculo
        if claim_id in completed_claim_ids:

            print(
                f"\nClaim {claim_id} ya evaluada. "
                "Se omite."
            )

            continue

        claim = row["claim"]

        gold_label = normalize_averitec_label(
            row["label"]
        )

        print(f"\nEvaluando claim {claim_id}...")
        print("Gold:", gold_label)

        # 3. Ejecutamos el pipeline completo
        result = verify_averitec_claim(
            claim_id=claim_id,
            claim=claim,
            knowledge_zip=knowledge_zip,
            embedding_model=embedding_model,
            text_splitter=text_splitter,
            client=client,
        )

        predicted_label = (
            result["verification"].verdict
        )

        # 4. Guardamos el resultado de la claim
        evaluation_results.append({
            "claim_id": claim_id,
            "claim": claim,
            "gold": gold_label,
            "predicted": predicted_label,
            "correct": predicted_label == gold_label,
            "evidence_sufficient": (
                result["verification"].evidence_sufficient
            ),
            "explanation": (
                result["verification"].explanation
            ),
        })

        completed_claim_ids.add(claim_id)

        # 5. Guardamos un checkpoint después de cada claim
        results_df = pd.DataFrame(
            evaluation_results
        )

        results_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8"
        )

        print("Predicted:", predicted_label)
        print(
            "Correct:",
            predicted_label == gold_label
        )
        print(
            f"Resultados guardados: {len(results_df)}"
        )

    return pd.DataFrame(evaluation_results)


#Ejecución para la evaluación con Averitec
def main():
    dev_df = pd.read_json(
        DEV_CLAIMS_PATH
    )

    dev_evaluation_sample = (
        dev_df
        .groupby("label", group_keys=False)
        .sample(n=10, random_state=42)
    )

    print(
        dev_evaluation_sample["label"].value_counts()
    )


    # test_sample = dev_evaluation_sample.head(1)

    test_results = evaluate_averitec_sample(
        evaluation_sample=dev_evaluation_sample,
        knowledge_zip=DEV_KNOWLEDGE_ZIP,
        embedding_model=embedding_model,
        text_splitter=text_splitter,
        client=client,
        output_path=PROCESSED_DIR / "dev_evaluation_test.csv",
    )

if __name__ == "__main__":
    main()