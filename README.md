# Agente inteligente para la detección de Fake News con LangGraph

Sistema de verificación de noticias desarrollado como Trabajo Fin de Máster.

La solución combina:

- extracción de afirmaciones verificables mediante LLM
- búsqueda de evidencias en la web
- recuperación de evidencias mediante RAG
- verificación factual por claim
- síntesis de un veredicto global
- clasificación auxiliar mediante Machine Learning para noticias en inglés
- orquestación del flujo mediante LangGraph
- interfaz de demostración desarrollada con Streamlit

## Arquitectura

Flujo principal:

Noticia
→ detección de idioma
→ Claim Analyzer
→ Research Agent
→ Live RAG
→ Evidence Verifier
→ modelo ML si el idioma es inglés
→ Verdict Synthesizer

## Estructura del repositorio

- `app/`: aplicación Streamlit
- `src/`: componentes principales del sistema
- `Notebooks/`: análisis, entrenamiento, evaluación y validación
- `Scripts/`: scripts auxiliares
- `Artefactos/`: modelo entrenado y metadatos
- `Data/`: datasets y resultados procesados
- `requirements.txt`: dependencias necesarias para ejecutar el proyecto

## Datos

Los datasets utilizados para el entrenamiento y evaluación se encuentran en `Data/`.

El Knowledge Store completo de AVeriTeC no se incluye en el repositorio debido a su tamaño.

El notebook:

`4 - Averitec_Retrieval.ipynb`

contiene las celdas necesarias para descargar el Knowledge Store desde Hugging Face y reconstruir la estructura utilizada durante la evaluación.

## Machine Learning

Para noticias en inglés se utiliza un clasificador SVM entrenado sobre datos de fake news.

La predicción del clasificador se utiliza únicamente como señal auxiliar.

El veredicto final prioriza la evidencia factual recuperada mediante el sistema de verificación.

Limitaciones:
- el clasificador ML solo está disponible para noticias en inglés
- el modelo ML presenta limitaciones de generalización derivadas del dataset de entrenamiento
- la calidad de la verificación depende de la disponibilidad y calidad de las fuentes recuperadas
- el decision_score del modelo SVM no representa una probabilidad

## Variables de entorno

El proyecto requiere las siguientes variables:

OPENAI_API_KEY=...
TAVILY_API_KEY=...

Estas variables deben almacenarse en un archivo `.env` local.


## Autor

Natalia de Oliveira

## Instalación

Crear y activar un entorno virtual y ejecutar:

```bash
pip install -r requirements.txt
