# Hate Speech Classification

An end-to-end NLP classification service for identifying hate or abusive text. The project is organized as a reproducible pipeline: data ingestion, cleaning and transformation, model training, evaluation against the current best model, model promotion to Google Cloud Storage, and FastAPI inference.

## Architecture

- `hate/components` contains ingestion, transformation, training, evaluation, and model-pushing stages.
- `hate/pipeline` orchestrates training and prediction.
- `hate/entity` holds configuration and artifact contracts.
- `app.py` exposes `/train` and `/predict` through FastAPI.
- Local model artifacts are used as a fallback when GCS is unavailable.

## Setup

```bash
conda create -n hate python=3.8.18 -y
conda activate hate
pip install -r requirements.txt
```

Configure Google Cloud SDK and the project constants before using cloud ingestion or model promotion.

## Run

```bash
python app.py
```

Open `/docs` for the interactive API documentation. Send text to `POST /predict`; trigger training with `GET /train` only in a controlled environment.

## Reproducibility and safety

Record dataset versions, class balance, thresholds, and evaluation metrics for each model. Hate-speech classification can encode bias and false positives; predictions should support human review rather than make high-impact decisions automatically.
