# Visa Approval Prediction MLOps

This workspace now mirrors the structure of the reference MLOps visa approval project in a local, runnable form.

## Project Layout

- `app.py`: FastAPI entry point for training and prediction
- `us_visa/`: package with pipeline, entity, utility, and service modules
- `config/`: model and schema configuration
- `templates/` and `static/`: UI for the visa prediction form

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` to use the form.