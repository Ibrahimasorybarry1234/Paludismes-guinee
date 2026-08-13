FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entraîne le modèle au moment du build si aucun modèle n'est déjà présent
RUN [ -f models/best_model.joblib ] || python src/train.py --data data/raw/paludisme.csv --outdir models

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
