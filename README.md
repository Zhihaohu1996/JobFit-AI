# JobFit AI

JobFit AI is a PyTorch + FastAPI project that compares resume text with internship job descriptions, predicts match level, and identifies missing technical skills.

## Run
```bash
pip install -r requirements.txt
python scripts/generate_dataset.py
python scripts/train.py
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs and test POST /analyze.
