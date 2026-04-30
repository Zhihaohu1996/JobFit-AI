import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "jobfit_training_data.csv"
rows = [
    ["Python FastAPI SQL REST API project", "Backend intern with Python FastAPI REST API and SQL", 2, "Backend"],
    ["Python PyTorch pandas machine learning project", "AI ML intern requiring Python PyTorch deep learning", 2, "AI/ML"],
    ["Excel marketing report communication", "ML intern with PyTorch NLP and model deployment", 0, "AI/ML"],
    ["Java Spring Boot MySQL API", "Software engineer intern with backend API and database", 2, "SDE"],
    ["Python pandas SQL dashboard", "Data analyst intern with SQL Python reporting", 2, "Data"],
    ["Kotlin Android app project", "Backend intern requiring FastAPI SQL Docker", 1, "Backend"],
]
with DATA.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["resume_text", "job_description", "label", "role_type"])
    writer.writerows(rows)
print(f"Generated dataset at {DATA}")
