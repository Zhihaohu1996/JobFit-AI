from scripts.preprocess import extract_skills, build_features

def test_extract_skills():
    skills = extract_skills("Python FastAPI SQL project")
    assert "python" in skills
    assert "fastapi" in skills

def test_build_features_length():
    features = build_features("Python SQL", "Python FastAPI SQL")
    assert len(features) == 5
