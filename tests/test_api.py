from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import create_app


def test_api_health_and_artifact_status():
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    response = client.get("/artifacts/status")
    assert response.status_code == 200
    assert "artifacts" in response.json()


def test_api_query_search_recommend_and_genrec():
    client = TestClient(create_app())
    parsed = client.post("/llm/query-understanding", json={"query_text": "gift beauty"})
    assert parsed.status_code == 200
    assert parsed.json()["provider"] == "mock"
    search = client.post("/search", json={"query_text": "gift beauty", "top_k": 2})
    assert search.status_code == 200
    assert search.json()["items"]
    recommend = client.get("/recommend/user_00001?top_k=2")
    assert recommend.status_code == 200
    assert recommend.json()["items"]
    genrec = client.post("/genrec", json={"query_text": "gift beauty", "top_k": 2})
    assert genrec.status_code == 200
    assert genrec.json()["hallucination_rate"] == 0.0


def test_api_leaderboard_business_and_error_taxonomy():
    client = TestClient(create_app())
    assert client.get("/models/leaderboard").status_code == 200
    assert client.get("/metrics/business").status_code == 200
    taxonomy = client.get("/errors/taxonomy")
    assert taxonomy.status_code == 200
    assert "markdown" in taxonomy.json()


def test_api_validation_error_is_json_not_stacktrace():
    client = TestClient(create_app())
    response = client.post("/search", json={"query_text": "", "top_k": 0})
    assert response.status_code == 422
    assert "Traceback" not in response.text
