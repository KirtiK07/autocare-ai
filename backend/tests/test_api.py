from fastapi.testclient import TestClient

from tests.conftest import SAMPLE, FakeLLM


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["database"]


def test_predict_stores_and_lists(client: TestClient) -> None:
    r = client.post("/api/predict", json=SAMPLE)
    assert r.status_code == 201
    rec = r.json()
    assert rec["id"] >= 1 and len(rec["top_features"]) == 3
    page = client.get("/api/predictions").json()
    assert page["total"] == 1 and page["items"][0]["id"] == rec["id"]
    detail = client.get(f"/api/predictions/{rec['id']}").json()
    assert detail["inputs"] == SAMPLE and detail["messages"] == []
    assert page["items"][0]["created_at"].endswith("Z")


def test_predict_validation_error_names_field(client: TestClient) -> None:
    r = client.post("/api/predict", json={**SAMPLE, "lub_oil_pressure": -3})
    assert r.status_code == 422
    assert r.json()["detail"][0]["field"] == "lub_oil_pressure"


def test_prediction_not_found(client: TestClient) -> None:
    r = client.get("/api/predictions/999")
    assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_batch(client: TestClient) -> None:
    csv = "Engine rpm,Lub oil pressure,Fuel pressure,Coolant pressure,lub oil temp,Coolant temp\n"
    csv += "700,2.5,11,3.1,84,81\n900,3.2,6.0,2.1,77,78\nabc,1,1,1,1,1\n"
    r = client.post("/api/predict/batch", files={"file": ("x.csv", csv, "text/csv")})
    body = r.json()
    assert r.status_code == 200
    assert body["rows"] == 3 and body["succeeded"] == 2 and body["failed"] == 1
    assert body["results"][2]["error"]


def test_batch_missing_columns(client: TestClient) -> None:
    r = client.post("/api/predict/batch", files={"file": ("x.csv", "engine_rpm\n700\n", "text/csv")})
    assert r.status_code == 400 and "missing columns" in r.json()["detail"]


def test_diagnose_saves_chat(client: TestClient, fake_llm: FakeLLM) -> None:
    pid = client.post("/api/predict", json=SAMPLE).json()["id"]
    r = client.post("/api/diagnose", json={"question": "Why is the risk high?", "prediction_id": pid})
    body = r.json()
    assert r.status_code == 200 and body["covered"] and body["citations"][0]["page"] == 42
    assert "PREDICTION:" in fake_llm.calls[0][1]
    msgs = client.get(f"/api/predictions/{pid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_diagnose_rate_limited(client: TestClient) -> None:
    codes = [client.post("/api/diagnose", json={"question": "oil light?"}).status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429


def test_model_endpoints(client: TestClient) -> None:
    m = client.get("/api/model/metrics").json()
    assert {x["key"] for x in m["models"]} >= {"dummy", "pytorch"}
    info = client.get("/api/model/info").json()
    assert set(info["samples"]) == {"healthy", "faulty"}
