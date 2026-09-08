from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint_confirms_api_is_running():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_streams_file_and_returns_queued_job(monkeypatch, tmp_path):
    raw_path = tmp_path / "queued.wav"

    def fake_process(session_id, uploaded_path, clean_path):
        raw_path.write_text(f"processed:{session_id}")

    monkeypatch.setattr("app.api.routes._process_audio", fake_process)
    response = client.post(
        "/api/upload",
        files={"file": ("lesson.wav", b"audio-data", "audio/wav")},
    )

    assert response.status_code == 202
    session_id = response.json()["session_id"]
    assert response.json()["status"] == "queued"
    assert client.get(f"/api/status/{session_id}").json()["status"] == "queued"