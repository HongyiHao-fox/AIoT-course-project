import io

import cv2
import pytest

pytest.importorskip("flask")

from blindnav.server import create_app  # noqa: E402


@pytest.fixture
def client(navigator):
    app = create_app(navigator_factory=lambda: navigator)
    return app.test_client()


def jpeg_bytes(image):
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def test_health(client):
    assert client.get("/api/health").get_json()["status"] == "ok"


def test_analyze_multipart(client, image):
    resp = client.post("/api/analyze", data={"image": (io.BytesIO(jpeg_bytes(image)), "f.jpg")})
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["detections"]) == 2
    assert body["description"] is None


def test_describe_raw_body(client, image):
    resp = client.post("/api/describe", data=jpeg_bytes(image), content_type="image/jpeg")
    assert resp.status_code == 200
    assert resp.get_json()["description"] == "前方左侧有行人。"


def test_rejects_missing_or_bad_image(client):
    assert client.post("/api/analyze").status_code == 400
    assert client.post("/api/analyze", data=b"not an image").status_code == 400


def test_index_page(client):
    assert client.get("/").status_code == 200
