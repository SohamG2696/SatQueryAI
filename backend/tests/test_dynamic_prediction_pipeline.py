"""
End-to-End Test for Dynamic Multi-Image Future Prediction Pipeline.

Tests:
1. Direct Agentic Controller invocation with region_01's 4 images (2019, 2021, 2023, 2025)
2. FastAPI Endpoint POST /api/predict-future-dynamic invocation via TestClient
"""

from pathlib import Path
import json
import sys

THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = THIS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from app.agent.controller import controller

RAW_DIR = PROJECT_ROOT / "models" / "future_prediction" / "data" / "raw"
REGION_01_IMGS = [
    RAW_DIR / "region_01_2019.tif",
    RAW_DIR / "region_01_2021.tif",
    RAW_DIR / "region_01_2023.tif",
    RAW_DIR / "region_01_2025.tif",
]
YEARS = [2019, 2021, 2023, 2025]
QUERY = "Predict the change of buildings, trees and roads in 2027."

client = TestClient(app)


def test_controller_dynamic_prediction_pipeline():
    """Verify Controller routes query to 'future_prediction' and executes dynamic predictor."""
    img_paths = [str(p.resolve()) for p in REGION_01_IMGS]

    response = controller.process_query(
        images=img_paths,
        query=QUERY,
        metadata={"years": YEARS},
    )

    assert response.task_detected == "future_prediction"
    assert response.execution_summary.task_route == "multi_year_future_landcover_prediction"
    assert "dynamic_landcover_predictor" in response.execution_summary.models_used

    params = response.execution_summary.parameters
    assert params["target_year"] == 2027
    assert "built_up_pct" in params["predictions"]
    assert "vegetation_pct" in params["predictions"]
    assert "roads" in params["unsupported_categories"]
    assert len(params["historical_data"]) == 4

    # Confirm percentages match region_01 baseline predictions
    built_pred = params["predictions"]["built_up_pct"]["value"]
    assert abs(built_pred - 62.09) < 0.5, f"Expected built_up ~62.09, got {built_pred}"


def test_api_predict_future_dynamic_endpoint():
    """Verify POST /api/predict-future-dynamic multipart API endpoint."""
    files = []
    for p in REGION_01_IMGS:
        files.append(("images", (p.name, p.read_bytes(), "image/tiff")))

    data = {
        "years": json.dumps(YEARS),
        "query": QUERY,
    }

    response = client.post("/api/predict-future-dynamic", files=files, data=data)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    res_json = response.json()
    assert res_json["task_detected"] == "future_prediction"
    assert "built_up_pct" in res_json["execution_summary"]["parameters"]["predictions"]
