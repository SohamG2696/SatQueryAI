"""SatQuery AI — Land Cover Future Prediction API Router."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import prediction_service

router = APIRouter(prefix="/api", tags=["Future Prediction"])


@router.post("/predict-future", response_model=PredictionResponse)
async def predict_landcover_future(request: PredictionRequest):
    """
    Statistical trend projection endpoint for future land-cover composition.

    Accepts a region_id and target_year, returning predicted built_up_pct,
    vegetation_pct, and water_pct along with confidence assessment, R² fit scores,
    and contextual interpretation notes.
    """
    try:
        result = prediction_service.get_future_prediction(
            region_id=request.region_id,
            target_year=request.target_year,
        )
        return PredictionResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-future-dynamic")
async def predict_landcover_future_dynamic(
    images: list[UploadFile] = File(...),
    years: str = Form(..., description="JSON array or comma-separated acquisition years matching images, e.g. '[2019, 2021, 2023, 2025]'"),
    query: str = Form(..., description="Natural language prediction query, e.g. 'Predict the change of buildings and trees in 2027.'"),
):
    """
    Dynamic multi-image future prediction endpoint for user-uploaded satellite imagery.

    Routes through the agentic controller -> dynamic_predictor pipeline:
    1. Saves uploaded images to temporary storage
    2. Validates spatial centroid overlap across images
    3. Extracts SCL land-cover metrics per image
    4. Executes weighted linear trend projection for requested target_year
    """
    import json
    import tempfile
    import uuid
    from pathlib import Path
    from app.agent.controller import controller

    if not images or len(images) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 historical satellite images are required for future trend prediction.",
        )

    # Parse years parameter
    parsed_years: list[int] = []
    try:
        if years.strip().startswith("["):
            parsed_years = [int(y) for y in json.loads(years)]
        else:
            parsed_years = [int(y.strip()) for y in years.split(",") if y.strip()]
    except Exception as pe:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid 'years' parameter format: {pe}. Expected JSON array like '[2019, 2021, 2023, 2025]'.",
        )

    if len(parsed_years) != len(images):
        raise HTTPException(
            status_code=400,
            detail=f"Mismatched counts: received {len(images)} images but {len(parsed_years)} years.",
        )

    temp_dir = Path(tempfile.gettempdir()) / f"satquery_upload_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    try:
        for idx, img_file in enumerate(images):
            ext = Path(img_file.filename or f"image_{idx}.tif").suffix or ".tif"
            out_path = temp_dir / f"upload_{idx}_{parsed_years[idx]}{ext}"
            contents = await img_file.read()
            out_path.write_bytes(contents)
            saved_paths.append(str(out_path.resolve()))

        # Route through Agentic Controller
        response = controller.process_query(
            images=saved_paths,
            query=query,
            metadata={"years": parsed_years},
        )
        return response

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dynamic prediction failed: {e}")
    finally:
        # Clean up temporary uploaded files
        try:
            for p in saved_paths:
                if Path(p).exists():
                    Path(p).unlink()
            if temp_dir.exists():
                temp_dir.rmdir()
        except Exception:
            pass
