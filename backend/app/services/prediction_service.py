"""
SatQuery AI — Land Cover Future Prediction Service.

Wraps models/future_prediction/inference/predictor.py with validation
and error handling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is on sys.path so models package is importable
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]  # SatQueryAI root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.future_prediction.inference.predictor import predict_future


class PredictionService:
    """Service wrapping future land-cover prediction logic."""

    @staticmethod
    def get_future_prediction(
        region_id: str,
        target_year: int,
        historical_csv_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Validates parameters and returns land-cover forecast dictionary.

        Raises:
            ValueError: For invalid region_id, past target_year, or extreme extrapolation horizon.
        """
        if not region_id or not isinstance(region_id, str):
            raise ValueError("region_id must be a non-empty string.")

        clean_region_id = region_id.strip()

        if target_year <= 2025:
            raise ValueError(f"target_year must be in the future (greater than 2025), got {target_year}.")

        if target_year > 2075:
            raise ValueError(f"target_year {target_year} exceeds maximum supported forecast horizon (2075 / +50 years).")

        try:
            result = predict_future(
                region_id=clean_region_id,
                target_year=target_year,
                historical_csv_path=historical_csv_path,
            )
            return result
        except ValueError as ve:
            raise ValueError(str(ve))
        except Exception as e:
            raise RuntimeError(f"Prediction execution failed: {e}")


prediction_service = PredictionService()
