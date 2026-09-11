"""
SatQuery AI — Copernicus Sentinel Hub Service

Provides CDSE / Sentinel Hub OAuth authentication and Sentinel-2 L2A imagery processing.
"""

from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from fastapi import HTTPException
from PIL import Image

from app.config import settings


class SentinelHubService:
    """Service abstraction for Copernicus Data Space Ecosystem (CDSE) / Sentinel Hub."""

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

    @property
    def client_id(self) -> str:
        return settings.copernicus_client_id or os.getenv("COPERNICUS_CLIENT_ID", "")

    @property
    def client_secret(self) -> str:
        return settings.copernicus_client_secret or os.getenv("COPERNICUS_CLIENT_SECRET", "")

    @property
    def token_url(self) -> str:
        return settings.copernicus_token_url or os.getenv(
            "COPERNICUS_TOKEN_URL",
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        )

    @property
    def base_url(self) -> str:
        return settings.sentinel_hub_base_url or os.getenv(
            "SENTINEL_HUB_BASE_URL", "https://sh.dataspace.copernicus.eu"
        )

    def get_access_token(self) -> str:
        """Obtains or retrieves cached OAuth 2.0 access token from CDSE identity server."""
        now = time.time()
        # Reuse valid cached token if expiry is > 60 seconds away
        if self._token and (self._token_expires_at - now) > 60:
            return self._token

        cid = self.client_id
        csec = self.client_secret

        if not cid or not csec or csec == "YOUR_CLIENT_SECRET":
            raise HTTPException(
                status_code=500,
                detail=(
                    "CDSE Copernicus credentials not configured. Please check COPERNICUS_CLIENT_ID "
                    "and COPERNICUS_CLIENT_SECRET in backend/.env"
                ),
            )

        payload = {
            "grant_type": "client_credentials",
            "client_id": cid,
            "client_secret": csec,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        try:
            res = requests.post(self.token_url, data=payload, headers=headers, timeout=15)
            if res.status_code != 200:
                detail_msg = f"CDSE OAuth Authentication Failed (HTTP {res.status_code})"
                if res.status_code in (400, 401, 403):
                    detail_msg += f": {res.text}"
                raise HTTPException(status_code=res.status_code, detail=detail_msg)

            data = res.json()
            token = data.get("access_token")
            expires_in = data.get("expires_in", 3600)

            if not token:
                raise HTTPException(
                    status_code=500,
                    detail="CDSE OAuth token endpoint returned success but missing access_token",
                )

            self._token = token
            self._token_expires_at = now + float(expires_in)
            return token

        except requests.exceptions.Timeout:
            raise HTTPException(status_code=504, detail="CDSE OAuth authentication request timed out")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Network error during CDSE OAuth authentication: {e}")

    def acquire_sentinel2_image(
        self,
        bbox: List[float],
        date_from: str = "2025-01-01T00:00:00Z",
        date_to: str = "2025-01-31T23:59:59Z",
        width: int = 256,
        height: int = 256,
        max_cloud_coverage: int = 30,
        upsampling: str = "BICUBIC",
        downsampling: str = "BICUBIC",
    ) -> Dict[str, Any]:
        """Requests Sentinel-2 L2A true-color RGB imagery from Sentinel Hub Processing API."""
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise HTTPException(
                status_code=400,
                detail="Invalid bbox. Must be a list of 4 floats: [min_lon, min_lat, max_lon, max_lat]",
            )

        token = self.get_access_token()

        evalscript = """//VERSION=3
function setup() {
  return {
    input: ["B04", "B03", "B02"],
    output: { bands: 3 }
  };
}
function evaluatePixel(sample) {
  return [3.5 * sample.B04, 3.5 * sample.B03, 3.5 * sample.B02];
}
"""

        request_payload = {
            "input": {
                "bounds": {
                    "bbox": [float(b) for b in bbox],
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                    },
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": date_from,
                                "to": date_to,
                            },
                            "maxCloudCoverage": max_cloud_coverage,
                        },
                        "processing": {
                            "upsampling": upsampling,
                            "downsampling": downsampling,
                        },
                    }
                ],
            },
            "output": {
                "width": int(width),
                "height": int(height),
                "responses": [
                    {
                        "identifier": "default",
                        "format": {"type": "image/png"},
                    }
                ],
            },
            "evalscript": evalscript,
        }

        api_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/png",
        }

        process_endpoints = [
            f"{self.base_url.rstrip('/')}/api/v1/process",
            f"{self.base_url.rstrip('/')}/process/v1",
        ]

        process_response = None
        for endpoint in process_endpoints:
            try:
                res = requests.post(endpoint, json=request_payload, headers=api_headers, timeout=35)
                if res.status_code == 200:
                    process_response = res
                    break
                elif res.status_code == 404:
                    continue
                else:
                    process_response = res
                    break
            except requests.exceptions.Timeout:
                raise HTTPException(status_code=504, detail="Sentinel Hub Processing API request timed out")
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=502, detail=f"Network error calling Sentinel Hub API: {e}")

        if not process_response or process_response.status_code != 200:
            status_code = process_response.status_code if process_response else 500
            err_text = process_response.text if process_response else "Unknown processing API error"
            raise HTTPException(
                status_code=status_code if status_code < 600 else 500,
                detail=f"Sentinel Hub Processing API error (HTTP {status_code}): {err_text[:300]}",
            )

        # Validate that image is not blank/no-data or completely cloud-saturated
        img = Image.open(io.BytesIO(process_response.content))
        arr = np.array(img).astype(float)
        arr_mean = float(arr.mean())
        arr_std = float(arr.std())

        # Saturated white cloud (mean >= 250, std < 3) or completely black no-data (mean <= 2, std < 2)
        if (arr_mean >= 250.0 and arr_std < 3.0) or (arr_mean <= 2.0 and arr_std < 2.0):
            is_cloud = arr_mean >= 250.0
            err_msg = (
                "No cloud-free Sentinel-2 imagery available for this date range: "
                "Target area is completely obscured by dense cloud cover (< 30% cloud-free imagery not found). "
                "Please try an alternate season (e.g. Nov–May) or expand your date range."
                if is_cloud
                else "No valid Sentinel-2 satellite data returned for this date range and bounding box."
            )
            return {
                "success": False,
                "image_id": None,
                "source": "Copernicus Data Space",
                "satellite": "Sentinel-2",
                "product": "Sentinel-2 L2A",
                "bbox": bbox,
                "date_from": date_from,
                "date_to": date_to,
                "image_url": None,
                "width": width,
                "height": height,
                "size_bytes": 0,
                "message": err_msg,
                "detail": err_msg,
            }

        # Save acquired PNG temporarily in upload cache
        temp_dir = settings.upload_path / "temporary"
        temp_dir.mkdir(parents=True, exist_ok=True)

        image_id = uuid.uuid4().hex[:12]
        output_filename = f"sentinel2_{image_id}.png"
        output_path = temp_dir / output_filename

        with open(output_path, "wb") as f:
            f.write(process_response.content)

        file_size = output_path.stat().st_size

        return {
            "success": True,
            "image_id": image_id,
            "source": "Copernicus Data Space",
            "satellite": "Sentinel-2",
            "product": "Sentinel-2 L2A",
            "bbox": bbox,
            "date_from": date_from,
            "date_to": date_to,
            "image_url": f"/api/satellite/image/{image_id}",
            "width": width,
            "height": height,
            "size_bytes": file_size,
        }


# Global singleton service instance
sentinel_service = SentinelHubService()
