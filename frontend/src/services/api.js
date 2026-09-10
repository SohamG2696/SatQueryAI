const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

/**
 * Executes a natural language vision-language query against the SatQuery AI FastAPI backend.
 *
 * @param {Object} params
 * @param {string} params.query - Natural language text prompt
 * @param {Array}  params.images - List of uploaded image payloads ({ file: File, ... })
 * @param {Object} [params.metadata] - Optional metadata dictionary
 * @returns {Promise<Object>} Standardized QueryResponse object from backend
 */
export async function executeQuery({ query, images = [], metadata = {} }) {
  if (!query || !query.trim()) {
    throw new Error("Query text is required.");
  }

  const formData = new FormData();
  formData.append("query", query.trim());

  // Attach raw image files to FormData
  images.forEach((imgObj) => {
    const rawFile = imgObj.file || imgObj;
    if (rawFile instanceof File || rawFile instanceof Blob) {
      formData.append("images", rawFile, imgObj.name || rawFile.name || "satellite_image.png");
    }
  });

  if (metadata && Object.keys(metadata).length > 0) {
    formData.append("metadata", JSON.stringify(metadata));
  }

  const response = await fetch(`${API_BASE_URL}/api/query`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let detail = `Backend HTTP error ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        detail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {}
    throw new Error(detail);
  }

  return await response.json();
}

/**
 * Predicts land-cover composition for a specified region and target year.
 *
 * @param {Object} params
 * @param {string} params.regionId - Region ID (e.g., 'region_01')
 * @param {number} params.targetYear - Target forecast year (e.g., 2027)
 * @returns {Promise<Object>} PredictionResponse from backend
 */
export async function predictFuture({ regionId, targetYear }) {
  if (!regionId) {
    throw new Error("region_id is required.");
  }
  if (!targetYear || targetYear <= 2025) {
    throw new Error("target_year must be greater than 2025.");
  }

  const response = await fetch(`${API_BASE_URL}/api/predict-future`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      region_id: regionId,
      target_year: Number(targetYear),
    }),
  });

  if (!response.ok) {
    let detail = `Backend HTTP error ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        detail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {}
    throw new Error(detail);
  }

  return await response.json();
}

/**
 * Executes dynamic multi-image land-cover future prediction using uploaded GeoTIFFs.
 *
 * @param {Object} params
 * @param {Array<File|Blob|Object>} params.images - List of uploaded image files
 * @param {Array<number>} params.years - Corresponding acquisition years list
 * @param {string} params.query - Natural language query prompt
 * @returns {Promise<Object>} Controller QueryResponse with execution_summary parameters
 */
export async function predictFutureDynamic({ images = [], years = [], query }) {
  if (!images || images.length < 2) {
    throw new Error("At least 2 historical satellite images are required.");
  }
  if (!years || years.length !== images.length) {
    throw new Error("Each uploaded image must have a corresponding acquisition year.");
  }
  if (!query || !query.trim()) {
    throw new Error("Prediction query string is required.");
  }

  const formData = new FormData();
  formData.append("query", query.trim());
  formData.append("years", JSON.stringify(years.map(Number)));

  images.forEach((imgObj, idx) => {
    const rawFile = imgObj.file || imgObj;
    if (rawFile instanceof File || rawFile instanceof Blob) {
      formData.append("images", rawFile, imgObj.name || rawFile.name || `satellite_${years[idx]}.tif`);
    }
  });

  const response = await fetch(`${API_BASE_URL}/api/predict-future-dynamic`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let detail = `Backend HTTP error ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        detail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {}
    throw new Error(detail);
  }

  return await response.json();
}

/**
 * Checks backend health status and active ready models.
 */
export async function getBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) return await res.json();
  } catch {}
  return null;
}
