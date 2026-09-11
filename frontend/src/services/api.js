export function getApiBaseUrl() {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  if (typeof window !== "undefined" && window.location && window.location.hostname) {
    const protocol = window.location.protocol || "http:";
    const hostname = window.location.hostname;
    return `${protocol}//${hostname}:8000`;
  }
  return "http://127.0.0.1:8000";
}

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

  const response = await fetch(`${getApiBaseUrl()}/api/query`, {
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

  const response = await fetch(`${getApiBaseUrl()}/api/predict-future`, {
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

  const response = await fetch(`${getApiBaseUrl()}/api/predict-future-dynamic`, {
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
    const res = await fetch(`${getApiBaseUrl()}/health`);
    if (res.ok) return await res.json();
  } catch {}
  return null;
}

/**
 * Acquires real Sentinel-2 L2A satellite imagery from CDSE / Sentinel Hub via backend.
 *
 * @param {Object} params
 * @param {Array<number>} params.bbox - Bounding box [min_lon, min_lat, max_lon, max_lat]
 * @param {string} [params.dateFrom] - ISO start date string
 * @param {string} [params.dateTo] - ISO end date string
 * @returns {Promise<Object>} Sentinel2Response from backend
 */
export async function acquireSentinel2Imagery({ bbox, dateFrom = "2025-01-01T00:00:00Z", dateTo = "2025-01-31T23:59:59Z", width = 256, height = 256 }) {
  if (!bbox || bbox.length !== 4) {
    throw new Error("bbox must contain exactly 4 coordinate floats: [min_lon, min_lat, max_lon, max_lat]");
  }

  console.log("[SatQuery-Trace] [acquireSentinel2Imagery] Initiating Sentinel-2 request to:", `${getApiBaseUrl()}/api/satellite/sentinel2`);
  const requestPayload = {
    bbox: bbox.map(Number),
    date_from: dateFrom,
    date_to: dateTo,
    width,
    height,
  };
  console.log("[SatQuery-Trace] [acquireSentinel2Imagery] Request Payload:", JSON.stringify(requestPayload));

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 35000);

  try {
    const response = await fetch(`${getApiBaseUrl()}/api/satellite/sentinel2`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestPayload),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    console.log("[SatQuery-Trace] [acquireSentinel2Imagery] HTTP Response Status:", response.status, response.statusText);

    if (!response.ok) {
      let detail = `Backend HTTP error ${response.status}`;
      try {
        const errJson = await response.json();
        console.error("[SatQuery-Trace] [acquireSentinel2Imagery] Backend error JSON:", errJson);
        if (errJson.detail) {
          detail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
        }
      } catch {}
      throw new Error(detail);
    }

    const data = await response.json();
    console.log("[SatQuery-Trace] [acquireSentinel2Imagery] Successful Response Data:", data);
    return data;
  } catch (err) {
    clearTimeout(timeoutId);
    console.error("[SatQuery-Trace] [acquireSentinel2Imagery] Error caught:", err);
    if (err.name === "AbortError") {
      throw new Error("Sentinel-2 acquisition request timed out after 35s.");
    }
    throw err;
  }
}
