/**
 * SatQuery AI - Satellite Imagery Acquisition Service
 * Provides demo region mapping, distance checking, and URL-to-File conversion.
 */

export const DEMO_REGIONS = [
  {
    id: 'region_01',
    name: 'Bengaluru East Tech Cluster',
    city: 'Bengaluru',
    lat: 12.9716,
    lng: 77.5946,
    imageUrl: '/demo_imagery/bengaluru_region_01.png',
    captureDate: '2025-03-15',
    sensor: 'Sentinel-2 / HighRes Optical (0.5m)',
    cloudCover: '1.2%'
  },
  {
    id: 'region_02',
    name: 'Mumbai Port & Coastal Zone',
    city: 'Mumbai',
    lat: 19.0760,
    lng: 72.8777,
    imageUrl: '/demo_imagery/mumbai_region_02.png',
    captureDate: '2025-02-28',
    sensor: 'Sentinel-2 / HighRes Optical (0.5m)',
    cloudCover: '0.5%'
  },
  {
    id: 'region_03',
    name: 'NCR Urban & Agricultural Margin',
    city: 'Delhi',
    lat: 28.6139,
    lng: 77.2090,
    imageUrl: '/demo_imagery/delhi_region_03.png',
    captureDate: '2025-01-20',
    sensor: 'Sentinel-2 / HighRes Optical (0.5m)',
    cloudCover: '2.0%'
  },
  {
    id: 'region_04',
    name: 'Kolkata Hooghly Basin Region',
    city: 'Kolkata',
    lat: 22.5726,
    lng: 88.3639,
    imageUrl: '/demo_imagery/bengaluru_region_01.png',
    captureDate: '2024-12-10',
    sensor: 'Sentinel-2 / HighRes Optical (0.5m)',
    cloudCover: '1.8%'
  },
  {
    id: 'region_05',
    name: 'Chennai Coastal & Industrial Zone',
    city: 'Chennai',
    lat: 13.0827,
    lng: 80.2707,
    imageUrl: '/demo_imagery/mumbai_region_02.png',
    captureDate: '2025-02-14',
    sensor: 'Sentinel-2 / HighRes Optical (0.5m)',
    cloudCover: '0.8%'
  },
  {
    id: 'region_06',
    name: 'Hyderabad HITEC Urban Belt',
    city: 'Hyderabad',
    lat: 17.3850,
    lng: 78.4867,
    imageUrl: '/demo_imagery/bengaluru_region_01.png',
    captureDate: '2025-01-05',
    sensor: 'Sentinel-2 / HighRes Optical (0.5m)',
    cloudCover: '1.0%'
  },
  {
    id: 'region_07',
    name: 'Pune Metropolitan Growth Area',
    city: 'Pune',
    lat: 18.5204,
    lng: 73.8567,
    imageUrl: '/demo_imagery/delhi_region_03.png',
    captureDate: '2024-11-18',
    sensor: 'Sentinel-2 / HighRes Optical (0.5m)',
    cloudCover: '1.5%'
  },
  {
    id: 'region_08',
    name: 'Jaipur Arid Urban Frontier',
    city: 'Jaipur',
    lat: 26.9124,
    lng: 75.7873,
    imageUrl: '/demo_imagery/delhi_region_03.png',
    captureDate: '2024-10-30',
    sensor: 'Sentinel-2 / HighRes Optical (0.5m)',
    cloudCover: '0.2%'
  }
];

/**
 * Calculates Haversine distance in kilometers between two coordinates.
 */
export function getDistanceKm(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

/**
 * Computes a bounding box centered at (lat, lng) with specified size (default: 2.5km x 2.5km).
 * 2.5km x 2.5km at Sentinel-2 native 10m resolution produces ~250x250 native sensor pixels,
 * perfectly aligning with downstream 256x256 model architectures (ChangeFormer & Spatial Grounding).
 *
 * 1 degree latitude ~ 111 km.
 * 1 degree longitude ~ 111 * cos(lat) km.
 */
export function getBoundingBox(lat, lng, sizeKm = 2.5) {
  const halfSize = sizeKm / 2;
  const latDelta = halfSize / 111;
  const lngDelta = halfSize / (111 * Math.cos((lat * Math.PI) / 180));

  const minLat = +(lat - latDelta).toFixed(6);
  const maxLat = +(lat + latDelta).toFixed(6);
  const minLng = +(lng - lngDelta).toFixed(6);
  const maxLng = +(lng + lngDelta).toFixed(6);

  return {
    minLat,
    maxLat,
    minLng,
    maxLng,
    sizeKm,
    areaKm2: +(sizeKm * sizeKm).toFixed(2),
    leafletBounds: [[minLat, minLng], [maxLat, maxLng]],
    formattedBBox: `[${minLng}, ${minLat}, ${maxLng}, ${maxLat}]`
  };
}

export function get1KmBoundingBox(lat, lng, sizeKm = 2.5) {
  return getBoundingBox(lat, lng, sizeKm);
}

/**
 * Resolves satellite imagery preview for given coordinates.
 * Returns match status and details.
 */
export function getSatellitePreview(lat, lng, maxRadiusKm = 45, sizeKm = 2.5) {
  let closestRegion = null;
  let minDistance = Infinity;

  for (const region of DEMO_REGIONS) {
    const dist = getDistanceKm(lat, lng, region.lat, region.lng);
    if (dist < minDistance) {
      minDistance = dist;
      closestRegion = region;
    }
  }

  const bbox = getBoundingBox(lat, lng, sizeKm);

  if (closestRegion && minDistance <= maxRadiusKm) {
    return {
      available: true,
      region: closestRegion,
      distanceKm: +minDistance.toFixed(1),
      bbox,
      acquiredAt: new Date().toISOString()
    };
  }

  return {
    available: false,
    nearestRegion: closestRegion,
    distanceKm: closestRegion ? +minDistance.toFixed(1) : null,
    bbox,
    acquiredAt: new Date().toISOString()
  };
}

/**
 * Utility to convert an image URL asset into a native browser JS File object.
 */
export async function urlToFile(url, filename, mimeType = 'image/png') {
  console.log("[SatQuery-Trace] [urlToFile] Fetching asset URL:", url, "Target filename:", filename);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000);

  try {
    const response = await fetch(url, { signal: controller.signal });
    clearTimeout(timeoutId);
    console.log("[SatQuery-Trace] [urlToFile] Response status for", url, ":", response.status, response.statusText);

    if (!response.ok) {
      throw new Error(`Failed to fetch image asset from ${url}: status ${response.status}`);
    }
    const blob = await response.blob();
    console.log("[SatQuery-Trace] [urlToFile] Blob received: size =", blob.size, "bytes, type =", blob.type);
    const file = new File([blob], filename, { type: blob.type || mimeType });
    console.log("[SatQuery-Trace] [urlToFile] File created successfully:", file.name, file.size, "bytes");
    return file;
  } catch (err) {
    clearTimeout(timeoutId);
    console.error("[SatQuery-Trace] [urlToFile] Error downloading asset:", err);
    if (err.name === 'AbortError') {
      throw new Error(`Request timed out while downloading image asset from ${url}`);
    }
    throw err;
  }
}

/**
 * Formats a requested year or ISO date string into a start/end ISO window for Sentinel-2 queries.
 * Example: '2020' or '2020-01-01' -> { dateFrom: '2020-01-01T00:00:00Z', dateTo: '2020-01-31T23:59:59Z' }
 */
export function formatSentinel2DateWindow(dateInput) {
  if (!dateInput) {
    return { dateFrom: '2025-01-01T00:00:00Z', dateTo: '2025-01-31T23:59:59Z' };
  }
  const str = String(dateInput).trim();
  const dateMatch = str.match(/\b(20\d\d)-(\d{2})-(\d{2})\b/);
  if (dateMatch) {
    const y = dateMatch[1];
    const m = dateMatch[2];
    return {
      dateFrom: `${y}-${m}-01T00:00:00Z`,
      dateTo: `${y}-${m}-28T23:59:59Z`,
    };
  }
  const yearMatch = str.match(/\b(20\d\d|19\d\d)\b/);
  if (yearMatch) {
    const y = yearMatch[1];
    return {
      dateFrom: `${y}-01-01T00:00:00Z`,
      dateTo: `${y}-01-31T23:59:59Z`,
    };
  }
  return { dateFrom: '2025-01-01T00:00:00Z', dateTo: '2025-01-31T23:59:59Z' };
}
