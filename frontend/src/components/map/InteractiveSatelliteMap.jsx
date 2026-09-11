import React, { useState, useEffect, useCallback, useRef } from 'react';
import { MapContainer, TileLayer, Rectangle, Marker, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './InteractiveSatelliteMap.css';
import { DEMO_REGIONS, getSatellitePreview, urlToFile, formatSentinel2DateWindow, getDistanceKm } from '../../services/satelliteImagery';
import { acquireSentinel2Imagery, getApiBaseUrl } from '../../services/api';

// Custom cyan pulsing target marker
const customMarkerIcon = L.divIcon({
  className: 'custom-cyan-marker',
  html: `<div class="marker-ping"></div><div class="marker-dot"></div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12]
});

// Map click event listener
function MapClickHandler({ onLocationSelect }) {
  useMapEvents({
    click(e) {
      onLocationSelect(e.latlng.lat, e.latlng.lng);
    }
  });
  return null;
}

// Programmatic map motion controller
function MapController({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center && center[0] !== undefined && center[1] !== undefined) {
      map.flyTo(center, zoom || 14, { duration: 1.2 });
    }
  }, [center, zoom, map]);
  return null;
}

export default function InteractiveSatelliteMap({ onAreaSelected }) {
  // Default centered at Bangalore (Demo Region 01)
  const [selectedLat, setSelectedLat] = useState(12.9716);
  const [selectedLng, setSelectedLng] = useState(77.5946);
  const [mapCenter, setMapCenter] = useState([12.9716, 77.5946]);
  const [mapZoom, setMapZoom] = useState(14);

  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState(null);

  const [previewInfo, setPreviewInfo] = useState(null);
  const [acquiredImage, setAcquiredImage] = useState(null);
  const [acquiredMetadata, setAcquiredMetadata] = useState(null);

  // Bi-temporal acquisition states
  const [temporalMode, setTemporalMode] = useState(false);
  const [dateA, setDateA] = useState('2020-01-01');
  const [dateB, setDateB] = useState('2025-01-01');
  const [acquiredBefore, setAcquiredBefore] = useState(null);
  const [acquiredAfter, setAcquiredAfter] = useState(null);
  const [isAcquiringBefore, setIsAcquiringBefore] = useState(false);
  const [isAcquiringAfter, setIsAcquiringAfter] = useState(false);

  const [isAcquiring, setIsAcquiring] = useState(false);
  const [acquisitionError, setAcquisitionError] = useState(null);

  const lastAcquiredCoordRef = useRef(null);

  // Recalculate satellite preview whenever selected lat/lng changes
  const updateLocation = useCallback((lat, lng, forceReset = false) => {
    const formattedLat = +lat.toFixed(6);
    const formattedLng = +lng.toFixed(6);
    setSelectedLat(formattedLat);
    setSelectedLng(formattedLng);

    const preview = getSatellitePreview(formattedLat, formattedLng);
    setPreviewInfo(preview);

    // Only reset acquired image if explicitly requested or location moved > 1.0 km
    let shouldReset = forceReset;
    if (lastAcquiredCoordRef.current) {
      const dist = getDistanceKm(
        lastAcquiredCoordRef.current.lat,
        lastAcquiredCoordRef.current.lng,
        formattedLat,
        formattedLng
      );
      if (dist > 1.0) {
        shouldReset = true;
      }
    } else {
      shouldReset = true;
    }

    if (shouldReset) {
      setAcquiredImage(null);
      setAcquiredMetadata(null);
      setAcquiredBefore(null);
      setAcquiredAfter(null);
      lastAcquiredCoordRef.current = null;
    }

    setSearchError(null);
    setAcquisitionError(null);
  }, []);

  useEffect(() => {
    updateLocation(12.9716, 77.5946, true);
  }, [updateLocation]);

  const handleSelectLocation = (lat, lng, cityName = '') => {
    setMapCenter([lat, lng]);
    updateLocation(lat, lng, true);
    if (cityName) setSearchQuery(cityName);
  };

  const handleSearchSubmit = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    // Check if query matches a known demo region
    const matchedRegion = DEMO_REGIONS.find(
      (r) =>
        r.city.toLowerCase().includes(searchQuery.toLowerCase()) ||
        r.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    if (matchedRegion) {
      handleSelectLocation(matchedRegion.lat, matchedRegion.lng, matchedRegion.city);
      return;
    }

    // Attempt geocoding lookup
    try {
      setIsSearching(true);
      setSearchError(null);
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}`
      );
      const data = await res.json();
      if (data && data.length > 0) {
        const lat = parseFloat(data[0].lat);
        const lon = parseFloat(data[0].lon);
        handleSelectLocation(lat, lon, data[0].display_name.split(',')[0]);
      } else {
        setSearchError('Location not found. Try searching Bangalore, Mumbai, Delhi...');
      }
    } catch (err) {
      setSearchError('Geocoding service unavailable.');
    } finally {
      setIsSearching(false);
    }
  };

  // Real CDSE Sentinel-2 Single Image Acquisition
  const handleAcquireRealSentinel2 = async () => {
    console.log("[SatQuery-Trace] [handleAcquireRealSentinel2] Button clicked! previewInfo:", previewInfo);
    if (!previewInfo || !previewInfo.bbox) {
      console.warn("[SatQuery-Trace] [handleAcquireRealSentinel2] Aborting: previewInfo or bbox missing!");
      return;
    }

    try {
      setIsAcquiring(true);
      setAcquisitionError(null);

      const bboxArray = [
        previewInfo.bbox.minLng,
        previewInfo.bbox.minLat,
        previewInfo.bbox.maxLng,
        previewInfo.bbox.maxLat,
      ];
      console.log("[SatQuery-Trace] [handleAcquireRealSentinel2] bboxArray:", bboxArray);

      const res = await acquireSentinel2Imagery({
        bbox: bboxArray,
        dateFrom: "2025-01-01T00:00:00Z",
        dateTo: "2025-01-31T23:59:59Z",
      });
      console.log("[SatQuery-Trace] [handleAcquireRealSentinel2] acquireSentinel2Imagery response:", res);

      if (!res || !res.success || !res.image_url) {
        throw new Error(res?.detail || res?.message || "Satellite imagery could not be acquired.");
      }

      const API_BASE_URL = getApiBaseUrl();
      const fullImageUrl = res.image_url.startsWith("http")
        ? res.image_url
        : `${API_BASE_URL}${res.image_url}`;
      console.log("[SatQuery-Trace] [handleAcquireRealSentinel2] Fetching image via urlToFile:", fullImageUrl);

      const imageFile = await urlToFile(
        fullImageUrl,
        `sentinel2_${res.image_id || Date.now()}.png`,
        "image/png"
      );
      console.log("[SatQuery-Trace] [handleAcquireRealSentinel2] Acquired imageFile:", imageFile.name, imageFile.size, "bytes");

      const previewObjectUrl = URL.createObjectURL(imageFile);
      console.log("[SatQuery-Trace] [handleAcquireRealSentinel2] Generated previewObjectUrl:", previewObjectUrl);

      const filePayload = {
        file: imageFile,
        id: `sentinel2_${Date.now()}`,
        name: `Sentinel-2 L2A (${selectedLat.toFixed(4)}° N, ${selectedLng.toFixed(4)}° E)`,
        ext: ".png",
        size: imageFile.size,
        url: previewObjectUrl,
        sensor: "Sentinel-2 L2A (Copernicus Data Space)",
        coordinates: `${selectedLat.toFixed(4)}° N, ${selectedLng.toFixed(4)}° E`,
        captureDate: `${res.date_from ? res.date_from.split('T')[0] : '2025-01-01'} to ${res.date_to ? res.date_to.split('T')[0] : '2025-01-31'}`,
        mapMetadata: {
          name: `Sentinel-2 L2A (${selectedLat.toFixed(4)}° N, ${selectedLng.toFixed(4)}° E)`,
          latitude: selectedLat,
          longitude: selectedLng,
          bbox: previewInfo.bbox.formattedBBox,
          source: "Copernicus Data Space Ecosystem (CDSE)",
          satellite: "Sentinel-2",
          product: "Sentinel-2 L2A",
          sensor: "Sentinel-2 L2A (Copernicus Data Space)",
          captureDate: `${res.date_from ? res.date_from.split('T')[0] : '2025-01-01'} to ${res.date_to ? res.date_to.split('T')[0] : '2025-01-31'}`,
          acquisitionMode: "Real CDSE Sentinel-2 Processing API",
          cloudCover: "< 30%",
        },
      };

      lastAcquiredCoordRef.current = { lat: selectedLat, lng: selectedLng };

      console.log("[SatQuery-Trace] [handleAcquireRealSentinel2] Setting acquiredImage state to:", previewObjectUrl);
      setAcquiredImage(previewObjectUrl);
      setAcquiredMetadata(filePayload.mapMetadata);

      if (onAreaSelected) {
        console.log("[SatQuery-Trace] [handleAcquireRealSentinel2] Triggering onAreaSelected callback with payload:", filePayload);
        onAreaSelected(filePayload);
      }
    } catch (err) {
      console.error("[SatQuery-Trace] [handleAcquireRealSentinel2] ERROR CAUGHT:", err);
      setAcquisitionError(err.message || "Satellite imagery could not be acquired.");
    } finally {
      setIsAcquiring(false);
      console.log("[SatQuery-Trace] [handleAcquireRealSentinel2] Finished acquisition handler.");
    }
  };

  // Real CDSE Sentinel-2 Single Temporal Role Acquisition (BEFORE or AFTER)
  const handleAcquireSingleTemporal = async (role = 'before', dateInput = '2020-01-01') => {
    if (!previewInfo || !previewInfo.bbox) return;

    try {
      if (role === 'before') setIsAcquiringBefore(true);
      else setIsAcquiringAfter(true);
      setIsAcquiring(true);
      setAcquisitionError(null);

      const bboxArray = [
        previewInfo.bbox.minLng,
        previewInfo.bbox.minLat,
        previewInfo.bbox.maxLng,
        previewInfo.bbox.maxLat,
      ];

      const { dateFrom, dateTo } = formatSentinel2DateWindow(dateInput);

      const res = await acquireSentinel2Imagery({
        bbox: bboxArray,
        dateFrom,
        dateTo,
      });

      if (!res || !res.success || !res.image_url) {
        throw new Error(res?.detail || res?.message || `Sentinel-2 imagery for ${role.toUpperCase()} date (${dateInput}) could not be acquired.`);
      }

      const API_BASE_URL = getApiBaseUrl();
      const fullImageUrl = res.image_url.startsWith("http")
        ? res.image_url
        : `${API_BASE_URL}${res.image_url}`;

      const imageFile = await urlToFile(
        fullImageUrl,
        `sentinel2_${role}_${res.image_id || Date.now()}.png`,
        "image/png"
      );

      const previewObjectUrl = URL.createObjectURL(imageFile);
      const captureRange = `${res.date_from ? res.date_from.split('T')[0] : dateFrom.split('T')[0]} to ${res.date_to ? res.date_to.split('T')[0] : dateTo.split('T')[0]}`;

      const filePayload = {
        file: imageFile,
        id: `sentinel2_${role}_${Date.now()}`,
        name: `Sentinel-2 L2A [${role.toUpperCase()}: ${dateInput}] (${selectedLat.toFixed(4)}° N, ${selectedLng.toFixed(4)}° E)`,
        ext: ".png",
        size: imageFile.size,
        url: previewObjectUrl,
        sensor: "Sentinel-2 L2A (Copernicus Data Space)",
        coordinates: `${selectedLat.toFixed(4)}° N, ${selectedLng.toFixed(4)}° E`,
        captureDate: captureRange,
        temporalRole: role,
        requestedDate: dateInput,
        mapMetadata: {
          latitude: selectedLat,
          longitude: selectedLng,
          bbox: previewInfo.bbox.formattedBBox,
          source: "Copernicus Data Space Ecosystem (CDSE)",
          satellite: "Sentinel-2",
          product: "Sentinel-2 L2A",
          temporalRole: role,
          requestedDate: dateInput,
          acquisitionMode: "Real CDSE Sentinel-2 Processing API",
          cloudCover: "< 30%",
        },
      };

      let newBefore = acquiredBefore;
      let newAfter = acquiredAfter;

      if (role === 'before') {
        setAcquiredBefore(filePayload);
        newBefore = filePayload;
      } else {
        setAcquiredAfter(filePayload);
        newAfter = filePayload;
      }

      if (newBefore && newAfter && onAreaSelected) {
        onAreaSelected([newBefore, newAfter]);
      } else if (onAreaSelected) {
        onAreaSelected(filePayload);
      }
    } catch (err) {
      console.error(`Sentinel-2 acquisition error (${role}):`, err);
      setAcquisitionError(err.message || `Satellite imagery could not be acquired for ${role.toUpperCase()} date.`);
      if (role === 'before') {
        setAcquiredBefore(null);
      } else {
        setAcquiredAfter(null);
      }
    } finally {
      setIsAcquiringBefore(false);
      setIsAcquiringAfter(false);
      setIsAcquiring(false);
    }
  };

  // Real CDSE Sentinel-2 Dual Temporal Scene Acquisition
  const handleAcquireBothTemporal = async () => {
    if (!previewInfo || !previewInfo.bbox) return;

    try {
      setIsAcquiring(true);
      setIsAcquiringBefore(true);
      setIsAcquiringAfter(true);
      setAcquisitionError(null);

      const bboxArray = [
        previewInfo.bbox.minLng,
        previewInfo.bbox.minLat,
        previewInfo.bbox.maxLng,
        previewInfo.bbox.maxLat,
      ];

      const windowA = formatSentinel2DateWindow(dateA);
      const windowB = formatSentinel2DateWindow(dateB);

      // Acquire Before image
      const resA = await acquireSentinel2Imagery({
        bbox: bboxArray,
        dateFrom: windowA.dateFrom,
        dateTo: windowA.dateTo,
      });

      if (!resA || !resA.success || !resA.image_url) {
        throw new Error(resA?.detail || resA?.message || `BEFORE satellite imagery for date ${dateA} could not be acquired.`);
      }

      // Acquire After image
      const resB = await acquireSentinel2Imagery({
        bbox: bboxArray,
        dateFrom: windowB.dateFrom,
        dateTo: windowB.dateTo,
      });

      if (!resB || !resB.success || !resB.image_url) {
        throw new Error(resB?.detail || resB?.message || `AFTER satellite imagery for date ${dateB} could not be acquired.`);
      }

      const API_BASE_URL = getApiBaseUrl();
      const urlA = resA.image_url.startsWith("http") ? resA.image_url : `${API_BASE_URL}${resA.image_url}`;
      const urlB = resB.image_url.startsWith("http") ? resB.image_url : `${API_BASE_URL}${resB.image_url}`;

      const fileA = await urlToFile(urlA, `sentinel2_before_${resA.image_id || Date.now()}.png`, "image/png");
      const fileB = await urlToFile(urlB, `sentinel2_after_${resB.image_id || Date.now()}.png`, "image/png");

      const previewAUrl = URL.createObjectURL(fileA);
      const previewBUrl = URL.createObjectURL(fileB);

      const beforePayload = {
        file: fileA,
        id: `sentinel2_before_${Date.now()}`,
        name: `Sentinel-2 L2A BEFORE [${dateA}] (${selectedLat.toFixed(4)}° N, ${selectedLng.toFixed(4)}° E)`,
        ext: ".png",
        size: fileA.size,
        url: previewAUrl,
        sensor: "Sentinel-2 L2A (Copernicus Data Space)",
        coordinates: `${selectedLat.toFixed(4)}° N, ${selectedLng.toFixed(4)}° E`,
        captureDate: `${resA.date_from ? resA.date_from.split('T')[0] : windowA.dateFrom.split('T')[0]} to ${resA.date_to ? resA.date_to.split('T')[0] : windowA.dateTo.split('T')[0]}`,
        temporalRole: "before",
        requestedDate: dateA,
        mapMetadata: {
          latitude: selectedLat,
          longitude: selectedLng,
          bbox: previewInfo.bbox.formattedBBox,
          source: "Copernicus Data Space Ecosystem (CDSE)",
          satellite: "Sentinel-2",
          product: "Sentinel-2 L2A",
          temporalRole: "before",
          requestedDate: dateA,
          acquisitionMode: "Real CDSE Sentinel-2 Processing API",
          cloudCover: "< 30%",
        },
      };

      const afterPayload = {
        file: fileB,
        id: `sentinel2_after_${Date.now()}`,
        name: `Sentinel-2 L2A AFTER [${dateB}] (${selectedLat.toFixed(4)}° N, ${selectedLng.toFixed(4)}° E)`,
        ext: ".png",
        size: fileB.size,
        url: previewBUrl,
        sensor: "Sentinel-2 L2A (Copernicus Data Space)",
        coordinates: `${selectedLat.toFixed(4)}° N, ${selectedLng.toFixed(4)}° E`,
        captureDate: `${resB.date_from ? resB.date_from.split('T')[0] : windowB.dateFrom.split('T')[0]} to ${resB.date_to ? resB.date_to.split('T')[0] : windowB.dateTo.split('T')[0]}`,
        temporalRole: "after",
        requestedDate: dateB,
        mapMetadata: {
          latitude: selectedLat,
          longitude: selectedLng,
          bbox: previewInfo.bbox.formattedBBox,
          source: "Copernicus Data Space Ecosystem (CDSE)",
          satellite: "Sentinel-2",
          product: "Sentinel-2 L2A",
          temporalRole: "after",
          requestedDate: dateB,
          acquisitionMode: "Real CDSE Sentinel-2 Processing API",
          cloudCover: "< 30%",
        },
      };

      setAcquiredBefore(beforePayload);
      setAcquiredAfter(afterPayload);

      if (onAreaSelected) {
        onAreaSelected([beforePayload, afterPayload]);
      }
    } catch (err) {
      console.error("Bi-temporal acquisition error:", err);
      setAcquisitionError(err.message || "Bi-temporal satellite imagery could not be acquired.");
    } finally {
      setIsAcquiringBefore(false);
      setIsAcquiringAfter(false);
      setIsAcquiring(false);
    }
  };

  const handleUseThisArea = async () => {
    if (!previewInfo || !previewInfo.available || !previewInfo.region) {
      setAcquisitionError('No satellite imagery available for this region.');
      return;
    }

    try {
      setIsAcquiring(true);
      setAcquisitionError(null);

      // Convert preview image asset into a JS File object
      const imageFile = await urlToFile(
        previewInfo.region.imageUrl,
        `${previewInfo.region.id}_satellite.png`,
        'image/png'
      );

      const filePayload = {
        file: imageFile,
        id: `map_${Date.now()}`,
        name: `${previewInfo.region.city} Satellite Acquisition (${selectedLat.toFixed(4)}, ${selectedLng.toFixed(4)})`,
        ext: '.png',
        size: imageFile.size,
        url: URL.createObjectURL(imageFile),
        sensor: previewInfo.region.sensor,
        coordinates: `${selectedLat.toFixed(4)}° N, ${selectedLng.toFixed(4)}° E`,
        captureDate: previewInfo.region.captureDate,
        mapMetadata: {
          latitude: selectedLat,
          longitude: selectedLng,
          bbox: previewInfo.bbox.formattedBBox,
          source: 'Interactive Satellite Map Acquisition',
          regionId: previewInfo.region.id,
          acquisitionMode: 'Demo Region Asset',
          cloudCover: previewInfo.region.cloudCover
        }
      };

      if (onAreaSelected) {
        onAreaSelected(filePayload);
      }
    } catch (err) {
      console.error('Acquisition error:', err);
      setAcquisitionError('Failed to load satellite imagery asset.');
    } finally {
      setIsAcquiring(false);
    }
  };

  const activeBounds = previewInfo?.bbox?.leafletBounds || [
    [selectedLat - 0.0045, selectedLng - 0.0045],
    [selectedLat + 0.0045, selectedLng + 0.0045]
  ];

  return (
    <div className="satellite-map-wrapper">
      {/* Top Header & Location Controls */}
      <div className="map-control-header">
        <form onSubmit={handleSearchSubmit} className="map-search-box">
          <svg
            className="map-search-icon"
            width="16"
            height="16"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <input
            type="text"
            className="map-search-input"
            placeholder="Search coordinates or city (e.g. Bangalore, Mumbai)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {isSearching && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-cyan-400 text-xs animate-pulse">
              Searching...
            </div>
          )}
        </form>

        {/* Quick Demo City Jump Badges */}
        <div className="quick-cities-bar">
          <span className="text-xs text-slate-400 font-medium mr-1">Demo Cities:</span>
          {DEMO_REGIONS.map((reg) => {
            const isActive = previewInfo?.region?.id === reg.id;
            return (
              <button
                key={reg.id}
                type="button"
                className={`quick-city-badge ${isActive ? 'active' : ''}`}
                onClick={() => handleSelectLocation(reg.lat, reg.lng, reg.city)}
              >
                {reg.city}
              </button>
            );
          })}
        </div>
      </div>

      {searchError && (
        <div className="px-4 py-2 bg-amber-500/15 border-b border-amber-500/30 text-amber-300 text-xs flex items-center gap-2">
          <span>⚠️</span> {searchError}
        </div>
      )}

      {/* Main Interactive Leaflet Map Stage */}
      <div className="map-stage-container">
        <MapContainer
          center={mapCenter}
          zoom={mapZoom}
          zoomControl={true}
          scrollWheelZoom={true}
        >
          {/* OpenStreetMap Standard Tile Layer (Free, No API Key, No Watermark) */}
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            maxZoom={19}
          />

          <MapController center={mapCenter} zoom={mapZoom} />
          <MapClickHandler onLocationSelect={(lat, lng) => updateLocation(lat, lng)} />

          {/* Selected Location Center Pulse Marker */}
          <Marker position={[selectedLat, selectedLng]} icon={customMarkerIcon} />

          {/* 1km x 1km Bounding Box Rectangle Overlay */}
          <Rectangle
            bounds={activeBounds}
            pathOptions={{
              color: '#38bdf8',
              weight: 2,
              fillColor: '#0284c7',
              fillOpacity: 0.15,
              dashArray: '6, 6'
            }}
          />
        </MapContainer>

        {/* Floating Telemetry HUD */}
        <div className="telemetry-overlay">
          <div className="telemetry-title">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping inline-block mr-1"></span>
            Target Telemetry HUD
          </div>
          <div className="telemetry-grid">
            <div>
              <div className="telemetry-item-label">Latitude</div>
              <div className="telemetry-item-val">{selectedLat.toFixed(5)}° N</div>
            </div>
            <div>
              <div className="telemetry-item-label">Longitude</div>
              <div className="telemetry-item-val">{selectedLng.toFixed(5)}° E</div>
            </div>
            <div>
              <div className="telemetry-item-label">Target Area</div>
              <div className="telemetry-item-val">
                {previewInfo?.bbox?.areaKm2 ? `${previewInfo.bbox.areaKm2} km² (${previewInfo.bbox.sizeKm}×${previewInfo.bbox.sizeKm} km)` : '6.25 km² (2.5×2.5 km)'}
              </div>
            </div>
            <div>
              <div className="telemetry-item-label">Grid Region</div>
              <div className="telemetry-item-val">
                {previewInfo?.region?.id || 'OUT_OF_BOUNDS'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Satellite Imagery Preview & Acquisition Bar */}
      <div className="satellite-preview-drawer">
        {/* MODE SELECTION TAB HEADER */}
        <div className="flex items-center gap-2 mb-3 bg-slate-950/80 border border-slate-800 p-1.5 rounded-xl text-xs">
          <button
            type="button"
            onClick={() => setTemporalMode(false)}
            className={`px-3 py-1.5 rounded-lg font-semibold transition cursor-pointer ${
              !temporalMode
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-[0_0_15px_rgba(34,211,238,0.15)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            📸 Single Scene Acquisition
          </button>
          <button
            type="button"
            onClick={() => setTemporalMode(true)}
            className={`px-3 py-1.5 rounded-lg font-semibold transition cursor-pointer ${
              temporalMode
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-[0_0_15px_rgba(34,211,238,0.15)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            ⏱️ Bi-Temporal Change Analysis (2 Dates)
          </button>
        </div>

        {/* TEMPORAL DATE CONTROLS */}
        {temporalMode && (
          <div className="grid grid-cols-2 gap-3 mb-3">
            <div className="bg-slate-900/80 p-2.5 rounded-xl border border-slate-800">
              <label className="text-[11px] font-bold text-cyan-400 block mb-1">
                BEFORE DATE (DATE A)
              </label>
              <input
                type="text"
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white focus:outline-none focus:border-cyan-400"
                placeholder="e.g. 2020 or 2020-01-01"
                value={dateA}
                onChange={(e) => setDateA(e.target.value)}
              />
            </div>
            <div className="bg-slate-900/80 p-2.5 rounded-xl border border-slate-800">
              <label className="text-[11px] font-bold text-violet-400 block mb-1">
                AFTER DATE (DATE B)
              </label>
              <input
                type="text"
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white focus:outline-none focus:border-violet-400"
                placeholder="e.g. 2025 or 2025-01-01"
                value={dateB}
                onChange={(e) => setDateB(e.target.value)}
              />
            </div>
          </div>
        )}

        {/* TEMPORAL PREVIEW CARDS (BEFORE / AFTER) */}
        {temporalMode ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              {/* BEFORE CARD */}
              <div className="preview-thumbnail-box relative">
                {acquiredBefore ? (
                  <>
                    <img src={acquiredBefore.url} alt="Before Scene" className="preview-thumbnail-img" />
                    <div className="preview-badge-status preview-badge-success">
                      BEFORE: {acquiredBefore.requestedDate}
                    </div>
                  </>
                ) : (
                  <div className="bg-slate-950 flex flex-col items-center justify-center p-3 text-center h-28 text-slate-400 text-xs">
                    <span className="text-xl mb-1">📸</span>
                    <span className="font-semibold text-cyan-300">BEFORE Scene</span>
                    <span className="text-[10px] text-slate-500 mt-1">Date: {dateA}</span>
                  </div>
                )}
              </div>

              {/* AFTER CARD */}
              <div className="preview-thumbnail-box relative">
                {acquiredAfter ? (
                  <>
                    <img src={acquiredAfter.url} alt="After Scene" className="preview-thumbnail-img" />
                    <div className="preview-badge-status preview-badge-success">
                      AFTER: {acquiredAfter.requestedDate}
                    </div>
                  </>
                ) : (
                  <div className="bg-slate-950 flex flex-col items-center justify-center p-3 text-center h-28 text-slate-400 text-xs">
                    <span className="text-xl mb-1">📸</span>
                    <span className="font-semibold text-violet-300">AFTER Scene</span>
                    <span className="text-[10px] text-slate-500 mt-1">Date: {dateB}</span>
                  </div>
                )}
              </div>
            </div>

            {acquisitionError && (
              <div className="p-3 rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-200 text-xs font-medium flex items-start gap-2.5 shadow-lg animate-fade-in">
                <span className="text-base shrink-0">⚠️</span>
                <div className="flex-1">
                  <div className="font-bold text-amber-300">Imagery Acquisition Notice</div>
                  <div className="mt-0.5 text-slate-300 leading-relaxed">{acquisitionError}</div>
                </div>
              </div>
            )}

            {/* ACQUISITION BUTTONS FOR TEMPORAL MODE */}
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                className="btn-use-this-area text-xs py-2 px-3 flex-1 cursor-pointer"
                onClick={() => handleAcquireSingleTemporal('before', dateA)}
                disabled={isAcquiring}
              >
                {isAcquiringBefore ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-1"></span>
                    Acquiring Before...
                  </>
                ) : (
                  `🛰️ Acquire Before (${dateA})`
                )}
              </button>

              <button
                type="button"
                className="btn-use-this-area text-xs py-2 px-3 flex-1 cursor-pointer"
                onClick={() => handleAcquireSingleTemporal('after', dateB)}
                disabled={isAcquiring}
              >
                {isAcquiringAfter ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-1"></span>
                    Acquiring After...
                  </>
                ) : (
                  `🛰️ Acquire After (${dateB})`
                )}
              </button>

              <button
                type="button"
                className="w-full btn-use-this-area bg-gradient-to-r from-cyan-600 via-blue-600 to-violet-600 text-xs py-2.5 px-4 font-bold flex items-center justify-center gap-2 cursor-pointer shadow-[0_0_20px_rgba(34,211,238,0.2)]"
                onClick={handleAcquireBothTemporal}
                disabled={isAcquiring}
              >
                {isAcquiring ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                    Acquiring Both Sentinel-2 Scenes ({dateA} vs {dateB})...
                  </>
                ) : (
                  <>
                    <span>⚡ ACQUIRE BOTH TEMPORAL SCENES ({dateA} vs {dateB})</span>
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                    </svg>
                  </>
                )}
              </button>
            </div>
          </div>
        ) : (
          /* STANDARD SINGLE SCENE PREVIEW DRAWER */
          acquiredImage ? (
            <>
              {/* REAL CDSE SENTINEL-2 ACQUIRED SCENE */}
              <div className="preview-thumbnail-box">
                <img
                  src={acquiredImage}
                  alt="Real Sentinel-2 L2A Acquired"
                  className="preview-thumbnail-img"
                />
                <div className="preview-badge-status preview-badge-success">
                  Real CDSE Sentinel-2 Acquired
                </div>
              </div>

              <div className="preview-details-col">
                <div>
                  <div className="preview-region-name text-cyan-300">
                    <span>🛰️ Real Sentinel-2 L2A Scene</span>
                    <span className="text-xs text-slate-400 font-normal">
                      ({selectedLat.toFixed(4)}° N, {selectedLng.toFixed(4)}° E)
                    </span>
                  </div>
                  <div className="preview-meta-tags mt-2">
                    <span className="preview-tag">Source: {acquiredMetadata?.source || "Copernicus Data Space Ecosystem (CDSE)"}</span>
                    <span className="preview-tag">Product: {acquiredMetadata?.product || "Sentinel-2 L2A"}</span>
                    <span className="preview-tag">Date: {acquiredMetadata?.captureDate || "2025-01-01 to 2025-01-31"}</span>
                    <span className="preview-tag">Clouds: {acquiredMetadata?.cloudCover || "< 30%"}</span>
                    <span className="preview-tag">BBox: {acquiredMetadata?.bbox || previewInfo?.bbox?.formattedBBox}</span>
                  </div>
                </div>

                {acquisitionError && (
                  <div className="p-3 rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-200 text-xs font-medium flex items-start gap-2.5 shadow-lg animate-fade-in mb-3">
                    <span className="text-base shrink-0">⚠️</span>
                    <div className="flex-1">
                      <div className="font-bold text-amber-300">Imagery Acquisition Notice</div>
                      <div className="mt-0.5 text-slate-300 leading-relaxed">{acquisitionError}</div>
                    </div>
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-3 mt-1">
                  <button
                    type="button"
                    className="btn-use-this-area cursor-pointer"
                    onClick={handleAcquireRealSentinel2}
                    disabled={isAcquiring}
                  >
                    {isAcquiring ? (
                      <>
                        <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                        Acquiring Real Sentinel-2 (CDSE)...
                      </>
                    ) : (
                      <>
                        <span>🛰️ RE-ACQUIRE REAL SENTINEL-2 (CDSE)</span>
                        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                        </svg>
                      </>
                    )}
                  </button>

                  {previewInfo?.available && (
                    <button
                      type="button"
                      className="btn-jump-demo cursor-pointer"
                      onClick={handleUseThisArea}
                      disabled={isAcquiring}
                    >
                      📁 Switch to Demo Preset
                    </button>
                  )}
                </div>
              </div>
            </>
          ) : previewInfo?.available ? (
            <>
              {/* DEMO PRESET REGION — NOT YET ACQUIRED */}
              <div className="preview-thumbnail-box relative">
                <img
                  src={previewInfo.region.imageUrl}
                  alt={previewInfo.region.name}
                  className="preview-thumbnail-img opacity-60 filter grayscale-[25%]"
                />
                <div className="preview-badge-status bg-amber-500/85 text-amber-100 border border-amber-400/40">
                  Preset Preview — Not Yet Acquired
                </div>
              </div>

              <div className="preview-details-col">
                <div>
                  <div className="preview-region-name">
                    <span>🛰️ {previewInfo.region.name}</span>
                    <span className="text-xs text-cyan-400 font-normal">
                      ({previewInfo.distanceKm} km to center)
                    </span>
                  </div>
                  <div className="preview-meta-tags mt-1">
                    <span className="preview-tag">Preset Region: {previewInfo.region.city}</span>
                    <span className="preview-tag">Target Area: {previewInfo.bbox.areaKm2 ? `${previewInfo.bbox.areaKm2} km²` : '6.25 km²'} AOI (Native ~10m/px)</span>
                    <span className="preview-tag">Status: Live CDSE Acquisition Ready</span>
                    <span className="preview-tag">BBox: {previewInfo.bbox.formattedBBox}</span>
                  </div>
                </div>

                {acquisitionError && (
                  <div className="p-2.5 rounded-lg bg-red-500/15 border border-red-500/30 text-red-300 text-xs font-semibold flex items-center gap-2">
                    <span>⚠️</span> {acquisitionError}
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    className="btn-use-this-area cursor-pointer"
                    onClick={handleAcquireRealSentinel2}
                    disabled={isAcquiring}
                  >
                    {isAcquiring ? (
                      <>
                        <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                        Acquiring Real Sentinel-2 (CDSE)...
                      </>
                    ) : (
                      <>
                        <span>🛰️ ACQUIRE REAL SENTINEL-2 (CDSE)</span>
                        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                        </svg>
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    className="btn-jump-demo cursor-pointer"
                    onClick={handleUseThisArea}
                    disabled={isAcquiring}
                  >
                    📁 Use Demo Region Preset
                  </button>
                </div>
              </div>
            </>
          ) : (
            <>
              {/* NON-DEMO TARGET REGION — NOT YET ACQUIRED */}
              <div className="preview-thumbnail-box">
                <div className="bg-slate-950 flex flex-col items-center justify-center p-4 text-center h-full">
                  <span className="text-3xl mb-2">🛰️</span>
                  <div className="preview-badge-status bg-slate-800 text-slate-300 border border-slate-700">
                    Preset Preview — Not Yet Acquired
                  </div>
                </div>
              </div>

              <div className="preview-details-col">
                <div>
                  <div className="preview-region-name text-cyan-300">
                    <span>🛰️ Custom Target Coordinates</span>
                    <span className="text-xs text-slate-400 font-normal">
                      ({selectedLat.toFixed(4)}° N, {selectedLng.toFixed(4)}° E)
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                    Real Sentinel-2 L2A satellite imagery can be acquired live from Copernicus Data Space for these selected coordinates.
                  </p>
                  <div className="preview-meta-tags mt-2">
                    <span className="preview-tag">Source: Copernicus Data Space</span>
                    <span className="preview-tag">Product: Sentinel-2 L2A</span>
                    <span className="preview-tag">BBox: {previewInfo?.bbox?.formattedBBox}</span>
                  </div>
                </div>

                {acquisitionError && (
                  <div className="p-3 rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-200 text-xs font-medium flex items-start gap-2.5 shadow-lg animate-fade-in mb-3">
                    <span className="text-base shrink-0">⚠️</span>
                    <div className="flex-1">
                      <div className="font-bold text-amber-300">Imagery Acquisition Notice</div>
                      <div className="mt-0.5 text-slate-300 leading-relaxed">{acquisitionError}</div>
                    </div>
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-3 mt-1">
                  <button
                    type="button"
                    className="btn-use-this-area cursor-pointer"
                    onClick={handleAcquireRealSentinel2}
                    disabled={isAcquiring}
                  >
                    {isAcquiring ? (
                      <>
                        <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                        Acquiring Real Sentinel-2 (CDSE)...
                      </>
                    ) : (
                      <>
                        <span>🛰️ ACQUIRE REAL SENTINEL-2 (CDSE)</span>
                        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                        </svg>
                      </>
                    )}
                  </button>

                  {previewInfo?.nearestRegion && (
                    <button
                      type="button"
                      className="btn-jump-demo cursor-pointer"
                      onClick={() =>
                        handleSelectLocation(
                          previewInfo.nearestRegion.lat,
                          previewInfo.nearestRegion.lng,
                          previewInfo.nearestRegion.city
                        )
                      }
                    >
                      📍 Jump to {previewInfo.nearestRegion.city} Demo Region
                    </button>
                  )}
                </div>
              </div>
            </>
          )
        )}
      </div>
    </div>
  );
}
