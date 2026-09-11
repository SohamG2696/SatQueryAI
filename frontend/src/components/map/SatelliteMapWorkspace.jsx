import React, { useState, useEffect, useCallback, useRef } from 'react';
import { MapContainer, TileLayer, Rectangle, Marker, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './InteractiveSatelliteMap.css';
import {
  ArrowLeft, Globe, Satellite, MapPin, Calendar,
  Layers, CloudOff, CheckCircle2, AlertTriangle,
  RefreshCw, Play, ChevronDown, Zap,
} from 'lucide-react';
import { DEMO_REGIONS, getSatellitePreview, urlToFile, formatSentinel2DateWindow, getDistanceKm } from '../../services/satelliteImagery';
import { acquireSentinel2Imagery, getApiBaseUrl, executeQuery } from '../../services/api';
import ResultPanel from '../analysis/ResultPanel';
import ExecutionSummary from '../analysis/ExecutionSummary';

// ── Leaflet helpers ──────────────────────────────────────────────────────────

const customMarkerIcon = L.divIcon({
  className: 'custom-cyan-marker',
  html: `<div class="marker-ping"></div><div class="marker-dot"></div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

function MapClickHandler({ onLocationSelect }) {
  useMapEvents({ click(e) { onLocationSelect(e.latlng.lat, e.latlng.lng); } });
  return null;
}

function MapController({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center?.[0] !== undefined) map.flyTo(center, zoom || 14, { duration: 1.2 });
  }, [center, zoom, map]);
  return null;
}


// ── Component ────────────────────────────────────────────────────────────────
export default function SatelliteMapWorkspace({ onBackToAnalysis, onImageAcquired }) {
  // Map state
  const [selectedLat, setSelectedLat] = useState(12.9716);
  const [selectedLng, setSelectedLng] = useState(77.5946);
  const [mapCenter, setMapCenter] = useState([12.9716, 77.5946]);
  const [mapZoom]   = useState(13);
  const [previewInfo, setPreviewInfo] = useState(null);

  // Search
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError]  = useState(null);

  // Temporal mode
  const [temporalMode, setTemporalMode] = useState(false);
  const [dateA, setDateA] = useState('2020-01-01');
  const [dateB, setDateB] = useState('2025-01-01');

  // Local acquisition state: 'idle' | 'acquiring' | 'acquired' | 'error'
  const [acquisitionState, setAcquisitionState] = useState('idle');
  // Local analysis state: 'idle' | 'analyzing' | 'complete' | 'error'
  const [analysisState, setAnalysisState] = useState('idle');
  const [acqError, setAcqError] = useState(null);
  const [anaError, setAnaError] = useState(null);

  // Acquired images array — [] | [singlePayload] | [beforePayload, afterPayload]
  const [acquiredImages, setAcquiredImages] = useState([]);

  // Analysis
  const [query,        setQuery]        = useState('');
  const [currentStep,  setCurrentStep]  = useState(0);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [executionTime,  setExecutionTime]  = useState(null);

  const belowMapRef          = useRef(null);
  const lastAcquiredCoordRef = useRef(null);

  // ── Location helpers ───────────────────────────────────────────────────────
  const updateLocation = useCallback((lat, lng, forceReset = false) => {
    const fLat = +lat.toFixed(6);
    const fLng = +lng.toFixed(6);
    setSelectedLat(fLat);
    setSelectedLng(fLng);
    setPreviewInfo(getSatellitePreview(fLat, fLng));

    let reset = forceReset;
    if (lastAcquiredCoordRef.current) {
      if (getDistanceKm(lastAcquiredCoordRef.current.lat, lastAcquiredCoordRef.current.lng, fLat, fLng) > 1.0) reset = true;
    } else {
      reset = true;
    }

    if (reset) {
      setAcquiredImages([]);
      setAnalysisResult(null);
      setAcquisitionState('idle');
      setAnalysisState('idle');
      setAcqError(null);
      setAnaError(null);
      lastAcquiredCoordRef.current = null;
    }

    setSearchError(null);
  }, []);

  useEffect(() => { updateLocation(12.9716, 77.5946, true); }, [updateLocation]);

  const handleSelectLocation = (lat, lng, cityName = '') => {
    setMapCenter([lat, lng]);
    updateLocation(lat, lng, true);
    if (cityName) setSearchQuery(cityName);
  };

  const handleSearchSubmit = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    const matched = DEMO_REGIONS.find(r =>
      r.city.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.name.toLowerCase().includes(searchQuery.toLowerCase())
    );
    if (matched) { handleSelectLocation(matched.lat, matched.lng, matched.city); return; }
    try {
      setIsSearching(true); setSearchError(null);
      const res  = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}`);
      const data = await res.json();
      if (data?.length > 0) {
        handleSelectLocation(parseFloat(data[0].lat), parseFloat(data[0].lon), data[0].display_name.split(',')[0]);
      } else {
        setSearchError('Location not found. Try Bengaluru, Mumbai, Delhi\u2026');
      }
    } catch { setSearchError('Geocoding service unavailable.'); }
    finally   { setIsSearching(false); }
  };

  // ── Acquisition ────────────────────────────────────────────────────────────
  const getBboxArray = () => [
    previewInfo.bbox.minLng, previewInfo.bbox.minLat,
    previewInfo.bbox.maxLng, previewInfo.bbox.maxLat,
  ];

  const buildPayload = (imageFile, res, role = null, dateLabel = null) => {
    const captureDate = `${res.date_from?.split('T')[0] || '2025-01-01'} to ${res.date_to?.split('T')[0] || '2025-01-31'}`;
    const roleSuffix  = role ? ` [${role.toUpperCase()}: ${dateLabel}]` : '';
    return {
      file: imageFile,
      id:   `sentinel2_${role || 'single'}_${Date.now()}`,
      name: `Sentinel-2 L2A${roleSuffix} (${selectedLat.toFixed(4)}\u00b0 N, ${selectedLng.toFixed(4)}\u00b0 E)`,
      ext:  '.png',
      size: imageFile.size,
      url:  URL.createObjectURL(imageFile),
      sensor:       'Sentinel-2 L2A (Copernicus Data Space)',
      coordinates:  `${selectedLat.toFixed(4)}\u00b0 N, ${selectedLng.toFixed(4)}\u00b0 E`,
      captureDate,
      temporalRole:  role,
      requestedDate: dateLabel,
      mapMetadata: {
        name:       `Sentinel-2 L2A${roleSuffix}`,
        latitude:   selectedLat,
        longitude:  selectedLng,
        bbox:       previewInfo.bbox.formattedBBox,
        source:     'Copernicus Data Space Ecosystem (CDSE)',
        satellite:  'Sentinel-2',
        product:    'Sentinel-2 L2A',
        sensor:     'Sentinel-2 L2A (Copernicus Data Space)',
        captureDate,
        temporalRole:    role,
        requestedDate:   dateLabel,
        acquisitionMode: 'Real CDSE Sentinel-2 Processing API',
        cloudCover:      '< 30%',
      },
    };
  };

  // Local acquisition callback: stores acquired image in SatelliteMapWorkspace's own state,
  // sets acquisitionState to 'acquired', and does NOT call any navigation/router function.
  const handleAcquisitionSuccess = (payload) => {
    setAcquisitionState('acquired');
    const list = Array.isArray(payload) ? payload : [payload];
    setAcquiredImages(list);
    // Cache silently in parent ref without triggering navigation
    if (onImageAcquired) {
      onImageAcquired(payload);
    }
  };

  const handleAcquire = async () => {
    if (!previewInfo?.bbox) return;
    setAcquisitionState('acquiring');
    setAnalysisState('idle');
    setAcqError(null);
    setAnaError(null);
    setAcquiredImages([]);
    setAnalysisResult(null);

    try {
      const API_BASE = getApiBaseUrl();
      const bbox     = getBboxArray();

      if (temporalMode) {
        const wA = formatSentinel2DateWindow(dateA);
        const wB = formatSentinel2DateWindow(dateB);

        const [resA, resB] = await Promise.all([
          acquireSentinel2Imagery({ bbox, dateFrom: wA.dateFrom, dateTo: wA.dateTo }),
          acquireSentinel2Imagery({ bbox, dateFrom: wB.dateFrom, dateTo: wB.dateTo }),
        ]);

        if (!resA?.success || !resA?.image_url)
          throw new Error(resA?.detail || resA?.message || `BEFORE imagery (${dateA}) could not be acquired.`);
        if (!resB?.success || !resB?.image_url)
          throw new Error(resB?.detail || resB?.message || `AFTER imagery (${dateB}) could not be acquired.`);

        const urlA = resA.image_url.startsWith('http') ? resA.image_url : `${API_BASE}${resA.image_url}`;
        const urlB = resB.image_url.startsWith('http') ? resB.image_url : `${API_BASE}${resB.image_url}`;

        const [fileA, fileB] = await Promise.all([
          urlToFile(urlA, `sentinel2_before_${Date.now()}.png`, 'image/png'),
          urlToFile(urlB, `sentinel2_after_${Date.now()}.png`,  'image/png'),
        ]);

        const before = buildPayload(fileA, resA, 'before', dateA);
        const after  = buildPayload(fileB, resB, 'after',  dateB);

        lastAcquiredCoordRef.current = { lat: selectedLat, lng: selectedLng };
        handleAcquisitionSuccess([before, after]);
      } else {
        const res = await acquireSentinel2Imagery({
          bbox,
          dateFrom: '2025-01-01T00:00:00Z',
          dateTo:   '2025-01-31T23:59:59Z',
        });
        if (!res?.success || !res?.image_url)
          throw new Error(res?.detail || res?.message || 'Sentinel-2 imagery could not be acquired.');

        const fullUrl   = res.image_url.startsWith('http') ? res.image_url : `${API_BASE}${res.image_url}`;
        const imageFile = await urlToFile(fullUrl, `sentinel2_${res.image_id || Date.now()}.png`, 'image/png');
        const payload   = buildPayload(imageFile, res);

        lastAcquiredCoordRef.current = { lat: selectedLat, lng: selectedLng };
        handleAcquisitionSuccess(payload);
      }

      setTimeout(() => belowMapRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 200);
    } catch (err) {
      setAcqError(err.message || 'Satellite imagery could not be acquired.');
      setAcquisitionState('error');
      setTimeout(() => belowMapRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 200);
    }
  };

  // ── Analysis ───────────────────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!query.trim() || !acquiredImages.length) return;
    setAnalysisState('analyzing');
    setAnaError(null);
    setCurrentStep(0);
    setAnalysisResult(null);
    const startTime = performance.now();

    const t1 = setTimeout(() => setCurrentStep(1), 300);
    const t2 = setTimeout(() => setCurrentStep(2), 700);
    const t3 = setTimeout(() => setCurrentStep(3), 1100);

    try {
      const datesList = acquiredImages.map(img => img.captureDate || img.requestedDate).filter(Boolean);
      const metadataPayload = {
        ...(acquiredImages[0]?.mapMetadata || {}),
        dates: datesList.length >= 2 ? datesList : (acquiredImages[0]?.mapMetadata?.dates || []),
      };
      const response = await executeQuery({ query, images: acquiredImages, metadata: metadataPayload });
      const elapsed  = ((performance.now() - startTime) / 1000).toFixed(2) + 's';
      setExecutionTime(elapsed);

      const task       = response.task_detected || response.task || 'vqa';
      const modelsUsed = response.execution_summary?.models_used || response.execution_summary?.models_invoked || [];
      const taskRoute  = response.execution_summary?.task_route || response.execution_summary?.route_selected || 'Standard Pipeline';
      const modelName  = modelsUsed.length > 0 ? modelsUsed.join(', ') : taskRoute !== 'Standard Pipeline' ? taskRoute : 'SatQuery Specialist Model';

      const keyFindings = [];
      if (response.synthesis?.key_findings?.length) {
        keyFindings.push(...response.synthesis.key_findings);
      } else if (response.execution_summary?.parameters?.interpretation?.key_findings?.length) {
        keyFindings.push(...response.execution_summary.parameters.interpretation.key_findings);
      } else {
        keyFindings.push(
          `Task Identified: ${task.toUpperCase()}`,
          `Route Invoked: ${taskRoute}`,
          `Processing Latency: ${response.execution_summary?.processing_time_ms ? `${response.execution_summary.processing_time_ms}ms` : elapsed}`,
        );
      }

      setAnalysisResult({
        rawResponse: response,
        sceneName:   acquiredImages[0]?.name || 'Sentinel-2 Satellite Scene',
        imageCount:  acquiredImages.length,
        query:       query.trim(),
        task_detected: task,
        model:       modelName,
        detectedObjects: task === 'grounding' ? 'Localized Region' : task.replace('_', ' ').toUpperCase(),
        objectType:  acquiredImages[0]?.mapMetadata?.source || (taskRoute !== 'Standard Pipeline' ? taskRoute : 'Geospatial Target'),
        changesDetected: task === 'change_vqa' ? 'Change Detection Applied' : 'Single Scene Inspection',
        confidence:  response.confidence != null ? `${Math.round(response.confidence * 100)}%` : 'Not available',
        message:     response.answer || 'Analysis complete.',
        keyFindings,
        recommendation: response.verification?.notes || response.execution_summary?.parameters?.interpretation?.summary || 'Analysis verified by SatQuery AI pipeline.',
        visual_evidence: response.visual_evidence,
        imageUrl:    acquiredImages[0]?.url || null,
        images:      acquiredImages,
        spectralData: {
          taskDetected:  task,
          modelsInvoked: modelsUsed.join(', ') || 'Active Module',
          routeSelected: taskRoute,
          imageCount:    `${acquiredImages.length} asset${acquiredImages.length !== 1 ? 's' : ''}`,
          source:        acquiredImages[0]?.mapMetadata?.source || 'Copernicus Data Space Ecosystem (CDSE)',
        },
      });
      setAnalysisState('complete');
    } catch (err) {
      setAnaError(err.message || 'Unable to connect to SatQuery AI backend. Ensure the server is running on port 8000.');
      setAnalysisState('error');
    } finally {
      clearTimeout(t1); clearTimeout(t2); clearTimeout(t3);
    }
  };

  // ── Derived ────────────────────────────────────────────────────────────────
  const regionInfo = previewInfo?.region;
  const activeBounds = previewInfo?.bbox?.leafletBounds || [
    [selectedLat - 0.0045, selectedLng - 0.0045],
    [selectedLat + 0.0045, selectedLng + 0.0045],
  ];
  const showBelowMap = acquisitionState !== 'idle';

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="relative min-h-screen text-slate-100 selection:bg-cyan-500/30">
      {/* Background glows */}
      <div className="fixed inset-0 pointer-events-none bg-[radial-gradient(circle_at_top_right,_rgba(6,182,212,0.10),transparent_40%),radial-gradient(circle_at_bottom_left,_rgba(139,92,246,0.08),transparent_45%)]" />

      <div className="relative z-10 mx-auto max-w-7xl px-4 py-6 md:px-8 md:py-8">

        {/* ── HEADER ── */}
        <div className="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-white/10 pb-5">
          <button
            type="button"
            onClick={onBackToAnalysis}
            className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-300 backdrop-blur-md transition-all hover:border-cyan-400/50 hover:bg-cyan-500/10 hover:text-cyan-200 hover:scale-105 cursor-pointer w-fit shadow-lg"
          >
            <ArrowLeft className="h-4 w-4" />
            <span>Back to Analysis</span>
          </button>
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            <span className="inline-block h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
            <span>SENTINEL-2 LIVE ACQUISITION ENGINE</span>
          </div>
        </div>

        {/* ── PAGE TITLE ── */}
        <section className="mb-8 text-center">
          <div className="mx-auto mb-4 inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-4 py-1.5 text-xs font-semibold tracking-wider text-cyan-300 backdrop-blur-md shadow-[0_0_15px_rgba(34,211,238,0.15)]">
            <Globe className="h-3.5 w-3.5 text-cyan-400" />
            <span>INTERACTIVE SATELLITE MAP</span>
          </div>
          <h1 className="mx-auto max-w-4xl text-3xl font-extrabold leading-tight tracking-tight sm:text-4xl md:text-5xl text-white">
            Select a Region.
            <br />
            <span className="bg-gradient-to-r from-cyan-300 via-blue-400 to-violet-400 bg-clip-text text-transparent">
              Acquire Real Sentinel-2 Imagery.
            </span>
          </h1>
          <p className="mx-auto mt-3 max-w-2xl text-xs leading-relaxed text-slate-400 sm:text-sm">
            Click anywhere on the map to select a target location. Acquire live Sentinel-2 L2A imagery from the Copernicus Data Space and analyze it with the SatQuery AI pipeline.
          </p>
        </section>

        {/* ═══════════════════════════════════════════════════════════════════
            TOP SECTION — MAP (left) | SELECTED REGION + ACQUIRE (right)
            The right panel is ALWAYS beside the map regardless of workflow state.
            ═══════════════════════════════════════════════════════════════════ */}
        <div className="grid gap-6 lg:grid-cols-[1fr_360px]">

          {/* ── LEFT: LARGE INTERACTIVE MAP ── */}
          <div className="rounded-3xl border border-white/10 bg-white/[0.02] overflow-hidden shadow-2xl backdrop-blur-xl">
            {/* Search bar */}
            <div className="map-control-header border-b border-white/5">
              <form onSubmit={handleSearchSubmit} className="map-search-box">
                <svg className="map-search-icon" width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input
                  type="text"
                  className="map-search-input"
                  placeholder="Search city or coordinates (e.g. Bengaluru, Mumbai)\u2026"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                {isSearching && (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 text-cyan-400 text-xs animate-pulse">
                    Searching\u2026
                  </div>
                )}
              </form>
              <div className="quick-cities-bar">
                <span className="text-xs text-slate-400 font-medium mr-1">Quick Jump:</span>
                {DEMO_REGIONS.map((reg) => (
                  <button
                    key={reg.id}
                    type="button"
                    className={`quick-city-badge ${previewInfo?.region?.id === reg.id ? 'active' : ''}`}
                    onClick={() => handleSelectLocation(reg.lat, reg.lng, reg.city)}
                  >
                    {reg.city}
                  </button>
                ))}
              </div>
            </div>

            {searchError && (
              <div className="px-4 py-2 bg-amber-500/15 border-b border-amber-500/30 text-amber-300 text-xs flex items-center gap-2">
                <span>&#9888;&#65039;</span> {searchError}
              </div>
            )}

            {/* Leaflet map */}
            <div className="map-stage-container" style={{ height: '520px' }}>
              <MapContainer
                center={mapCenter}
                zoom={mapZoom}
                zoomControl
                scrollWheelZoom
                style={{ height: '100%', width: '100%' }}
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  maxZoom={19}
                />
                <MapController center={mapCenter} zoom={mapZoom} />
                <MapClickHandler onLocationSelect={(lat, lng) => updateLocation(lat, lng)} />
                <Marker position={[selectedLat, selectedLng]} icon={customMarkerIcon} />
                <Rectangle
                  bounds={activeBounds}
                  pathOptions={{ color: '#38bdf8', weight: 2, fillColor: '#0284c7', fillOpacity: 0.15, dashArray: '6, 6' }}
                />
              </MapContainer>

              {/* Telemetry HUD */}
              <div className="telemetry-overlay">
                <div className="telemetry-title">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping inline-block mr-1" />
                  Target Telemetry HUD
                </div>
                <div className="telemetry-grid">
                  <div>
                    <div className="telemetry-item-label">Latitude</div>
                    <div className="telemetry-item-val">{selectedLat.toFixed(5)}&deg; N</div>
                  </div>
                  <div>
                    <div className="telemetry-item-label">Longitude</div>
                    <div className="telemetry-item-val">{selectedLng.toFixed(5)}&deg; E</div>
                  </div>
                  <div>
                    <div className="telemetry-item-label">Target Area</div>
                    <div className="telemetry-item-val">
                      {previewInfo?.bbox?.areaKm2 ? `${previewInfo.bbox.areaKm2} km² (${previewInfo.bbox.sizeKm}×${previewInfo.bbox.sizeKm} km)` : '6.25 km² (2.5×2.5 km)'}
                    </div>
                  </div>
                  <div>
                    <div className="telemetry-item-label">Grid Region</div>
                    <div className="telemetry-item-val">{previewInfo?.region?.id || 'OUT_OF_BOUNDS'}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* ── RIGHT COLUMN: stays beside the map at ALL workflow states ── */}
          <div className="flex flex-col gap-4">

            {/* SELECTED REGION card */}
            <div className="rounded-2xl border border-white/10 bg-[#070d18]/90 p-5 shadow-xl backdrop-blur-xl">
              <div className="flex items-center gap-2 mb-4">
                <MapPin className="h-4 w-4 text-cyan-400 shrink-0" />
                <p className="text-xs font-semibold tracking-[0.25em] text-cyan-400">SELECTED REGION</p>
              </div>

              <h3 className="text-lg font-bold text-white leading-snug mb-1">
                {regionInfo ? regionInfo.name : 'Custom Target Coordinates'}
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                {regionInfo ? regionInfo.city : `${selectedLat.toFixed(4)}\u00b0 N, ${selectedLng.toFixed(4)}\u00b0 E`}
              </p>

              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div className="rounded-lg bg-slate-900/80 border border-slate-800 px-3 py-2">
                  <div className="text-slate-500 mb-0.5 font-semibold uppercase tracking-wider">Latitude</div>
                  <div className="text-cyan-300 font-mono">{selectedLat.toFixed(5)}&deg; N</div>
                </div>
                <div className="rounded-lg bg-slate-900/80 border border-slate-800 px-3 py-2">
                  <div className="text-slate-500 mb-0.5 font-semibold uppercase tracking-wider">Longitude</div>
                  <div className="text-cyan-300 font-mono">{selectedLng.toFixed(5)}&deg; E</div>
                </div>
                <div className="rounded-lg bg-slate-900/80 border border-slate-800 px-3 py-2">
                  <div className="text-slate-500 mb-0.5 font-semibold uppercase tracking-wider">Area</div>
                  <div className="text-white font-mono">{previewInfo?.bbox?.areaKm2 ? `${previewInfo.bbox.areaKm2} km²` : '6.25 km²'}</div>
                </div>
                <div className="rounded-lg bg-slate-900/80 border border-slate-800 px-3 py-2">
                  <div className="text-slate-500 mb-0.5 font-semibold uppercase tracking-wider">BBox</div>
                  <div className="text-white font-mono truncate text-[10px]">
                    {previewInfo?.bbox?.formattedBBox || '\u2014'}
                  </div>
                </div>
                <div className="rounded-lg bg-slate-900/80 border border-slate-800 px-3 py-2 col-span-2">
                  <div className="text-slate-500 mb-0.5 font-semibold uppercase tracking-wider">Sensor</div>
                  <div className="text-white text-[10px]">
                    {regionInfo?.sensor || 'Sentinel-2 L2A (Copernicus Data Space)'}
                  </div>
                </div>
                <div className="rounded-lg bg-slate-900/80 border border-slate-800 px-3 py-2">
                  <div className="text-slate-500 mb-0.5 font-semibold uppercase tracking-wider">Cloud Cover</div>
                  <div className="text-white">{regionInfo?.cloudCover || '&lt; 30%'}</div>
                </div>
                {/* Acquisition status — live */}
                <div className="rounded-lg bg-slate-900/80 border border-slate-800 px-3 py-2">
                  <div className="text-slate-500 mb-0.5 font-semibold uppercase tracking-wider">Status</div>
                  <div className="flex items-center gap-1.5">
                    {acquisitionState === 'idle' && (
                      <><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /><span className="text-emerald-300 text-[10px]">Ready</span></>
                    )}
                    {acquisitionState === 'acquiring' && (
                      <><span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-ping" /><span className="text-sky-300 text-[10px]">Acquiring&hellip;</span></>
                    )}
                    {acquisitionState === 'error' && (
                      <><AlertTriangle className="w-3 h-3 text-amber-400" /><span className="text-amber-300 text-[10px]">Acq Error</span></>
                    )}
                    {acquisitionState === 'acquired' && analysisState === 'idle' && (
                      <><CheckCircle2 className="w-3 h-3 text-emerald-400" /><span className="text-emerald-300 text-[10px]">Acquired</span></>
                    )}
                    {acquisitionState === 'acquired' && analysisState === 'analyzing' && (
                      <><span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-ping" /><span className="text-violet-300 text-[10px]">Analyzing&hellip;</span></>
                    )}
                    {acquisitionState === 'acquired' && analysisState === 'complete' && (
                      <><CheckCircle2 className="w-3 h-3 text-violet-400" /><span className="text-violet-300 text-[10px]">Complete</span></>
                    )}
                    {acquisitionState === 'acquired' && analysisState === 'error' && (
                      <><AlertTriangle className="w-3 h-3 text-red-400" /><span className="text-red-300 text-[10px]">Analysis Error</span></>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* ACQUISITION MODE + ACQUIRE BUTTON card */}
            <div className="rounded-2xl border border-white/10 bg-[#070d18]/90 p-5 shadow-xl backdrop-blur-xl flex flex-col gap-3">
              <p className="text-xs font-semibold tracking-[0.25em] text-slate-400">ACQUISITION MODE</p>

              {/* Mode tabs */}
              <div className="flex items-center bg-slate-950/80 border border-slate-800 p-1 rounded-xl text-xs">
                <button
                  type="button"
                  onClick={() => setTemporalMode(false)}
                  disabled={acquisitionState === 'acquiring' || analysisState === 'analyzing'}
                  className={`flex-1 py-2 px-2 rounded-lg font-semibold transition cursor-pointer flex items-center justify-center gap-1.5 disabled:opacity-50 ${
                    !temporalMode
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-[0_0_15px_rgba(34,211,238,0.15)]'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Satellite className="w-3.5 h-3.5" />
                  Single Scene
                </button>
                <button
                  type="button"
                  onClick={() => setTemporalMode(true)}
                  disabled={acquisitionState === 'acquiring' || analysisState === 'analyzing'}
                  className={`flex-1 py-2 px-2 rounded-lg font-semibold transition cursor-pointer flex items-center justify-center gap-1.5 disabled:opacity-50 ${
                    temporalMode
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-[0_0_15px_rgba(34,211,238,0.15)]'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Calendar className="w-3.5 h-3.5" />
                  Bi-Temporal
                </button>
              </div>

              {/* Date inputs */}
              {temporalMode && (
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-slate-900/80 p-2.5 rounded-xl border border-slate-800">
                    <label className="text-[11px] font-bold text-cyan-400 block mb-1">BEFORE DATE</label>
                    <input
                      type="text"
                      className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2 py-1 text-xs text-white focus:outline-none focus:border-cyan-400"
                      placeholder="e.g. 2020-01-01"
                      value={dateA}
                      onChange={(e) => setDateA(e.target.value)}
                      disabled={acquisitionState === 'acquiring' || analysisState === 'analyzing'}
                    />
                  </div>
                  <div className="bg-slate-900/80 p-2.5 rounded-xl border border-slate-800">
                    <label className="text-[11px] font-bold text-violet-400 block mb-1">AFTER DATE</label>
                    <input
                      type="text"
                      className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2 py-1 text-xs text-white focus:outline-none focus:border-violet-400"
                      placeholder="e.g. 2025-01-01"
                      value={dateB}
                      onChange={(e) => setDateB(e.target.value)}
                      disabled={acquisitionState === 'acquiring' || analysisState === 'analyzing'}
                    />
                  </div>
                </div>
              )}

              {/* ACQUIRE BUTTON */}
              <button
                type="button"
                disabled={acquisitionState === 'acquiring' || analysisState === 'analyzing'}
                onClick={handleAcquire}
                className="w-full flex items-center justify-center gap-2 px-5 py-3.5 rounded-xl font-bold text-sm bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:from-cyan-400 hover:to-blue-500 hover:scale-[1.02] disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all shadow-[0_0_25px_rgba(34,211,238,0.25)] cursor-pointer"
              >
                {acquisitionState === 'acquiring' ? (
                  <><RefreshCw className="w-4 h-4 animate-spin" />Acquiring&hellip;</>
                ) : (
                  <>
                    <Layers className="w-4 h-4" />
                    {temporalMode
                      ? `\u26a1 Acquire Both Scenes (${dateA} vs ${dateB})`
                      : '\uD83D\uDEF0\uFE0F Acquire Real Sentinel-2 (CDSE)'}
                  </>
                )}
              </button>

              {/* Re-acquire shortcut */}
              {acquisitionState === 'acquired' && analysisState !== 'analyzing' && (
                <button
                  type="button"
                  onClick={handleAcquire}
                  className="w-full flex items-center justify-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 border border-slate-700 hover:border-cyan-400/40 hover:text-cyan-300 transition cursor-pointer"
                >
                  <RefreshCw className="w-3 h-3" />
                  Re-Acquire New Scene
                </button>
              )}
            </div>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════
            BELOW MAP SECTION — state-driven workflow output
            ═══════════════════════════════════════════════════════════════════ */}
        {showBelowMap && (
          <div ref={belowMapRef} className="mt-10 space-y-8">

            {/* Section label */}
            <div className="flex items-center gap-4">
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-cyan-400/30 to-cyan-400/50" />
              <div className="flex items-center gap-2 text-xs font-mono text-cyan-400 tracking-[0.25em]">
                <ChevronDown className="w-3.5 h-3.5" />
                ACQUISITION WORKFLOW
              </div>
              <div className="h-px flex-1 bg-gradient-to-l from-transparent via-violet-400/30 to-violet-400/50" />
            </div>

            {/* ── STATE: ACQUIRING (loading) ── */}
            {acquisitionState === 'acquiring' && (
              <div className="rounded-3xl border border-cyan-400/20 bg-[#070d18]/90 p-10 backdrop-blur-xl shadow-2xl text-center">
                <div className="mx-auto mb-6 w-16 h-16 rounded-full border-2 border-cyan-400/30 border-t-cyan-400 animate-spin" />
                <p className="text-xs font-semibold tracking-[0.3em] text-cyan-400 mb-2">ACQUIRING SENTINEL-2</p>
                <h3 className="text-xl font-bold text-white mb-6">Connecting to Copernicus Data Space&hellip;</h3>
                <div className="max-w-sm mx-auto space-y-2 text-xs text-slate-400">
                  {[
                    'Searching available acquisitions for selected AOI',
                    'Applying cloud cover filter (< 30%)',
                    'Preparing Sentinel-2 L2A imagery',
                  ].map((msg, i) => (
                    <div key={msg} className="flex items-center gap-2 justify-center">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" style={{ animationDelay: `${i * 0.3}s` }} />
                      {msg}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── STATE: ERROR ── */}
            {acquisitionState === 'error' && (
              <div className="rounded-3xl border border-amber-500/30 bg-[#070d18]/90 p-8 backdrop-blur-xl shadow-2xl">
                <div className="flex items-start gap-4 mb-6">
                  <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 shrink-0">
                    <AlertTriangle className="w-6 h-6 text-amber-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold tracking-[0.25em] text-amber-400 mb-1">SENTINEL-2 ACQUISITION FAILED</p>
                    <h3 className="text-lg font-bold text-white mb-3">Unable to acquire imagery for selected AOI</h3>
                    <p className="text-sm text-slate-300 leading-relaxed bg-amber-950/20 border border-amber-500/20 rounded-xl p-3 font-mono break-words">
                      {acqError}
                    </p>
                    <p className="mt-3 text-xs text-amber-400/80">
                      💡 Try selecting a different season (e.g. Jan&ndash;Mar for low cloud cover), a nearby region, or use the Quick Jump buttons above.
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleAcquire}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-sm bg-amber-500/20 border border-amber-500/40 text-amber-300 hover:bg-amber-500/30 transition cursor-pointer"
                >
                  <RefreshCw className="w-4 h-4" />
                  Retry Acquisition
                </button>
              </div>
            )}

            {/* ── STATE: ACQUIRED ── */}
            {acquisitionState === 'acquired' && (
              <div className="space-y-8">

                {/* ── TWO-COLUMN SIDE-BY-SIDE GRID (DESKTOP) / STACKED (MOBILE) ── */}
                <div className="grid gap-6 lg:grid-cols-2 items-stretch">

                  {/* LEFT COLUMN: ACQUIRED SENTINEL-2 IMAGERY */}
                  <div className="rounded-3xl border border-cyan-400/20 bg-[#070d18]/90 overflow-hidden shadow-2xl backdrop-blur-xl flex flex-col justify-between">
                    <div>
                      <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between flex-wrap gap-3">
                        <div>
                          <p className="text-xs font-semibold tracking-[0.28em] text-cyan-400">ACQUIRED SENTINEL-2 IMAGERY</p>
                          <h3 className="mt-0.5 text-lg font-bold text-white">
                            {temporalMode ? 'Bi-Temporal Change Analysis Pair' : 'Single Scene \u00b7 Real CDSE Acquisition'}
                          </h3>
                        </div>
                        <div className="flex items-center gap-1.5 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-300 shrink-0">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Real CDSE Data
                        </div>
                      </div>

                      <div className="p-6 space-y-4">
                        {temporalMode && acquiredImages.length === 2 ? (
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            {[
                              { payload: acquiredImages[0], label: 'BEFORE', color: 'cyan' },
                              { payload: acquiredImages[1], label: 'AFTER',  color: 'violet' },
                            ].map(({ payload, label, color }) => (
                              <div key={label} className="space-y-2">
                                <div className="flex items-center gap-2">
                                  <span className={`text-[11px] font-bold tracking-wider text-${color}-400`}>{label}</span>
                                  <span className="text-[10px] text-slate-500 font-mono">{payload.requestedDate}</span>
                                </div>
                                <div className={`rounded-xl overflow-hidden border border-${color}-400/20 shadow-[0_0_20px_rgba(34,211,238,0.1)]`}>
                                  <img
                                    src={payload.url}
                                    alt={`${label} — Sentinel-2 L2A`}
                                    className="w-full object-cover"
                                    style={{ maxHeight: '200px' }}
                                  />
                                </div>
                                <div className="flex flex-wrap gap-1.5 text-[10px]">
                                  <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700 truncate max-w-[130px]">
                                    {payload.captureDate}
                                  </span>
                                  <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                                    Sentinel-2 L2A
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          acquiredImages[0] && (
                            <div className="space-y-4">
                              <div className="rounded-xl overflow-hidden border border-cyan-400/20 shadow-[0_0_25px_rgba(34,211,238,0.12)]">
                                <img
                                  src={acquiredImages[0].url}
                                  alt="Acquired Sentinel-2 L2A"
                                  className="w-full object-cover"
                                  style={{ maxHeight: '260px' }}
                                />
                              </div>
                              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                                {[
                                  ['Source',      'Copernicus Data Space'],
                                  ['Product',     'Sentinel-2 L2A'],
                                  ['Capture',     acquiredImages[0].captureDate],
                                  ['Cloud Cover', '< 30%'],
                                  ['Location',    `${selectedLat.toFixed(4)}\u00b0 N, ${selectedLng.toFixed(4)}\u00b0 E`],
                                  ['Sensor',      'Optical Multi-Spectral'],
                                ].map(([lbl, val]) => (
                                  <div key={lbl} className="rounded-lg bg-slate-900/80 border border-slate-800 px-3 py-2 text-[11px]">
                                    <div className="text-slate-500 mb-0.5 font-semibold uppercase tracking-wider text-[9px]">{lbl}</div>
                                    <div className="text-white font-mono text-[10px] truncate" title={val}>{val}</div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  </div>

                  {/* RIGHT COLUMN: AI INFERENCE INTERFACE / ANALYSIS QUERY */}
                  <div className="rounded-3xl border border-violet-400/20 bg-[#070d18]/90 p-6 md:p-8 shadow-2xl backdrop-blur-xl flex flex-col justify-between">
                    <div>
                      <div className="mb-5 flex items-start justify-between">
                        <div>
                          <p className="text-xs font-semibold tracking-[0.28em] text-violet-400 mb-1">AI INFERENCE INTERFACE</p>
                          <h3 className="text-2xl font-bold text-white">Ask the Satellite</h3>
                          <p className="mt-1 text-xs text-slate-400">
                            Enter your natural language query. The SatQuery AI pipeline will analyze the acquired imagery.
                          </p>
                        </div>
                        <div className="rounded-lg border border-violet-400/20 bg-violet-400/5 p-2 text-violet-400 shrink-0">
                          <Zap className="h-5 w-5" />
                        </div>
                      </div>

                      {analysisState === 'error' && (
                        <div className="mb-4 p-3 rounded-xl bg-red-500/15 border border-red-500/30 text-red-300 text-xs flex items-start gap-2.5">
                          <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                          <div>
                            <div className="font-bold mb-0.5">Analysis Error</div>
                            <div className="text-slate-300 leading-relaxed">{anaError}</div>
                          </div>
                        </div>
                      )}

                      <div className="relative mb-4">
                        <textarea
                          rows={4}
                          disabled={analysisState === 'analyzing'}
                          className="w-full resize-none rounded-2xl border border-white/15 bg-black/40 p-4 text-sm text-white placeholder:text-slate-500 outline-none transition duration-200 focus:border-violet-400 focus:ring-1 focus:ring-violet-400/50 disabled:opacity-60"
                          placeholder={"Ask the satellite what you want to analyze\u2026\n\nExamples: Identify buildings \u00b7 Detect urban expansion \u00b7 Analyze vegetation changes \u00b7 How many buildings are visible?"}
                          value={query}
                          onChange={(e) => setQuery(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                              if (query.trim() && analysisState !== 'analyzing') handleAnalyze();
                            }
                          }}
                        />
                        <div className="absolute bottom-3 right-3 text-[10px] text-slate-500 font-mono pointer-events-none">
                          Ctrl+Enter to run
                        </div>
                      </div>

                      {/* Quick suggestion chips */}
                      <div className="mb-5">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-medium tracking-wider text-slate-400 flex items-center gap-1.5">
                            <span className="text-violet-400">&#10022;</span>
                            RECOMMENDED QUERIES
                          </span>
                          <span className="text-[11px] text-slate-500">Click to apply</span>
                        </div>
                        <div className="flex flex-col gap-2">
                          {[
                            'Identify buildings in this region',
                            'Detect urban expansion',
                            'Analyze vegetation changes',
                            'How many buildings are visible?',
                            'Analyze land-cover changes',
                          ].map((s) => (
                            <button
                              key={s}
                              type="button"
                              disabled={analysisState === 'analyzing'}
                              onClick={() => setQuery(s)}
                              className={`group flex items-center justify-between rounded-xl border px-3.5 py-2 text-left text-xs transition-all duration-200 cursor-pointer disabled:opacity-50 ${
                                query === s
                                  ? 'border-violet-400/60 bg-violet-950/40 text-violet-200 shadow-[0_0_15px_rgba(167,139,250,0.15)]'
                                  : 'border-white/10 bg-white/[0.02] text-slate-300 hover:border-violet-400/30 hover:bg-white/[0.05] hover:text-white'
                              }`}
                            >
                              <span className="truncate">{s}</span>
                              <span className="ml-2 text-violet-400 opacity-0 transition-opacity group-hover:opacity-100">&#8629;</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <button
                      type="button"
                      disabled={!query.trim() || analysisState === 'analyzing'}
                      onClick={handleAnalyze}
                      className="w-full flex items-center justify-center gap-2 px-6 py-4 rounded-2xl font-bold text-base bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 text-white hover:from-violet-500 hover:to-fuchsia-500 hover:scale-[1.01] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all shadow-[0_0_30px_rgba(139,92,246,0.35)] cursor-pointer"
                    >
                      {analysisState === 'analyzing' ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
                          <span>Running Analysis Pipeline...</span>
                        </>
                      ) : (
                        <>
                          <Zap className="w-5 h-5" />
                          <span>ANALYZE QUERY</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* ── ANALYZING: pipeline loading state via ResultPanel ── */}
                {analysisState === 'analyzing' && (
                  <div className="rounded-3xl border border-violet-400/20 bg-[#070d18]/90 p-8 backdrop-blur-xl shadow-2xl">
                    <ResultPanel
                      analysisResult={null}
                      isAnalyzing={true}
                      currentStep={currentStep}
                      error={null}
                      images={acquiredImages}
                    />
                  </div>
                )}

                {/* ── COMPLETE: Mission Telemetry + Intelligence Report ── */}
                {analysisState === 'complete' && (
                  <div className="space-y-8">
                    <div className="flex items-center gap-4">
                      <div className="h-px flex-1 bg-gradient-to-r from-transparent via-violet-400/30 to-violet-400/50" />
                      <span className="text-xs font-mono text-violet-400 tracking-[0.25em]">INTELLIGENCE OUTPUT</span>
                      <div className="h-px flex-1 bg-gradient-to-l from-transparent via-cyan-400/30 to-cyan-400/50" />
                    </div>

                    {/* Full Intelligence Report */}
                    <ResultPanel
                      analysisResult={analysisResult}
                      isAnalyzing={false}
                      currentStep={0}
                      error={null}
                      images={acquiredImages}
                    />

                    {/* Mission Telemetry */}
                    <ExecutionSummary
                      analysisResult={analysisResult}
                      isAnalyzing={false}
                      executionTime={executionTime}
                      images={acquiredImages}
                    />

                    {/* Run another query CTA */}
                    <div className="rounded-2xl border border-white/10 bg-[#070d18]/60 p-5 flex flex-col sm:flex-row items-center justify-between gap-4 backdrop-blur-xl">
                      <div>
                        <p className="text-sm font-semibold text-white">Run another query on this scene?</p>
                        <p className="text-xs text-slate-400 mt-0.5">The acquired Sentinel-2 imagery is still loaded.</p>
                      </div>
                      <button
                        type="button"
                        onClick={() => { setAnalysisState('idle'); setAnalysisResult(null); setQuery(''); }}
                        className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold bg-gradient-to-r from-cyan-600 to-blue-600 text-white hover:from-cyan-500 hover:to-blue-500 transition cursor-pointer shrink-0"
                      >
                        <Play className="w-3.5 h-3.5" />
                        New Query
                      </button>
                    </div>

                    <button
                      type="button"
                      onClick={onBackToAnalysis}
                      className="w-full flex items-center justify-center gap-2 px-5 py-3 rounded-xl font-bold text-sm border border-slate-700 text-slate-300 hover:border-cyan-400/40 hover:text-cyan-300 transition cursor-pointer"
                    >
                      <ArrowLeft className="w-4 h-4" />
                      Back to Analysis Workspace
                    </button>
                  </div>
                )}

              </div>
            )}
          </div>
        )}

        {/* ── FOOTER ── */}
        <footer className="mt-16 border-t border-white/10 pt-6 pb-8 text-xs text-slate-500 flex flex-col md:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="text-slate-300 font-semibold">SatQuery AI</span>
            <span>&mdash; Live Sentinel-2 Acquisition via Copernicus Data Space</span>
          </div>
          <div className="flex items-center gap-3 font-mono text-[11px] text-slate-400">
            <span className="flex items-center gap-1.5">
              <CloudOff className="h-3.5 w-3.5 text-sky-400" />
              Cloud Filter &lt; 30%
            </span>
            <span>&bull;</span>
            <span>2.5 km AOI (Native ~10m/px)</span>
            <span>&bull;</span>
            <span>Sentinel-2 L2A</span>
          </div>
        </footer>

      </div>
    </div>
  );
}
