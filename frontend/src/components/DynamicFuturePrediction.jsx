import React, { useState } from "react";
import {
  Upload,
  Calendar,
  Sparkles,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Info,
  Layers,
  Plus,
  Trash2,
  FileSpreadsheet,
  Building2,
  Trees,
  Waves,
  FileCheck,
  TrendingUp,
} from "lucide-react";
import { predictFutureDynamic } from "../services/api";

const DEFAULT_SLOTS = [
  { id: 1, file: null, year: 2019 },
  { id: 2, file: null, year: 2021 },
  { id: 3, file: null, year: 2023 },
  { id: 4, file: null, year: 2025 },
];

export default function DynamicFuturePrediction() {
  const [slots, setSlots] = useState(DEFAULT_SLOTS);
  const [query, setQuery] = useState("Predict the change of buildings and trees in 2027");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (slotId, file) => {
    setSlots((prev) =>
      prev.map((s) => (s.id === slotId ? { ...s, file } : s))
    );
    setError(null);
  };

  const handleYearChange = (slotId, year) => {
    setSlots((prev) =>
      prev.map((s) => (s.id === slotId ? { ...s, year: Number(year) } : s))
    );
  };

  const handleAddSlot = () => {
    if (slots.length >= 5) return;
    const maxYear = Math.max(...slots.map((s) => s.year || 2025));
    setSlots((prev) => [
      ...prev,
      { id: Date.now(), file: null, year: maxYear + 2 },
    ]);
  };

  const handleRemoveSlot = (slotId) => {
    if (slots.length <= 2) {
      setError("At least 2 historical image slots are required.");
      return;
    }
    setSlots((prev) => prev.filter((s) => s.id !== slotId));
  };

  const handleForecast = async () => {
    setError(null);
    const validSlots = slots.filter((s) => s.file !== null && s.year);

    if (validSlots.length < 2) {
      setError("Please upload at least 2 Sentinel-2 GeoTIFF images with their corresponding acquisition years.");
      return;
    }

    if (!query || !query.trim()) {
      setError("Please enter a prediction query prompt.");
      return;
    }

    setLoading(true);

    try {
      const response = await predictFutureDynamic({
        images: validSlots.map((s) => s.file),
        years: validSlots.map((s) => s.year),
        query: query.trim(),
      });

      setResult(response);
    } catch (err) {
      setError(err.message || "Dynamic future prediction failed.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceBadge = (confStr) => {
    switch (confStr?.toLowerCase()) {
      case "high":
        return {
          bg: "bg-emerald-500/10 border-emerald-500/30 text-emerald-400",
          icon: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />,
          label: "HIGH CONFIDENCE",
        };
      case "moderate":
        return {
          bg: "bg-amber-500/10 border-amber-500/30 text-amber-400",
          icon: <Info className="w-3.5 h-3.5 text-amber-400" />,
          label: "MODERATE CONFIDENCE",
        };
      case "low":
        return {
          bg: "bg-rose-500/10 border-rose-500/30 text-rose-400",
          icon: <AlertCircle className="w-3.5 h-3.5 text-rose-400" />,
          label: "LOW CONFIDENCE",
        };
      default:
        return {
          bg: "bg-cyan-500/10 border-cyan-500/30 text-cyan-400",
          icon: <Info className="w-3.5 h-3.5 text-cyan-400" />,
          label: "STATISTICAL FORECAST",
        };
    }
  };

  // Render SVG Trajectory Chart
  const renderChart = () => {
    const params = result?.execution_summary?.parameters;
    if (!params) return null;

    const historical = params.historical_data || [];
    const predictions = params.predictions || {};
    const targetYear = params.target_year;

    if (!historical.length) return null;

    const points = historical.map((h) => ({
      year: h.year,
      built_up_pct: h.built_up_pct,
      vegetation_pct: h.vegetation_pct,
      water_pct: h.water_pct,
      isForecast: false,
    }));

    if (targetYear) {
      const forecastPoint = {
        year: targetYear,
        isForecast: true,
      };
      if (predictions.built_up_pct) forecastPoint.built_up_pct = predictions.built_up_pct.value;
      if (predictions.vegetation_pct) forecastPoint.vegetation_pct = predictions.vegetation_pct.value;
      if (predictions.water_pct) forecastPoint.water_pct = predictions.water_pct.value;
      points.push(forecastPoint);
    }

    const padding = { top: 30, right: 40, bottom: 40, left: 50 };
    const width = 640;
    const height = 260;
    const innerW = width - padding.left - padding.right;
    const innerH = height - padding.top - padding.bottom;

    const minYear = Math.min(...points.map((p) => p.year));
    const maxYear = Math.max(...points.map((p) => p.year));

    const getX = (yr) =>
      minYear === maxYear
        ? padding.left + innerW / 2
        : padding.left + ((yr - minYear) / (maxYear - minYear)) * innerW;

    const getY = (val) =>
      val == null ? padding.top + innerH : padding.top + innerH - (val / 100) * innerH;

    const histPts = points.filter((p) => !p.isForecast);
    const forePt = points.find((p) => p.isForecast);
    const lastHistPt = histPts[histPts.length - 1];

    const buildPath = (seriesKey, pts) =>
      pts
        .filter((p) => p[seriesKey] != null)
        .map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.year)} ${getY(p[seriesKey])}`)
        .join(" ");

    const buildForecastPath = (seriesKey) => {
      if (!lastHistPt || !forePt || forePt[seriesKey] == null) return "";
      return `M ${getX(lastHistPt.year)} ${getY(lastHistPt[seriesKey])} L ${getX(forePt.year)} ${getY(forePt[seriesKey])}`;
    };

    return (
      <div className="w-full overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto text-xs select-none">
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map((val) => (
            <g key={val}>
              <line
                x1={padding.left}
                y1={getY(val)}
                x2={width - padding.right}
                y2={getY(val)}
                stroke="rgba(255,255,255,0.06)"
                strokeDasharray="4 4"
              />
              <text x={padding.left - 10} y={getY(val) + 4} fill="#64748b" textAnchor="end">
                {val}%
              </text>
            </g>
          ))}

          {/* X Axis Labels */}
          {points.map((p) => (
            <g key={p.year}>
              <line
                x1={getX(p.year)}
                y1={height - padding.bottom}
                x2={getX(p.year)}
                y2={height - padding.bottom + 5}
                stroke="#475569"
              />
              <text
                x={getX(p.year)}
                y={height - padding.bottom + 20}
                fill={p.isForecast ? "#38bdf8" : "#94a3b8"}
                fontWeight={p.isForecast ? "700" : "400"}
                textAnchor="middle"
              >
                {p.year} {p.isForecast ? "(Forecast)" : ""}
              </text>
            </g>
          ))}

          {/* Lines */}
          {predictions.built_up_pct && (
            <>
              <path d={buildPath("built_up_pct", histPts)} fill="none" stroke="#06b6d4" strokeWidth="2.5" />
              {forePt && <path d={buildForecastPath("built_up_pct")} fill="none" stroke="#06b6d4" strokeWidth="2.5" strokeDasharray="6 4" />}
            </>
          )}

          {predictions.vegetation_pct && (
            <>
              <path d={buildPath("vegetation_pct", histPts)} fill="none" stroke="#10b981" strokeWidth="2.5" />
              {forePt && <path d={buildForecastPath("vegetation_pct")} fill="none" stroke="#10b981" strokeWidth="2.5" strokeDasharray="6 4" />}
            </>
          )}

          {predictions.water_pct && (
            <>
              <path d={buildPath("water_pct", histPts)} fill="none" stroke="#3b82f6" strokeWidth="2.5" />
              {forePt && <path d={buildForecastPath("water_pct")} fill="none" stroke="#3b82f6" strokeWidth="2.5" strokeDasharray="6 4" />}
            </>
          )}

          {/* Points */}
          {points.map((p) => (
            <g key={p.year}>
              {p.built_up_pct != null && (
                <circle
                  cx={getX(p.year)}
                  cy={getY(p.built_up_pct)}
                  r={p.isForecast ? 6 : 4}
                  fill={p.isForecast ? "#06b6d4" : "#083344"}
                  stroke="#06b6d4"
                  strokeWidth={p.isForecast ? 3 : 2}
                />
              )}
              {p.vegetation_pct != null && (
                <circle
                  cx={getX(p.year)}
                  cy={getY(p.vegetation_pct)}
                  r={p.isForecast ? 6 : 4}
                  fill={p.isForecast ? "#10b981" : "#064e3b"}
                  stroke="#10b981"
                  strokeWidth={p.isForecast ? 3 : 2}
                />
              )}
            </g>
          ))}
        </svg>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-2 text-xs text-slate-400">
          {predictions.built_up_pct && (
            <div className="flex items-center gap-2">
              <span className="w-3 h-0.5 bg-cyan-500 inline-block"></span>
              <span>Built-up %</span>
            </div>
          )}
          {predictions.vegetation_pct && (
            <div className="flex items-center gap-2">
              <span className="w-3 h-0.5 bg-emerald-500 inline-block"></span>
              <span>Vegetation %</span>
            </div>
          )}
          {predictions.water_pct && (
            <div className="flex items-center gap-2">
              <span className="w-3 h-0.5 bg-blue-500 inline-block"></span>
              <span>Water %</span>
            </div>
          )}
          <div className="flex items-center gap-2 border-l border-slate-700 pl-4">
            <span className="w-4 h-0 border-t-2 border-dashed border-cyan-400 inline-block"></span>
            <span className="text-cyan-400 font-medium">Extrapolated Trend</span>
          </div>
        </div>
      </div>
    );
  };

  const params = result?.execution_summary?.parameters;

  return (
    <div className="w-full bg-slate-950/80 backdrop-blur-xl border border-slate-800/80 rounded-2xl p-6 shadow-2xl space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 rounded-xl text-cyan-400 shadow-inner">
            <Upload className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white tracking-wide flex items-center gap-2">
              Dynamic Multi-Image Timeline Forecaster
              <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                Custom Upload Mode
              </span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Upload 3–5 Sentinel-2 GeoTIFFs (with SCL band 15) to predict custom future land-cover trends.
            </p>
          </div>
        </div>
      </div>

      {/* Upload Slots Section */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs font-medium text-slate-300">
          <span className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            Upload Timeline Imagery (Sentinel-2 GeoTIFFs)
          </span>
          <button
            onClick={handleAddSlot}
            disabled={slots.length >= 5}
            className="flex items-center gap-1 text-cyan-400 hover:text-cyan-300 disabled:opacity-40 cursor-pointer font-semibold"
          >
            <Plus className="w-3.5 h-3.5" />
            Add Image Slot ({slots.length}/5)
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
          {slots.map((slot, index) => (
            <div
              key={slot.id}
              className={`relative bg-slate-900/80 border ${
                slot.file ? "border-cyan-500/50 bg-cyan-950/20" : "border-slate-800"
              } rounded-xl p-3.5 flex flex-col justify-between gap-3 transition-all`}
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-mono text-slate-400">
                  Image #{index + 1}
                </span>
                {slots.length > 2 && (
                  <button
                    onClick={() => handleRemoveSlot(slot.id)}
                    className="text-slate-500 hover:text-rose-400 transition-colors p-1"
                    title="Remove slot"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              {/* File Input / Drop Area */}
              <label className="flex flex-col items-center justify-center border border-dashed border-slate-700 hover:border-cyan-500/60 rounded-lg p-3 cursor-pointer bg-slate-950/50 hover:bg-slate-900/60 transition-colors group text-center">
                <input
                  type="file"
                  accept=".tif,.tiff"
                  onChange={(e) => handleFileChange(slot.id, e.target.files[0] || null)}
                  className="hidden"
                />
                {slot.file ? (
                  <div className="space-y-1">
                    <FileCheck className="w-5 h-5 text-emerald-400 mx-auto" />
                    <p className="text-[11px] font-medium text-slate-200 truncate max-w-[120px]">
                      {slot.file.name}
                    </p>
                    <p className="text-[10px] text-slate-500 font-mono">
                      {(slot.file.size / (1024 * 1024)).toFixed(1)} MB
                    </p>
                  </div>
                ) : (
                  <div className="space-y-1">
                    <Upload className="w-5 h-5 text-slate-500 group-hover:text-cyan-400 mx-auto transition-colors" />
                    <p className="text-[11px] text-slate-400 group-hover:text-slate-200">
                      Select GeoTIFF
                    </p>
                  </div>
                )}
              </label>

              {/* Year Input */}
              <div className="flex items-center justify-between bg-slate-950/80 border border-slate-800 rounded-lg px-2.5 py-1.5">
                <span className="text-[11px] text-slate-400 flex items-center gap-1">
                  <Calendar className="w-3 h-3 text-cyan-400" />
                  Year:
                </span>
                <input
                  type="number"
                  min={2015}
                  max={2026}
                  value={slot.year}
                  onChange={(e) => handleYearChange(slot.id, e.target.value)}
                  className="w-14 bg-transparent text-xs font-mono font-bold text-cyan-300 text-right focus:outline-none"
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Query Bar & Action */}
      <div className="space-y-2">
        <label className="text-xs font-medium text-slate-300">
          Natural Language Prediction Query
        </label>
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. Predict the change of buildings and trees in 2027"
            className="flex-1 bg-slate-900/90 border border-slate-800 focus:border-cyan-500/60 rounded-xl px-4 py-2.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none"
          />
          <button
            onClick={handleForecast}
            disabled={loading}
            className="flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-semibold text-xs px-6 py-2.5 rounded-xl transition-all shadow-lg shadow-cyan-500/20 active:scale-95 disabled:opacity-50 cursor-pointer"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            Execute Forecast
          </button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-xs">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Results Block */}
      {result && params && (
        <div className="space-y-6 pt-2">
          {/* Intelligence Explanation Box */}
          <div className="p-4 bg-slate-900/80 border border-slate-800 rounded-xl space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs font-semibold text-cyan-300">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span>AI Agentic Intelligence Summary</span>
              </div>
              <div className="text-[11px] font-mono text-slate-400">
                Latency: {result.execution_summary?.processing_time_ms}ms
              </div>
            </div>
            <p className="text-xs text-slate-200 leading-relaxed">
              {result.answer || params.explanation}
            </p>

            {/* Unsupported categories notice */}
            {params.unsupported_categories && params.unsupported_categories.length > 0 && (
              <div className="text-[11px] text-amber-400 flex items-center gap-1.5 pt-1">
                <Info className="w-3.5 h-3.5 shrink-0" />
                <span>
                  Note: [{params.unsupported_categories.join(", ")}] not yet supported for prediction.
                </span>
              </div>
            )}
          </div>

          {/* Line Chart */}
          <div className="bg-slate-900/50 border border-slate-800/60 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
                <TrendingUp className="w-4 h-4 text-cyan-400" />
                <span>Extrapolated Trajectory (Target: {params.target_year})</span>
              </div>
            </div>
            {renderChart()}
          </div>

          {/* Prediction Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {params.predictions?.built_up_pct && (
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-400 flex items-center gap-1.5">
                    <Building2 className="w-4 h-4 text-cyan-400" />
                    Built-up % ({params.target_year})
                  </span>
                  {(() => {
                    const badge = getConfidenceBadge(params.predictions.built_up_pct.confidence);
                    return (
                      <span className={`px-2 py-0.5 text-[10px] font-semibold border rounded-full ${badge.bg}`}>
                        {badge.label}
                      </span>
                    );
                  })()}
                </div>
                <div className="text-2xl font-bold text-cyan-300 font-mono">
                  {params.predictions.built_up_pct.value}%
                </div>
                <div className="text-[11px] text-slate-400 font-mono">
                  Slope: {params.predictions.built_up_pct.annual_slope >= 0 ? "+" : ""}
                  {params.predictions.built_up_pct.annual_slope}%/yr | R²={params.predictions.built_up_pct.r2}
                </div>
              </div>
            )}

            {params.predictions?.vegetation_pct && (
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-400 flex items-center gap-1.5">
                    <Trees className="w-4 h-4 text-emerald-400" />
                    Vegetation % ({params.target_year})
                  </span>
                  {(() => {
                    const badge = getConfidenceBadge(params.predictions.vegetation_pct.confidence);
                    return (
                      <span className={`px-2 py-0.5 text-[10px] font-semibold border rounded-full ${badge.bg}`}>
                        {badge.label}
                      </span>
                    );
                  })()}
                </div>
                <div className="text-2xl font-bold text-emerald-400 font-mono">
                  {params.predictions.vegetation_pct.value}%
                </div>
                <div className="text-[11px] text-slate-400 font-mono">
                  Slope: {params.predictions.vegetation_pct.annual_slope >= 0 ? "+" : ""}
                  {params.predictions.vegetation_pct.annual_slope}%/yr | R²={params.predictions.vegetation_pct.r2}
                </div>
              </div>
            )}

            {params.predictions?.water_pct && (
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-400 flex items-center gap-1.5">
                    <Waves className="w-4 h-4 text-blue-400" />
                    Water % ({params.target_year})
                  </span>
                  {(() => {
                    const badge = getConfidenceBadge(params.predictions.water_pct.confidence);
                    return (
                      <span className={`px-2 py-0.5 text-[10px] font-semibold border rounded-full ${badge.bg}`}>
                        {badge.label}
                      </span>
                    );
                  })()}
                </div>
                <div className="text-2xl font-bold text-blue-400 font-mono">
                  {params.predictions.water_pct.value}%
                </div>
                <div className="text-[11px] text-slate-400 font-mono">
                  Slope: {params.predictions.water_pct.annual_slope >= 0 ? "+" : ""}
                  {params.predictions.water_pct.annual_slope}%/yr | R²={params.predictions.water_pct.r2}
                </div>
              </div>
            )}
          </div>

          {/* Historical Change Table */}
          {params.historical_data && params.historical_data.length > 0 && (
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-3">
              <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
                <FileSpreadsheet className="w-4 h-4 text-cyan-400" />
                <span>Extracted Historical Land-Cover Timeline (SCL Band 15)</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 font-sans">
                      <th className="py-2 px-3">Year</th>
                      <th className="py-2 px-3">Built-up %</th>
                      <th className="py-2 px-3">Vegetation %</th>
                      <th className="py-2 px-3">Water %</th>
                      <th className="py-2 px-3">Valid Pixels</th>
                      <th className="py-2 px-3">Quality</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-200">
                    {params.historical_data.map((row, idx, arr) => {
                      const prev = idx > 0 ? arr[idx - 1] : null;
                      const builtDelta = prev ? row.built_up_pct - prev.built_up_pct : null;
                      const vegDelta = prev ? row.vegetation_pct - prev.vegetation_pct : null;

                      return (
                        <tr key={row.year} className="hover:bg-slate-800/30">
                          <td className="py-2 px-3 font-bold text-cyan-300">{row.year}</td>
                          <td className="py-2 px-3">
                            {row.built_up_pct}%
                            {builtDelta != null && (
                              <span className={`ml-2 text-[11px] ${builtDelta >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                                ({builtDelta >= 0 ? "+" : ""}{builtDelta.toFixed(2)}%)
                              </span>
                            )}
                          </td>
                          <td className="py-2 px-3">
                            {row.vegetation_pct}%
                            {vegDelta != null && (
                              <span className={`ml-2 text-[11px] ${vegDelta >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                                ({vegDelta >= 0 ? "+" : ""}{vegDelta.toFixed(2)}%)
                              </span>
                            )}
                          </td>
                          <td className="py-2 px-3">{row.water_pct}%</td>
                          <td className="py-2 px-3">{row.valid_px_pct}%</td>
                          <td className="py-2 px-3">
                            {row.low_confidence ? (
                              <span className="px-2 py-0.5 text-[10px] rounded bg-rose-500/10 text-rose-400 border border-rose-500/30">
                                Low Confidence
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 text-[10px] rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                                Clean SCL
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <div className="text-[11px] italic text-slate-400 text-center pt-2">
            {params.disclaimer}
          </div>
        </div>
      )}
    </div>
  );
}
