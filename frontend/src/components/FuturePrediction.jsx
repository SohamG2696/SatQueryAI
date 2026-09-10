import React, { useState, useEffect } from "react";
import {
  TrendingUp,
  Calendar,
  MapPin,
  AlertCircle,
  CheckCircle2,
  Info,
  Loader2,
  Sparkles,
  Layers,
  Building2,
  Trees,
  Waves,
} from "lucide-react";
import { predictFuture } from "../services/api";

const REGION_OPTIONS = [
  { id: "region_01", name: "region_01 (Bangalore)", city: "Bangalore" },
  { id: "region_02", name: "region_02 (Mumbai)", city: "Mumbai" },
  { id: "region_03", name: "region_03 (Delhi)", city: "Delhi" },
  { id: "region_04", name: "region_04 (Kolkata)", city: "Kolkata" },
  { id: "region_05", name: "region_05 (Chennai)", city: "Chennai" },
  { id: "region_06", name: "region_06 (Hyderabad)", city: "Hyderabad" },
  { id: "region_07", name: "region_07 (Pune)", city: "Pune" },
  { id: "region_08", name: "region_08 (Jaipur)", city: "Jaipur" },
];

// Historical land-cover time-series data (2019-2025 SCL baseline)
const HISTORICAL_DATA = {
  region_01: [
    { year: 2019, built_up_pct: 65.98, vegetation_pct: 34.02, water_pct: 0.0 },
    { year: 2021, built_up_pct: 65.72, vegetation_pct: 34.28, water_pct: 0.0 },
    { year: 2023, built_up_pct: 64.97, vegetation_pct: 35.03, water_pct: 0.0 },
    { year: 2025, built_up_pct: 62.59, vegetation_pct: 37.37, water_pct: 0.0 },
  ],
  region_02: [
    { year: 2019, built_up_pct: 99.38, vegetation_pct: 0.43, water_pct: 0.17 },
    { year: 2021, built_up_pct: 99.69, vegetation_pct: 0.14, water_pct: 0.18 },
    { year: 2023, built_up_pct: 99.01, vegetation_pct: 0.31, water_pct: 0.68 },
    { year: 2025, built_up_pct: 98.61, vegetation_pct: 0.58, water_pct: 0.77 },
  ],
  region_03: [
    { year: 2019, built_up_pct: 85.28, vegetation_pct: 14.72, water_pct: 0.0 },
    { year: 2021, built_up_pct: 90.00, vegetation_pct: 10.00, water_pct: 0.0 },
    { year: 2023, built_up_pct: 74.91, vegetation_pct: 25.09, water_pct: 0.0 },
    { year: 2025, built_up_pct: 90.79, vegetation_pct: 9.21, water_pct: 0.0 },
  ],
  region_04: [
    { year: 2019, built_up_pct: 99.94, vegetation_pct: 0.04, water_pct: 0.0 },
    { year: 2021, built_up_pct: 100.00, vegetation_pct: 0.00, water_pct: 0.0 },
    { year: 2023, built_up_pct: 99.97, vegetation_pct: 0.00, water_pct: 0.03 },
    { year: 2025, built_up_pct: 99.79, vegetation_pct: 0.16, water_pct: 0.05 },
  ],
  region_05: [
    { year: 2019, built_up_pct: 95.31, vegetation_pct: 4.30, water_pct: 0.0 },
    { year: 2021, built_up_pct: 90.84, vegetation_pct: 8.89, water_pct: 0.0 },
    { year: 2023, built_up_pct: 87.81, vegetation_pct: 12.19, water_pct: 0.0 },
    { year: 2025, built_up_pct: 86.97, vegetation_pct: 13.03, water_pct: 0.0 },
  ],
  region_06: [
    { year: 2019, built_up_pct: 97.12, vegetation_pct: 2.88, water_pct: 0.0 },
    { year: 2021, built_up_pct: 98.80, vegetation_pct: 1.20, water_pct: 0.0 },
    { year: 2023, built_up_pct: 98.30, vegetation_pct: 1.70, water_pct: 0.0 },
    { year: 2025, built_up_pct: 98.32, vegetation_pct: 1.68, water_pct: 0.0 },
  ],
  region_07: [
    { year: 2019, built_up_pct: 96.12, vegetation_pct: 3.21, water_pct: 0.0 },
    { year: 2021, built_up_pct: 94.67, vegetation_pct: 4.56, water_pct: 0.0 },
    { year: 2023, built_up_pct: 96.12, vegetation_pct: 3.85, water_pct: 0.0 },
    { year: 2025, built_up_pct: 94.62, vegetation_pct: 4.46, water_pct: 0.0 },
  ],
  region_08: [
    { year: 2019, built_up_pct: 97.79, vegetation_pct: 2.21, water_pct: 0.0 },
    { year: 2021, built_up_pct: 96.73, vegetation_pct: 3.27, water_pct: 0.0 },
    { year: 2023, built_up_pct: 95.97, vegetation_pct: 4.03, water_pct: 0.0 },
    { year: 2025, built_up_pct: 96.07, vegetation_pct: 3.93, water_pct: 0.0 },
  ],
};

export default function FuturePrediction({ defaultRegion = "region_01", defaultYear = 2027 }) {
  const [selectedRegion, setSelectedRegion] = useState(defaultRegion);
  const [targetYear, setTargetYear] = useState(defaultYear);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Auto-run forecast on mount or when defaultRegion changes
  useEffect(() => {
    handleForecast(defaultRegion, defaultYear);
  }, []);

  const handleForecast = async (regionId = selectedRegion, yr = targetYear) => {
    if (!regionId) return;
    if (!yr || yr <= 2025 || yr > 2075) {
      setError("Please select a target year between 2026 and 2075.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await predictFuture({ regionId, targetYear: Number(yr) });
      setPrediction(res);
    } catch (err) {
      setError(err.message || "Failed to generate land-cover forecast.");
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceBadge = (confidence) => {
    switch (confidence?.toLowerCase()) {
      case "high":
        return {
          bg: "bg-emerald-500/10 border-emerald-500/30 text-emerald-400",
          icon: <CheckCircle2 className="w-4 h-4 text-emerald-400" />,
          label: "HIGH CONFIDENCE",
        };
      case "moderate":
        return {
          bg: "bg-amber-500/10 border-amber-500/30 text-amber-400",
          icon: <Info className="w-4 h-4 text-amber-400" />,
          label: "MODERATE CONFIDENCE",
        };
      case "low":
        return {
          bg: "bg-rose-500/10 border-rose-500/30 text-rose-400",
          icon: <AlertCircle className="w-4 h-4 text-rose-400" />,
          label: "LOW CONFIDENCE",
        };
      default:
        return {
          bg: "bg-cyan-500/10 border-cyan-500/30 text-cyan-400",
          icon: <Info className="w-4 h-4 text-cyan-400" />,
          label: "STATISTICAL PROJECTION",
        };
    }
  };

  // SVG Chart rendering helper
  const renderChart = () => {
    const historicals = HISTORICAL_DATA[selectedRegion] || [];
    const points = [...historicals];

    if (prediction && prediction.forecast_year) {
      points.push({
        year: prediction.forecast_year,
        built_up_pct: prediction.built_up_pct,
        vegetation_pct: prediction.vegetation_pct,
        water_pct: prediction.water_pct,
        isForecast: true,
      });
    }

    const padding = { top: 30, right: 40, bottom: 40, left: 50 };
    const width = 640;
    const height = 260;
    const innerW = width - padding.left - padding.right;
    const innerH = height - padding.top - padding.bottom;

    const minYear = 2019;
    const maxYear = Math.max(2025, prediction?.forecast_year || 2027);

    const getX = (yr) => padding.left + ((yr - minYear) / (maxYear - minYear)) * innerW;
    const getY = (val) => padding.top + innerH - (val / 100) * innerH;

    // Separate historical vs forecast line segments
    const histPts = points.filter((p) => !p.isForecast);
    const forePt = points.find((p) => p.isForecast);
    const lastHistPt = histPts[histPts.length - 1];

    const buildPath = (seriesKey, pts) =>
      pts.map((p, i) => `${i === 0 ? "M" : "L"} ${getX(p.year)} ${getY(p[seriesKey])}`).join(" ");

    const buildForecastPath = (seriesKey) => {
      if (!lastHistPt || !forePt) return "";
      return `M ${getX(lastHistPt.year)} ${getY(lastHistPt[seriesKey])} L ${getX(forePt.year)} ${getY(forePt[seriesKey])}`;
    };

    return (
      <div className="w-full overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto text-xs select-none">
          {/* Background Grid */}
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

          {/* Built-up Line (Cyan) */}
          <path d={buildPath("built_up_pct", histPts)} fill="none" stroke="#06b6d4" strokeWidth="2.5" />
          {forePt && (
            <path
              d={buildForecastPath("built_up_pct")}
              fill="none"
              stroke="#06b6d4"
              strokeWidth="2.5"
              strokeDasharray="6 4"
            />
          )}

          {/* Vegetation Line (Emerald) */}
          <path d={buildPath("vegetation_pct", histPts)} fill="none" stroke="#10b981" strokeWidth="2.5" />
          {forePt && (
            <path
              d={buildForecastPath("vegetation_pct")}
              fill="none"
              stroke="#10b981"
              strokeWidth="2.5"
              strokeDasharray="6 4"
            />
          )}

          {/* Water Line (Blue) */}
          <path d={buildPath("water_pct", histPts)} fill="none" stroke="#3b82f6" strokeWidth="2.5" />
          {forePt && (
            <path
              d={buildForecastPath("water_pct")}
              fill="none"
              stroke="#3b82f6"
              strokeWidth="2.5"
              strokeDasharray="6 4"
            />
          )}

          {/* Data Nodes */}
          {points.map((p) => (
            <g key={p.year}>
              {/* Built-up Node */}
              <circle
                cx={getX(p.year)}
                cy={getY(p.built_up_pct)}
                r={p.isForecast ? 6 : 4}
                fill={p.isForecast ? "#06b6d4" : "#083344"}
                stroke="#06b6d4"
                strokeWidth={p.isForecast ? 3 : 2}
              />
              {/* Vegetation Node */}
              <circle
                cx={getX(p.year)}
                cy={getY(p.vegetation_pct)}
                r={p.isForecast ? 6 : 4}
                fill={p.isForecast ? "#10b981" : "#064e3b"}
                stroke="#10b981"
                strokeWidth={p.isForecast ? 3 : 2}
              />
            </g>
          ))}
        </svg>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-2 text-xs text-slate-400">
          <div className="flex items-center gap-2">
            <span className="w-3 h-0.5 bg-cyan-500 inline-block"></span>
            <span>Built-up %</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-0.5 bg-emerald-500 inline-block"></span>
            <span>Vegetation %</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-0.5 bg-blue-500 inline-block"></span>
            <span>Water %</span>
          </div>
          <div className="flex items-center gap-2 border-l border-slate-700 pl-4">
            <span className="w-4 h-0 border-t-2 border-dashed border-cyan-400 inline-block"></span>
            <span className="text-cyan-400 font-medium">Trend Forecast</span>
          </div>
        </div>
      </div>
    );
  };

  const confidenceBadge = prediction ? getConfidenceBadge(prediction.confidence) : null;

  return (
    <div className="w-full bg-slate-950/80 backdrop-blur-xl border border-slate-800/80 rounded-2xl p-6 shadow-2xl space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-cyan-400 shadow-inner">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white tracking-wide flex items-center gap-2">
              Multi-Year Land-Cover Forecaster
              <span className="text-xs font-mono font-normal px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                Linear Trend
              </span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Statistical baseline projection trained on Sentinel-2 SCL time series (2019–2025).
            </p>
          </div>
        </div>

        {/* Controls: Region Selector & Target Year Input */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Region Dropdown */}
          <div className="flex items-center gap-1.5 bg-slate-900/90 border border-slate-800 rounded-xl px-3 py-1.5">
            <MapPin className="w-4 h-4 text-cyan-400 shrink-0" />
            <select
              value={selectedRegion}
              onChange={(e) => {
                setSelectedRegion(e.target.value);
                handleForecast(e.target.value, targetYear);
              }}
              className="bg-transparent text-xs text-slate-200 focus:outline-none cursor-pointer pr-2 font-medium"
            >
              {REGION_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id} className="bg-slate-900 text-slate-200">
                  {opt.name}
                </option>
              ))}
            </select>
          </div>

          {/* Target Year Input */}
          <div className="flex items-center gap-1.5 bg-slate-900/90 border border-slate-800 rounded-xl px-3 py-1.5">
            <Calendar className="w-4 h-4 text-cyan-400 shrink-0" />
            <span className="text-xs text-slate-400">Year:</span>
            <input
              type="number"
              min={2026}
              max={2075}
              value={targetYear}
              onChange={(e) => setTargetYear(Number(e.target.value))}
              className="w-16 bg-transparent text-xs font-mono text-cyan-300 focus:outline-none font-bold"
            />
          </div>

          {/* Forecast Button */}
          <button
            onClick={() => handleForecast(selectedRegion, targetYear)}
            disabled={loading}
            className="flex items-center gap-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-semibold text-xs px-4 py-2 rounded-xl transition-all shadow-lg shadow-cyan-500/20 active:scale-95 disabled:opacity-50 cursor-pointer"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            Forecast
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-xs">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Line Chart Section */}
      <div className="bg-slate-900/50 border border-slate-800/60 rounded-xl p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2 text-xs font-medium text-slate-300">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span>Land-Cover Trajectory (2019 → {prediction?.forecast_year || targetYear})</span>
          </div>
          {confidenceBadge && (
            <div
              className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-semibold border ${confidenceBadge.bg}`}
            >
              {confidenceBadge.icon}
              <span>{confidenceBadge.label}</span>
            </div>
          )}
        </div>

        {renderChart()}
      </div>

      {/* Results Section */}
      {prediction && (
        <div className="space-y-4">
          {/* Stat Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {/* Target Year Card */}
            <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-4 relative overflow-hidden">
              <div className="text-xs font-medium text-slate-400 flex items-center justify-between">
                <span>Forecast Year</span>
                <Calendar className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="text-2xl font-bold text-cyan-300 font-mono mt-2">
                {prediction.forecast_year}
              </div>
              <div className="text-[11px] text-slate-500 mt-1 font-mono">
                +{prediction.forecast_year - 2025} yrs horizon
              </div>
            </div>

            {/* Built-up % Card */}
            <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-4 relative overflow-hidden">
              <div className="text-xs font-medium text-slate-400 flex items-center justify-between">
                <span>Built-up %</span>
                <Building2 className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="text-2xl font-bold text-white font-mono mt-2">
                {prediction.built_up_pct}%
              </div>
              <div className="text-[11px] text-slate-400 mt-1 font-mono flex items-center gap-1">
                <span className={prediction.annual_slopes?.built_up_pct >= 0 ? "text-emerald-400" : "text-rose-400"}>
                  {prediction.annual_slopes?.built_up_pct >= 0 ? "+" : ""}
                  {prediction.annual_slopes?.built_up_pct}/yr
                </span>
              </div>
            </div>

            {/* Vegetation % Card */}
            <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-4 relative overflow-hidden">
              <div className="text-xs font-medium text-slate-400 flex items-center justify-between">
                <span>Vegetation %</span>
                <Trees className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="text-2xl font-bold text-emerald-400 font-mono mt-2">
                {prediction.vegetation_pct}%
              </div>
              <div className="text-[11px] text-slate-400 mt-1 font-mono flex items-center gap-1">
                <span className={prediction.annual_slopes?.vegetation_pct >= 0 ? "text-emerald-400" : "text-rose-400"}>
                  {prediction.annual_slopes?.vegetation_pct >= 0 ? "+" : ""}
                  {prediction.annual_slopes?.vegetation_pct}/yr
                </span>
              </div>
            </div>

            {/* Water % Card */}
            <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-4 relative overflow-hidden">
              <div className="text-xs font-medium text-slate-400 flex items-center justify-between">
                <span>Water %</span>
                <Waves className="w-4 h-4 text-blue-400" />
              </div>
              <div className="text-2xl font-bold text-blue-400 font-mono mt-2">
                {prediction.water_pct}%
              </div>
              <div className="text-[11px] text-slate-500 mt-1 font-mono">
                Stable / Low
              </div>
            </div>
          </div>

          {/* Interpretation Notes */}
          {prediction.interpretation_notes && prediction.interpretation_notes.length > 0 && (
            <div className="p-3.5 bg-cyan-950/30 border border-cyan-500/20 rounded-xl space-y-1 text-xs">
              <div className="flex items-center gap-2 font-medium text-cyan-300">
                <Info className="w-4 h-4 shrink-0 text-cyan-400" />
                <span>Interpretation & Model Fit Notes</span>
              </div>
              <ul className="list-disc list-inside space-y-1 text-slate-300 pl-1 text-[11px]">
                {prediction.interpretation_notes.map((note, idx) => (
                  <li key={idx}>{note}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Disclaimer */}
          <div className="text-[11px] italic text-slate-400 text-center pt-1">
            {prediction.disclaimer}
          </div>
        </div>
      )}
    </div>
  );
}
