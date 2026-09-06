import React from "react";
import { Cpu, ShieldCheck, Zap, Radio, Activity, Layers } from "lucide-react";

/**
 * Mission Telemetry panel.
 *
 * MODEL field:
 *   - STANDBY  → "Awaiting Agentic Controller"
 *   - COMPLETE → analysisResult.model if provided by the backend,
 *                otherwise "Not reported by controller"
 *
 * CONFIDENCE field:
 *   - STANDBY/ANALYZING → "--"
 *   - COMPLETE          → analysisResult.confidence if provided,
 *                         otherwise "Not reported"
 *
 * Neither field ever shows a hardcoded fake value.
 */
function ExecutionSummary({ analysisResult, isAnalyzing, executionTime, images = [] }) {
  const isIdle = !analysisResult && !isAnalyzing;
  const isComplete = !!analysisResult && !isAnalyzing;

  const imageCount = images.length;
  const primaryImageName = images[0]?.name || null;

  // Model: resolved by the agentic controller; null until analysis completes
  const resolvedModel = analysisResult?.model || null;

  // Confidence: only shown after analysis; only if backend provides it
  const resolvedConfidence = analysisResult?.confidence || null;

  // Input label for telemetry
  const inputLabel =
    imageCount > 1
      ? `${imageCount} Satellite Assets`
      : primaryImageName || "Awaiting Input";

  const inputSublabel =
    imageCount > 1
      ? `Multi-temporal · ${images[0]?.sensor || "Optical / SAR"}`
      : images[0]?.sensor || "Multispectral / SAR Ready";

  return (
    <div className="rounded-2xl border border-cyan-400/20 bg-[#070d18]/80 p-6 backdrop-blur-xl shadow-lg">
      {/* TELEMETRY HEADER */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-4">
        <div>
          <p className="text-xs font-semibold tracking-[0.25em] text-cyan-400">
            MISSION TELEMETRY
          </p>
          <h3 className="mt-1 text-xl font-bold text-white flex items-center gap-2">
            <span>PIPELINE STATUS</span>
            <span className="text-slate-500">—</span>
            <span className="text-emerald-400 font-mono text-base">ONLINE</span>
          </h3>
        </div>

        <div className="flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3.5 py-1.5 text-xs font-semibold text-cyan-300">
          <span
            className={`h-2 w-2 rounded-full ${
              isAnalyzing
                ? "bg-cyan-400 animate-ping"
                : isIdle
                ? "bg-slate-500"
                : "bg-emerald-400"
            }`}
          />
          {isAnalyzing
            ? "INFERENCE RUNNING"
            : isIdle
            ? "SYSTEM READY (STANDBY)"
            : "EXECUTION SUCCESSFUL"}
        </div>
      </div>

      {/* TELEMETRY METRIC GRID */}
      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {/* MODEL — set by the agentic controller */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-4">
          <div className="flex items-center justify-between text-slate-400 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider">MODEL</span>
            <Cpu className="h-3.5 w-3.5 text-cyan-400" />
          </div>
          {isIdle || isAnalyzing ? (
            <>
              <p className="text-sm font-semibold text-slate-400 italic">
                Awaiting Agentic Controller
              </p>
              <p className="mt-1 text-[10px] text-slate-500 font-mono">
                {isAnalyzing ? "Controller resolving..." : "No analysis run yet"}
              </p>
            </>
          ) : (
            <>
              <p className="text-base font-bold text-white">
                {resolvedModel || "Not reported by controller"}
              </p>
              <p className="mt-1 text-[10px] text-cyan-300 font-mono">
                {resolvedModel ? "Agentic Controller Selected" : "Model not returned by backend"}
              </p>
            </>
          )}
        </div>

        {/* INPUT */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-4">
          <div className="flex items-center justify-between text-slate-400 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider">INPUT</span>
            <Radio className="h-3.5 w-3.5 text-cyan-400" />
          </div>
          <p
            className="text-sm font-semibold text-slate-200 truncate"
            title={inputLabel}
          >
            {inputLabel}
          </p>
          <p className="mt-1 text-[10px] text-slate-400 font-mono truncate">
            {inputSublabel}
          </p>
        </div>

        {/* PROCESSING */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-4">
          <div className="flex items-center justify-between text-slate-400 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider">PROCESSING</span>
            <Activity className="h-3.5 w-3.5 text-violet-400" />
          </div>
          <p className="text-xs font-semibold text-slate-200 leading-tight">
            {imageCount > 1
              ? "Vision → Change Detection → VLM Reasoning"
              : "Vision → Spectral Analysis → VLM Reasoning"}
          </p>
          <p className="mt-1 text-[10px] text-violet-300 font-mono">
            {isAnalyzing
              ? "Executing Pipeline..."
              : isIdle
              ? "Pipeline Standby"
              : "Pipeline Complete"}
          </p>
        </div>

        {/* LATENCY */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-4">
          <div className="flex items-center justify-between text-slate-400 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider">LATENCY</span>
            <Zap className="h-3.5 w-3.5 text-cyan-400" />
          </div>
          <p className="text-xl font-extrabold text-cyan-400 font-mono">
            {isComplete && executionTime ? executionTime : "--"}
          </p>
          <p className="mt-1 text-[10px] text-slate-400 font-mono">
            {isIdle ? "Awaiting inference" : isAnalyzing ? "Measuring..." : "Inference latency"}
          </p>
        </div>

        {/* INTELLIGENCE & CONFIDENCE */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-4 sm:col-span-2 lg:col-span-1">
          <div className="flex items-center justify-between text-slate-400 mb-1">
            <span className="text-[11px] font-mono uppercase tracking-wider">INTELLIGENCE</span>
            <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
          </div>
          <p className="text-sm font-bold text-emerald-400">
            {isIdle ? "READY" : isAnalyzing ? "PROCESSING" : "VERIFIED"}
          </p>
          <p className="mt-1 text-[10px] text-slate-300 font-mono">
            Confidence:{" "}
            <span className="text-emerald-300">
              {isComplete && resolvedConfidence ? resolvedConfidence : "--"}
            </span>
          </p>
        </div>
      </div>

      {/* IMAGE SET SUMMARY (only when images are loaded) */}
      {imageCount > 0 && (
        <div className="mt-4 flex items-center gap-2 rounded-xl border border-white/10 bg-black/20 px-4 py-2.5 text-xs text-slate-400">
          <Layers className="h-3.5 w-3.5 text-cyan-400 shrink-0" />
          <span>
            <span className="font-semibold text-slate-200">{imageCount}</span>{" "}
            satellite asset{imageCount !== 1 ? "s" : ""} staged
            {imageCount > 1 ? " for multi-temporal analysis" : " for scene analysis"}
          </span>
        </div>
      )}
    </div>
  );
}

export default ExecutionSummary;
