import React, { useState } from "react";
import {
  Sparkles,
  Layers,
  AlertTriangle,
  CheckCircle2,
  Copy,
  Check,
  Download,
  TrendingUp,
  Activity,
  FileCheck,
  Eye,
  MapPin,
  Crosshair,
} from "lucide-react";

function ResultPanel({ analysisResult, isAnalyzing, currentStep, error, images = [] }) {
  const [copied, setCopied] = useState(false);

  // ERROR STATE
  if (error) {
    return (
      <div className="rounded-2xl border border-red-500/40 bg-[#070d18]/90 p-8 md:p-10 backdrop-blur-xl shadow-[0_0_30px_rgba(239,68,68,0.15)]">
        <div className="flex items-center gap-3 text-red-400 mb-4">
          <AlertTriangle className="h-6 w-6" />
          <h3 className="text-lg font-bold text-white">Analysis Execution Error</h3>
        </div>
        <p className="text-sm text-slate-300 leading-relaxed bg-red-950/30 border border-red-500/20 rounded-xl p-4 font-mono">
          {error}
        </p>
        <p className="mt-4 text-xs text-slate-400">
          Please check that your backend server is running on <code className="text-cyan-300">http://127.0.0.1:8000</code>.
        </p>
      </div>
    );
  }

  const handleCopy = () => {
    if (!analysisResult) return;
    const text = `SATELLITE INTELLIGENCE REPORT
Target Scene: ${analysisResult.sceneName || "Uploaded Scene"}
Detected Features: ${analysisResult.detectedObjects || "Not provided"} (${analysisResult.objectType || "Not provided"})
Change Detected: ${analysisResult.changesDetected || "Not provided"}
AI Confidence: ${analysisResult.confidence || "Not available"}

ANALYSIS SUMMARY:
${analysisResult.message || "Not provided"}

KEY FINDINGS:
${analysisResult.keyFindings && analysisResult.keyFindings.length > 0 ? analysisResult.keyFindings.map((f) => `- ${f}`).join("\n") : "Not provided"}

RECOMMENDATION:
${analysisResult.recommendation || "Not provided"}`;

    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadReport = () => {
    if (!analysisResult) return;

    const reportContent = `===================================================================
                       SATQUERY AI INTELLIGENCE REPORT
===================================================================
Generated At : ${new Date().toISOString()}
Target Asset : ${analysisResult.sceneName || "Uploaded Satellite Imagery"}
User Query   : ${analysisResult.query || "Geospatial Feature Analysis"}

-------------------------------------------------------------------
1. PRIMARY METRICS
-------------------------------------------------------------------
Detected Features : ${analysisResult.detectedObjects || "Not provided"} (${analysisResult.objectType || "Not provided"})
Change Detected   : ${analysisResult.changesDetected || "Not provided"}
AI Confidence     : ${analysisResult.confidence || "Not available"}

-------------------------------------------------------------------
2. AI SYNTHESIS & SCENE INTERPRETATION
-------------------------------------------------------------------
${analysisResult.message || "Not provided."}

-------------------------------------------------------------------
3. OBSERVATIONAL FINDINGS
-------------------------------------------------------------------
${
  analysisResult.keyFindings && analysisResult.keyFindings.length > 0
    ? analysisResult.keyFindings.map((f, i) => `[${i + 1}] ${f}`).join("\n")
    : "Not provided."
}

-------------------------------------------------------------------
4. RECOMMENDED ACTIONS
-------------------------------------------------------------------
${analysisResult.recommendation || "Not provided."}

-------------------------------------------------------------------
5. SPECTRAL & SENSOR METRICS
-------------------------------------------------------------------
${
  analysisResult.spectralData
    ? Object.entries(analysisResult.spectralData)
        .map(([k, v]) => `${k}: ${v}`)
        .join("\n")
    : "Not available."
}

-------------------------------------------------------------------
6. MISSION TELEMETRY
-------------------------------------------------------------------
Pipeline Status : ONLINE
Model Engine    : ${analysisResult.model || "Not reported by controller"}
Input Platform  : ${analysisResult.spectralData?.sensorPlatform || "Optical / SAR Imagery"}
Processing      : Vision -> Spectral Analysis -> VLM Reasoning
Intelligence    : VERIFIED
Confidence      : ${analysisResult.confidence || "Not available"}
===================================================================
`;

    const blob = new Blob([reportContent], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const sanitizedName = (analysisResult.sceneName || "Satellite_Scene").replace(/[^a-z0-9]/gi, "_");
    link.download = `SatQuery_Intelligence_Report_${sanitizedName}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // LOADING / SCANNING STATE
  if (isAnalyzing) {
    const steps = [
      "Decompressing raster imagery & radiometric calibration...",
      "Extracting multi-spectral bands & computing index signatures...",
      "Running convolutional neural detection & change differential...",
      "Synthesizing geospatial intelligence and recommendations...",
    ];

    return (
      <div className="rounded-2xl border border-cyan-400/30 bg-[#070d18]/90 p-8 md:p-10 backdrop-blur-xl">
        <div className="flex flex-col items-center justify-center text-center">
          {/* RADAR SWEEP ANIMATION */}
          <div className="relative mb-6 flex h-24 w-24 items-center justify-center">
            <div className="absolute inset-0 rounded-full border border-cyan-400/20" />
            <div className="absolute inset-2 rounded-full border border-cyan-400/30" />
            <div className="absolute inset-5 rounded-full border border-cyan-400/40" />
            <div
              className="absolute inset-0 rounded-full border-t-2 border-cyan-400 shadow-[0_0_20px_#22d3ee]"
              style={{
                animation: "spin 1.5s linear infinite",
              }}
            />
            <Sparkles className="h-8 w-8 text-cyan-300 animate-pulse" />
          </div>

          <p className="text-xs font-semibold tracking-[0.3em] text-cyan-400 uppercase">
            SATQUERY AI INFERENCE ENGINE
          </p>
          <h2 className="mt-2 text-2xl font-bold text-white">
            Analyzing Geospatial Scene
          </h2>
          <p className="mt-2 max-w-md text-sm text-slate-400">
            Applying vision-language neural models to extract spatial features and anomalies in the uploaded imagery.
          </p>

          {/* STEP PROGRESS */}
          <div className="mt-6 w-full max-w-md space-y-2 text-left">
            {steps.map((step, idx) => {
              const isCompleted = currentStep > idx;
              const isCurrent = currentStep === idx;
              return (
                <div
                  key={idx}
                  className={`flex items-center gap-2.5 rounded-lg px-3 py-1.5 text-xs transition-colors duration-300 ${
                    isCompleted
                      ? "bg-emerald-500/10 text-emerald-300 border border-emerald-500/20"
                      : isCurrent
                      ? "bg-cyan-500/15 text-cyan-200 border border-cyan-400/40"
                      : "text-slate-600"
                  }`}
                >
                  <div
                    className={`h-1.5 w-1.5 rounded-full ${
                      isCompleted
                        ? "bg-emerald-400"
                        : isCurrent
                        ? "bg-cyan-400 animate-ping"
                        : "bg-slate-700"
                    }`}
                  />
                  <span>{step}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  // INITIAL CLEAN EMPTY / STANDBY STATE
  if (!analysisResult) {
    return (
      <div className="rounded-2xl border border-white/10 bg-[#070d18]/60 p-8 md:p-10 backdrop-blur-xl">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs font-semibold tracking-[0.25em] text-cyan-400">
              INTELLIGENCE OUTPUT
            </p>
            <h2 className="mt-1 text-2xl font-bold text-white">
              Awaiting Target Analysis
            </h2>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-slate-700 bg-slate-800/60 px-3.5 py-1.5 text-xs text-slate-400">
            <span className="h-2 w-2 rounded-full bg-slate-500" />
            STANDBY MODE
          </div>
        </div>

        <div className="mt-8 flex min-h-[200px] flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 bg-black/20 p-8 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-slate-400 mb-3">
            <Layers className="h-7 w-7" />
          </div>
          <p className="text-base font-medium text-slate-300">
            INTELLIGENCE OUTPUT
          </p>
          <p className="mt-1 max-w-md text-xs text-slate-400 leading-relaxed">
            Upload satellite imagery and run an analysis to generate evidence-grounded insights.
          </p>
        </div>
      </div>
    );
  }

  // COMPLETED ANALYSIS STATE
  return (
    <div className="rounded-2xl border border-cyan-400/30 bg-[#070d18]/90 p-6 md:p-8 backdrop-blur-xl shadow-[0_0_50px_rgba(34,211,238,0.08)]">
      {/* HEADER */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-6">
        <div>
          <div className="flex items-center gap-2">
            <p className="text-xs font-semibold tracking-[0.25em] text-cyan-400">
              INTELLIGENCE DOSSIER
            </p>
            <span className="text-slate-600">•</span>
            <span className="text-xs text-slate-400 font-mono">
              {analysisResult.sceneName || "Uploaded Target Scene"}
            </span>
          </div>
          <h2 className="mt-1.5 text-2xl md:text-3xl font-bold text-white">
            Geospatial Analysis Report
          </h2>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={handleCopy}
            className="flex items-center gap-1.5 rounded-xl border border-white/15 bg-white/5 px-3.5 py-2 text-xs font-medium text-slate-200 transition hover:bg-white/10 hover:text-white"
          >
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5 text-emerald-400" />
                <span>Copied</span>
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5 text-slate-400" />
                <span>Copy Summary</span>
              </>
            )}
          </button>

          <button
            type="button"
            onClick={handleDownloadReport}
            className="flex items-center gap-1.5 rounded-xl border border-cyan-400/40 bg-cyan-500/10 px-3.5 py-2 text-xs font-semibold text-cyan-300 transition hover:bg-cyan-500/20 hover:border-cyan-400"
          >
            <Download className="h-3.5 w-3.5 text-cyan-400" />
            <span>DOWNLOAD REPORT</span>
          </button>

          <div className="flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-xs font-semibold text-emerald-400 shadow-[0_0_15px_rgba(52,211,153,0.15)]">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            ANALYSIS COMPLETE
          </div>
        </div>
      </div>

      {/* THREE PRIMARY METRIC CARDS */}
      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        {/* CARD 1: DETECTED OBJECTS */}
        <div className="relative overflow-hidden rounded-2xl border border-cyan-400/20 bg-cyan-950/20 p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium tracking-wider text-cyan-300">
              DETECTED OBJECTS
            </span>
            <Activity className="h-4 w-4 text-cyan-400" />
          </div>

          <h3 className="mt-3 text-2xl md:text-3xl font-extrabold text-cyan-300">
            {analysisResult.detectedObjects || "Not provided"}
          </h3>

          <p className="mt-2 text-xs text-slate-400">
            {analysisResult.objectType || "Not provided"}
          </p>
        </div>

        {/* CARD 2: CHANGE DETECTION */}
        <div className="relative overflow-hidden rounded-2xl border border-violet-400/20 bg-violet-950/20 p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium tracking-wider text-violet-300">
              CHANGE DETECTION
            </span>
            <TrendingUp className="h-4 w-4 text-violet-400" />
          </div>

          <h3 className="mt-3 text-xl md:text-2xl font-bold text-violet-200">
            {analysisResult.changesDetected || "Not provided"}
          </h3>

          <p className="mt-2 text-xs text-slate-400">
            Displacement vs baseline
          </p>
        </div>

        {/* CARD 3: AI CONFIDENCE */}
        <div className="relative overflow-hidden rounded-2xl border border-emerald-400/20 bg-emerald-950/20 p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium tracking-wider text-emerald-300">
              CONFIDENCE SCORE
            </span>
            <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          </div>

          <h3 className="mt-3 text-2xl md:text-3xl font-extrabold text-emerald-300">
            {analysisResult.confidence || "Not available"}
          </h3>

          <div className="mt-2 flex items-center gap-2">
            <div className="h-1.5 flex-1 rounded-full bg-emerald-950">
              <div
                className="h-full rounded-full bg-emerald-400"
                style={{
                  width: analysisResult.confidence?.includes("%")
                    ? analysisResult.confidence
                    : "100%",
                }}
              />
            </div>
            <span className="text-[10px] text-emerald-400 font-mono">
              VALIDATED
            </span>
          </div>
        </div>
      </div>

      {/* AI INSIGHT / NARRATIVE */}
      <div className="mt-6 rounded-2xl border border-cyan-400/20 bg-gradient-to-br from-cyan-950/20 via-black/40 to-black/60 p-6">
        <div className="flex items-center gap-2 text-xs font-semibold tracking-wider text-cyan-400">
          <Sparkles className="h-4 w-4" />
          AI SYNTHESIS & SCENE INTERPRETATION
        </div>

        <p className="mt-3 text-sm leading-relaxed text-slate-200 font-sans">
          {analysisResult.message || "Not provided"}
        </p>
      </div>

      {/* VISUAL GROUNDING EVIDENCE */}
      {(() => {
        const ve = analysisResult.visual_evidence;
        if (!ve) return null;
        const maskB64 = ve.data?.change_mask_base64 || ve.data?.mask_base64 || ve.change_mask_base64 || ve.mask_base64;
        const coords = ve.coordinates || ve.data?.coordinates;
        const veType = ve.type || (maskB64 ? "change_mask" : coords ? "bbox" : "none");

        if (!maskB64 && (!coords || coords.length === 0) && veType === "none") return null;

        return (
          <div className="mt-6 rounded-2xl border border-cyan-400/30 bg-black/40 p-6 shadow-[0_0_30px_rgba(34,211,238,0.06)]">
            <div className="flex items-center gap-2 text-xs font-semibold tracking-wider text-cyan-400 mb-4">
              <Eye className="h-4 w-4" />
              VISUAL GROUNDING EVIDENCE ({veType.toUpperCase()})
            </div>

            {maskB64 && (() => {
              const isGrounding = analysisResult.task_detected === "grounding" || ve.data?.classification_method || ve.classification_method;
              const method = ve.data?.classification_method || ve.classification_method;
              const targetClass = ve.data?.target_class || ve.target_class || analysisResult.rawResponse?.target || "Target Region";
              const coveragePct = ve.data?.target_coverage_pct ?? ve.target_coverage_pct;
              const coveredPx = ve.data?.target_pixels ?? ve.data?.changed_pixels ?? ve.target_pixels ?? ve.changed_pixels;
              const totalPx = ve.data?.total_pixels ?? ve.total_pixels;
              const isFallback = Boolean(analysisResult.rawResponse?.execution_summary?.parameters?.semantic_fallback);
              const warningMsg = analysisResult.rawResponse?.execution_summary?.parameters?.warning;

              return (
                <div className="flex flex-col items-center gap-4">
                  {/* Title & Classification Pill */}
                  <div className="flex flex-wrap items-center justify-between w-full gap-2 text-xs">
                    <span className="text-xs text-slate-300 font-medium tracking-wide">
                      {isGrounding ? `Grounded Region Highlight Overlay (${targetClass})` : "Change Mask Overlay"}
                    </span>
                    {method && (
                      <div className="flex items-center gap-2">
                        {method === "spectral_scl" ? (
                          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 text-[11px] font-medium shadow-[0_0_15px_rgba(16,185,129,0.2)]">
                            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                            Sentinel-2 SCL Spectral Classification (Validated)
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-950/80 border border-amber-500/40 text-amber-300 text-[11px] font-medium shadow-[0_0_15px_rgba(245,158,11,0.2)]">
                            <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
                            RGB Texture &amp; Spectral Heuristic (Lower Precision)
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Highlight Overlay Image Viewport */}
                  <div className="relative w-full max-w-2xl mx-auto rounded-2xl overflow-hidden border border-cyan-400/40 bg-black/90 shadow-[0_0_30px_rgba(34,211,238,0.15)] group">
                    <img
                      src={`data:image/png;base64,${maskB64}`}
                      alt={isGrounding ? "Region Grounding Highlight Overlay" : "Change Detection Mask"}
                      className="w-full h-auto block object-contain max-h-[520px] select-none"
                    />
                    {/* Subtle Satellite HUD Scanlines */}
                    <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_bottom,transparent_50%,rgba(0,0,0,0.22)_51%)] bg-[length:100%_4px] opacity-25" />
                  </div>

                  {/* Coverage Metrics */}
                  <div className="flex flex-wrap items-center justify-between w-full text-xs text-slate-400 font-mono px-1">
                    {coveragePct != null && (
                      <div>
                        Target Scene Coverage: <span className="text-cyan-300 font-bold">{coveragePct}%</span>
                      </div>
                    )}
                    {coveredPx != null && totalPx != null && (
                      <div>
                        {isGrounding ? "Target Pixels: " : "Changed Pixels: "}
                        <span className="text-cyan-300 font-semibold">{coveredPx.toLocaleString()}</span> / {totalPx.toLocaleString()} px
                      </div>
                    )}
                  </div>

                  {/* Semantic Fallback Notice (Only if RGB Heuristic Fallback is truly active) */}
                  {isFallback && warningMsg && (
                    <div className="w-full flex items-start gap-2.5 rounded-xl border border-amber-500/30 bg-amber-950/20 px-4 py-2.5 text-xs text-amber-200 mt-1">
                      <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                      <div className="space-y-0.5">
                        <p className="font-semibold text-amber-300">Semantic Fallback Mode Active</p>
                        <p className="text-[11px] text-amber-200/80 leading-relaxed">
                          {warningMsg}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}

            {!maskB64 && coords && coords.length >= 4 && (() => {
              const [x1Raw, y1Raw, x2Raw, y2Raw] = coords;
              const x1 = parseFloat(x1Raw);
              const y1 = parseFloat(y1Raw);
              const x2 = parseFloat(x2Raw);
              const y2 = parseFloat(y2Raw);

              const xMin = Math.max(0, Math.min(1, Math.min(x1, x2)));
              const xMax = Math.max(0, Math.min(1, Math.max(x1, x2)));
              const yMin = Math.max(0, Math.min(1, Math.min(y1, y2)));
              const yMax = Math.max(0, Math.min(1, Math.max(y1, y2)));

              const leftPct = (xMin * 100).toFixed(2);
              const topPct = (yMin * 100).toFixed(2);
              const widthPct = Math.max(2, (xMax - xMin) * 100).toFixed(2);
              const heightPct = Math.max(2, (yMax - yMin) * 100).toFixed(2);

              const displayImageUrl =
                images?.[0]?.url ||
                analysisResult?.imageUrl ||
                analysisResult?.images?.[0]?.url ||
                (typeof images?.[0] === "string" ? images[0] : null);

              const rawParams = analysisResult.rawResponse?.execution_summary?.parameters || {};
              const targetName =
                rawParams.requested_target ||
                analysisResult.rawResponse?.target ||
                rawParams.target ||
                analysisResult.objectType ||
                "Target Region";
              const modelTarget = rawParams.model_target;
              const isFallback = Boolean(rawParams.semantic_fallback);
              const warningMsg = rawParams.warning;
              const confidenceStr = analysisResult.confidence;

              return (
                <div className="space-y-4">
                  {/* SATELLITE IMAGE WITH HIGHLIGHTED BOUNDING BOX OVERLAY */}
                  {displayImageUrl ? (
                    <div className="flex flex-col items-center gap-3">
                      {/* Target Header Bar */}
                      <div className="flex flex-wrap items-center justify-between w-full gap-2 text-xs">
                        <div className="flex items-center gap-2">
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-cyan-950/60 border border-cyan-400/30 text-cyan-300 font-semibold tracking-wide uppercase text-[11px]">
                            <Crosshair className="h-3.5 w-3.5 text-cyan-400" />
                            Target: {targetName}
                          </span>
                          {isFallback && modelTarget && (
                            <span className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-amber-950/40 border border-amber-500/30 text-amber-300 text-[10px] font-medium">
                              Class: {modelTarget} (Fallback)
                            </span>
                          )}
                        </div>
                        {confidenceStr && confidenceStr !== "Not available" && (
                          <span className="font-mono text-[11px] text-slate-400">
                            Confidence: <span className="text-cyan-300 font-semibold">{confidenceStr}</span>
                          </span>
                        )}
                      </div>

                      {/* Interactive Visual Overlay Viewport */}
                      <div className="relative w-full max-w-2xl mx-auto rounded-2xl overflow-hidden border border-cyan-400/40 bg-black/90 shadow-[0_0_30px_rgba(34,211,238,0.15)] group">
                        {/* Base Uploaded Satellite Imagery */}
                        <img
                          src={displayImageUrl}
                          alt={`Grounded Satellite Scene: ${targetName}`}
                          className="w-full h-auto block object-contain max-h-[520px] select-none"
                        />

                        {/* Responsive Highlighted Bounding Box Overlay */}
                        <div
                          className="absolute pointer-events-none transition-all duration-300"
                          style={{
                            left: `${leftPct}%`,
                            top: `${topPct}%`,
                            width: `${widthPct}%`,
                            height: `${heightPct}%`,
                          }}
                        >
                          {/* Glowing Neon Box */}
                          <div className="w-full h-full border-2 border-cyan-400 bg-cyan-400/20 shadow-[0_0_20px_rgba(34,211,238,0.6),inset_0_0_15px_rgba(34,211,238,0.2)] rounded-sm relative">
                            {/* High-Precision GIS Corner Brackets */}
                            <div className="absolute -top-1 -left-1 w-2.5 h-2.5 border-t-2 border-l-2 border-cyan-300 shadow-[0_0_6px_rgba(34,211,238,0.8)]" />
                            <div className="absolute -top-1 -right-1 w-2.5 h-2.5 border-t-2 border-r-2 border-cyan-300 shadow-[0_0_6px_rgba(34,211,238,0.8)]" />
                            <div className="absolute -bottom-1 -left-1 w-2.5 h-2.5 border-b-2 border-l-2 border-cyan-300 shadow-[0_0_6px_rgba(34,211,238,0.8)]" />
                            <div className="absolute -bottom-1 -right-1 w-2.5 h-2.5 border-b-2 border-r-2 border-cyan-300 shadow-[0_0_6px_rgba(34,211,238,0.8)]" />

                            {/* Floating Target Label */}
                            <div
                              className={`absolute left-0 flex items-center gap-1.5 px-2 py-0.5 rounded shadow-lg tracking-wider uppercase font-bold text-[10px] whitespace-nowrap border ${
                                parseFloat(topPct) > 8
                                  ? "-top-6 bg-[#040914]/95 border-cyan-400/80 text-cyan-300 shadow-[0_0_12px_rgba(34,211,238,0.4)]"
                                  : "top-1 left-1 bg-[#040914]/95 border-cyan-400/80 text-cyan-300 shadow-[0_0_12px_rgba(34,211,238,0.4)]"
                              }`}
                            >
                              <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-pulse" />
                              <span>{targetName}</span>
                              {isFallback && (
                                <span className="text-[9px] text-amber-300 font-normal">({modelTarget})</span>
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Subtle Satellite HUD Scanlines */}
                        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_bottom,transparent_50%,rgba(0,0,0,0.22)_51%)] bg-[length:100%_4px] opacity-25" />
                      </div>

                      {/* Semantic Fallback Notice */}
                      {isFallback && (
                        <div className="w-full flex items-start gap-2.5 rounded-xl border border-amber-500/30 bg-amber-950/20 px-4 py-2.5 text-xs text-amber-200">
                          <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
                          <div className="space-y-0.5">
                            <p className="font-semibold text-amber-300">Semantic Fallback Mode Active</p>
                            <p className="text-[11px] text-amber-200/80 leading-relaxed">
                              {warningMsg || `The requested entity "${targetName}" was grounded to the broader "${modelTarget}" land-cover class supported by the model.`}
                            </p>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : null}

                  {/* NUMERIC COORDINATE BADGES */}
                  <div className="rounded-xl border border-violet-400/30 bg-violet-950/20 p-4">
                    <div className="flex items-center justify-between text-xs text-slate-300 font-medium mb-3">
                      <span className="flex items-center gap-1.5 text-violet-300">
                        <MapPin className="h-4 w-4 text-violet-400" />
                        Grounded Spatial Coordinates
                      </span>
                      <span className="font-mono text-[10px] text-slate-400 uppercase">
                        System: {ve.coordinate_system || "normalized"} · Range: [0.0000 – 1.0000]
                      </span>
                    </div>
                    <div className="grid grid-cols-4 gap-2 text-center font-mono text-xs">
                      <div className="rounded-lg bg-black/50 p-2.5 border border-white/10">
                        <span className="block text-[10px] text-slate-500 mb-1">X MIN</span>
                        <span className="text-cyan-300 font-bold text-sm">{coords[0]}</span>
                      </div>
                      <div className="rounded-lg bg-black/50 p-2.5 border border-white/10">
                        <span className="block text-[10px] text-slate-500 mb-1">Y MIN</span>
                        <span className="text-cyan-300 font-bold text-sm">{coords[1]}</span>
                      </div>
                      <div className="rounded-lg bg-black/50 p-2.5 border border-white/10">
                        <span className="block text-[10px] text-slate-500 mb-1">X MAX</span>
                        <span className="text-cyan-300 font-bold text-sm">{coords[2]}</span>
                      </div>
                      <div className="rounded-lg bg-black/50 p-2.5 border border-white/10">
                        <span className="block text-[10px] text-slate-500 mb-1">Y MAX</span>
                        <span className="text-cyan-300 font-bold text-sm">{coords[3]}</span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()}
          </div>
        );
      })()}

      {/* KEY FINDINGS & RECOMMENDATIONS */}
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        {/* FINDINGS */}
        <div className="rounded-2xl border border-white/10 bg-black/30 p-5">
          <h4 className="flex items-center gap-2 text-xs font-semibold tracking-wider text-slate-300">
            <FileCheck className="h-4 w-4 text-cyan-400" />
            CORE OBSERVATIONAL FINDINGS
          </h4>

          <ul className="mt-3 space-y-2 text-xs text-slate-300">
            {analysisResult.keyFindings && analysisResult.keyFindings.length > 0 ? (
              analysisResult.keyFindings.map((finding, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-400" />
                  <span>{finding}</span>
                </li>
              ))
            ) : (
              <li className="text-slate-400">Not provided</li>
            )}
          </ul>
        </div>

        {/* RECOMMENDATION */}
        <div className="rounded-2xl border border-white/10 bg-black/30 p-5">
          <h4 className="flex items-center gap-2 text-xs font-semibold tracking-wider text-amber-300">
            <AlertTriangle className="h-4 w-4 text-amber-400" />
            RECOMMENDED ACTION
          </h4>

          <p className="mt-3 text-xs leading-relaxed text-slate-300">
            {analysisResult.recommendation || "Not provided"}
          </p>

          {analysisResult.spectralData ? (
            <div className="mt-4 border-t border-white/10 pt-3">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
                Spectral Bands / Sensor Metrics
              </span>
              <div className="mt-1.5 flex flex-wrap gap-2 text-[11px] font-mono text-slate-300">
                {Object.entries(analysisResult.spectralData).map(([key, val]) => (
                  <span
                    key={key}
                    className="rounded bg-white/5 px-2 py-0.5 border border-white/10"
                  >
                    {key}: <span className="text-cyan-300">{val}</span>
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <div className="mt-4 border-t border-white/10 pt-3 text-[11px] text-slate-500">
              Spectral metrics: Not available
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ResultPanel;
