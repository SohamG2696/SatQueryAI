import React, { useState, useRef } from "react";
import ImageUploader from "./ImageUploader";
import QueryBox from "./QueryBox";
import ResultPanel from "./ResultPanel";
import ExecutionSummary from "./ExecutionSummary";
import HistoryPanel from "./HistoryPanel";
import Spotlight from "./Spotlight";
import FuturePrediction from "../FuturePrediction";
import DynamicFuturePrediction from "../DynamicFuturePrediction";
import { History, ShieldCheck, Terminal, AlertCircle } from "lucide-react";
import { executeQuery } from "@/services/api";
import { saveQueryHistory } from "@/services/history";
import "@/styles/analysis.css";

export default function AnalysisWorkspace({ supabase, user, openAuthModal }) {
  const [images, setImages] = useState([]);           // multi-image array
  const [query, setQuery] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [executionTime, setExecutionTime] = useState(null);
  const [error, setError] = useState(null);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [forecasterTab, setForecasterTab] = useState("preset");

  const resultsRef = useRef(null);

  const handleImagesChange = (updatedImages) => {
    setImages(updatedImages);
    setAnalysisResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!images.length || isAnalyzing) return;

    setIsAnalyzing(true);
    setCurrentStep(0);
    setExecutionTime(null);
    setError(null);
    const startTime = performance.now();

    const t1 = setTimeout(() => setCurrentStep(1), 300);
    const t2 = setTimeout(() => setCurrentStep(2), 700);
    const t3 = setTimeout(() => setCurrentStep(3), 1100);

    try {
      const response = await executeQuery({ query, images });
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2) + "s";
      setExecutionTime(elapsed);

      const task = response.task_detected || response.task || "vqa";
      const modelsUsed = response.execution_summary?.models_used || response.execution_summary?.models_invoked || [];
      const taskRoute = response.execution_summary?.task_route || response.execution_summary?.route_selected || "Standard Pipeline";
      const modelName = modelsUsed.length > 0
        ? modelsUsed.join(", ")
        : taskRoute !== "Standard Pipeline" ? taskRoute : "SatQuery Specialist Model";

      const keyFindings = [];
      if (response.synthesis?.key_findings?.length) {
        keyFindings.push(...response.synthesis.key_findings);
      } else if (response.execution_summary?.parameters?.interpretation?.key_findings?.length) {
        keyFindings.push(...response.execution_summary.parameters.interpretation.key_findings);
      } else {
        keyFindings.push(
          `Task Identified: ${task.toUpperCase()}`,
          `Route Invoked: ${taskRoute}`,
          `Processing Latency: ${response.execution_summary?.processing_time_ms ? `${response.execution_summary.processing_time_ms}ms` : (response.execution_summary?.processing_time_seconds || elapsed)}`
        );
      }

      const result = {
        rawResponse: response,
        sceneName: images[0]?.name || images[0]?.file?.name || "Uploaded Satellite Imagery",
        imageCount: images.length,
        query: query.trim(),
        task_detected: task,
        model: modelName,
        detectedObjects: task === "grounding" ? "Localized Region" : task.replace("_", " ").toUpperCase(),
        objectType: taskRoute !== "Standard Pipeline" ? taskRoute : "Geospatial Target",
        changesDetected: task === "change_vqa" ? "Change Detection Applied" : "Single Scene Inspection",
        confidence: response.confidence != null ? `${Math.round(response.confidence * 100)}%` : "Not available",
        message: response.answer || "Analysis complete.",
        keyFindings: keyFindings,
        recommendation: response.verification?.notes || response.execution_summary?.parameters?.interpretation?.summary || "Analysis verified by SatQuery AI pipeline.",
        visual_evidence: response.visual_evidence,
        spectralData: {
          taskDetected: task,
          modelsInvoked: modelsUsed.join(", ") || "Active Module",
          routeSelected: taskRoute,
          imageCount: `${images.length} asset${images.length !== 1 ? "s" : ""}`,
        },
      };

      setAnalysisResult(result);
      setIsAnalyzing(false);

      // Save to Supabase query history + storage
      if (supabase && user?.id) {
        saveQueryHistory({
          supabase,
          user,
          query,
          images,
          response,
          analysisResult: result,
        }).catch((err) => {
          console.warn("Background history save note:", err);
        });
      }

      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (err) {
      console.error("Analysis execution error:", err);
      setIsAnalyzing(false);
      setError(err.message || "Unable to connect to SatQuery AI backend. Please ensure the backend server is running on port 8000.");
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    }
  };

  const handleRestoreHistoryItem = (item, restoredImages) => {
    if (item.query) {
      setQuery(item.query);
    }
    if (restoredImages && restoredImages.length > 0) {
      setImages(restoredImages);
    }

    const savedRawResult = item.execution_summary?.raw_result || null;
    const savedRawResponse = item.execution_summary?.raw_response || null;

    const taskDetected = item.task_detected || savedRawResult?.task_detected || "vqa";
    const modelName = item.model_used || savedRawResult?.model || item.execution_summary?.models_used?.join(", ") || "SatQuery Controller";
    const taskRoute = item.execution_summary?.task_route || item.execution_summary?.route_selected || savedRawResult?.objectType || (taskDetected === "change_vqa" ? "bi_temporal_change_analysis" : "Standard Pipeline");

    const restoredSceneName = restoredImages?.[0]?.name || savedRawResult?.sceneName || "Restored Satellite Scene";
    const restoredImageCount = restoredImages?.length || savedRawResult?.imageCount || item.images?.length || 0;

    let keyFindings = savedRawResult?.keyFindings || [];
    if (!keyFindings || keyFindings.length === 0) {
      if (savedRawResponse?.synthesis?.key_findings?.length) {
        keyFindings = savedRawResponse.synthesis.key_findings;
      } else {
        keyFindings = [
          `Task Identified: ${taskDetected.toUpperCase()}`,
          `Route Invoked: ${taskRoute}`,
          `Model Engine: ${modelName}`,
        ];
      }
    }

    const restoredResult = {
      rawResponse: savedRawResponse || item.execution_summary || null,
      sceneName: restoredSceneName,
      imageCount: restoredImageCount,
      query: item.query || savedRawResult?.query || "",
      task_detected: taskDetected,
      model: modelName,
      detectedObjects: savedRawResult?.detectedObjects || (taskDetected === "grounding" ? "Localized Region" : taskDetected.replace("_", " ").toUpperCase()),
      objectType: taskRoute !== "Standard Pipeline" ? taskRoute : (savedRawResult?.objectType || "Geospatial Target"),
      changesDetected: savedRawResult?.changesDetected || (taskDetected === "change_vqa" ? "Change Detection Applied" : "Single Scene Inspection"),
      confidence: item.confidence || savedRawResult?.confidence || (savedRawResponse?.confidence != null ? `${Math.round(savedRawResponse.confidence * 100)}%` : "Validated"),
      message: item.answer || savedRawResult?.message || "",
      keyFindings: keyFindings,
      recommendation: savedRawResult?.recommendation || savedRawResponse?.verification?.notes || "Analysis restored from persistent user history.",
      visual_evidence: item.visual_evidence || savedRawResult?.visual_evidence || savedRawResponse?.visual_evidence || null,
      spectralData: savedRawResult?.spectralData || {
        taskDetected: taskDetected,
        modelsInvoked: modelName,
        routeSelected: taskRoute,
        imageCount: `${restoredImageCount} asset${restoredImageCount !== 1 ? "s" : ""}`,
      },
    };

    setAnalysisResult(restoredResult);

    setTimeout(() => {
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
  };

  return (
    <div className="analysis-workspace-root font-sans">
      {/* PREMIUM BACKGROUND GLOWS */}
      <div className="fixed inset-0 pointer-events-none bg-[radial-gradient(circle_at_top_right,_rgba(6,182,212,0.12),transparent_40%),radial-gradient(circle_at_bottom_left,_rgba(139,92,246,0.10),transparent_45%)]" />

      {/* SPOTLIGHT COMPONENT */}
      <Spotlight
        className="-top-40 left-0 md:-top-20 md:left-40"
        duration={8}
        xOffset={120}
      />

      <div className="relative z-10">
        {/* ================= HERO SUB-HEADER ================= */}
        <section className="mx-auto max-w-7xl px-6 pb-6 pt-6 text-center md:px-10 md:pt-8">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-4 py-1.5 text-xs font-medium tracking-widest text-cyan-300 shadow-[0_0_20px_rgba(34,211,238,0.15)]">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-ping" />
            INTELLIGENT SATELLITE ANALYSIS WORKSPACE
          </div>

          <h2 className="mx-auto max-w-4xl text-3xl font-extrabold leading-tight tracking-tight sm:text-5xl md:text-6xl">
            Ask the Earth from
            <br />
            <span className="bg-gradient-to-r from-cyan-300 via-blue-400 to-violet-400 bg-clip-text text-transparent">
              a Smarter Perspective.
            </span>
          </h2>

          <p className="mx-auto mt-3 max-w-2xl text-xs leading-relaxed text-slate-400 sm:text-sm md:text-base">
            Upload multiple satellite images, enter your natural language query, and extract evidence-grounded spatial intelligence across time.
          </p>
        </section>

        {/* ================= MAIN INTERACTIVE WORKSPACE ================= */}
        <section className="mx-auto max-w-7xl px-6 pb-16 md:px-10">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* LEFT COLUMN: SATELLITE DATA INPUT */}
            <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-1 backdrop-blur-2xl transition duration-500 hover:border-cyan-400/40 shadow-2xl">
              <div className="rounded-[22px] bg-[#070d18]/90 p-6 md:p-8">
                <div className="mb-6 flex items-start justify-between">
                  <div>
                    <p className="text-xs font-semibold tracking-[0.28em] text-cyan-400">
                      SATELLITE DATA INPUT
                    </p>
                    <h3 className="mt-1 text-2xl font-bold text-white">
                      Target Imagery
                    </h3>
                    <p className="mt-1 text-xs text-slate-400">
                      Upload your satellite imagery — 5 images minimum for multi-temporal analysis.
                    </p>
                  </div>
                  {/* HISTORY ICON BUTTON REPLACING GLOBE2 */}
                  <button
                    type="button"
                    onClick={() => setIsHistoryOpen(true)}
                    title="Open Persistent Query History"
                    className="rounded-lg border border-cyan-400/30 bg-cyan-400/10 p-2.5 text-cyan-300 hover:bg-cyan-400/20 hover:border-cyan-400 hover:scale-105 transition cursor-pointer flex items-center gap-1.5 shadow-[0_0_15px_rgba(34,211,238,0.15)]"
                  >
                    <History className="h-5 w-5" />
                  </button>
                </div>

                <ImageUploader
                  images={images}
                  onImagesChange={handleImagesChange}
                />
              </div>
            </div>

            {/* RIGHT COLUMN: AI QUERY & ANALYSIS */}
            <div className="rounded-3xl border border-white/10 bg-white/[0.03] p-1 backdrop-blur-2xl transition duration-500 hover:border-violet-400/40 shadow-2xl">
              <div className="rounded-[22px] bg-[#070d18]/90 p-6 md:p-8">
                <div className="mb-6 flex items-start justify-between">
                  <div>
                    <p className="text-xs font-semibold tracking-[0.28em] text-violet-400">
                      AI INFERENCE INTERFACE
                    </p>
                    <h3 className="mt-1 text-2xl font-bold text-white">
                      Ask the Satellite
                    </h3>
                    <p className="mt-1 text-xs text-slate-400">
                      Ask questions regarding objects, change detection, or spectral anomalies.
                    </p>
                  </div>
                  <div className="rounded-lg border border-violet-400/20 bg-violet-400/5 p-2 text-violet-400">
                    <Terminal className="h-5 w-5" />
                  </div>
                </div>

                <QueryBox
                  query={query}
                  setQuery={setQuery}
                  images={images}
                  handleAnalyze={handleAnalyze}
                  isAnalyzing={isAnalyzing}
                />
              </div>
            </div>
          </div>

          {/* ================= RESULTS / INTELLIGENCE REPORT ================= */}
          <div ref={resultsRef} className="mt-12">
            <div className="mb-6 flex items-center gap-4">
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-cyan-400/30 to-cyan-400/50" />
              <span className="text-xs font-semibold tracking-[0.3em] text-cyan-300">
                INTELLIGENCE OUTPUT
              </span>
              <div className="h-px flex-1 bg-gradient-to-l from-transparent via-violet-400/30 to-violet-400/50" />
            </div>

            <ResultPanel
              analysisResult={analysisResult}
              isAnalyzing={isAnalyzing}
              currentStep={currentStep}
              error={error}
            />
          </div>

          {/* ================= MISSION TELEMETRY SUMMARY ================= */}
          <div className="mt-8">
            <ExecutionSummary
              analysisResult={analysisResult}
              isAnalyzing={isAnalyzing}
              executionTime={executionTime}
              images={images}
            />
          </div>

          {/* ================= FUTURE LAND-COVER FORECASTING ================= */}
          <div className="mt-12">
            <div className="mb-6 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-4 flex-1 w-full">
                <div className="h-px flex-1 bg-gradient-to-r from-transparent via-cyan-400/30 to-cyan-400/50" />
                <span className="text-xs font-semibold tracking-[0.3em] text-cyan-300">
                  MULTI-YEAR LAND-COVER FORECASTING
                </span>
                <div className="h-px flex-1 bg-gradient-to-l from-transparent via-violet-400/30 to-violet-400/50" />
              </div>

              {/* Mode Switcher Tabs */}
              <div className="flex items-center bg-slate-900/90 border border-slate-800 p-1 rounded-xl text-xs">
                <button
                  onClick={() => setForecasterTab("preset")}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
                    forecasterTab === "preset"
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 shadow-inner"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Preset Demo Regions
                </button>
                <button
                  onClick={() => setForecasterTab("dynamic")}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
                    forecasterTab === "dynamic"
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 shadow-inner"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Upload Custom Timeline
                </button>
              </div>
            </div>

            {forecasterTab === "preset" ? (
              <FuturePrediction defaultRegion="region_01" defaultYear={2027} />
            ) : (
              <DynamicFuturePrediction />
            )}
          </div>
        </section>

        {/* ================= FOOTER ================= */}
        <footer className="border-t border-white/10 bg-[#02050e] px-6 py-6">
          <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 text-xs text-slate-500 md:flex-row">
            <div className="flex items-center gap-2">
              <span className="text-slate-300 font-semibold">SatQuery AI</span>
              <span>— Intelligent Vision-Language Assistant for Satellite Earth Observation</span>
            </div>

            <div className="flex items-center gap-6 font-mono text-[11px] text-slate-400">
              <span className="flex items-center gap-1.5">
                <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
                256-bit Encrypted Telemetry
              </span>
              <span>•</span>
              <span>SatQuery Agentic Pipeline</span>
              <span>•</span>
              <span>Sub-meter GSD Capable</span>
            </div>
          </div>
        </footer>
      </div>

      {/* PERSISTENT QUERY HISTORY SLIDE-OVER PANEL */}
      <HistoryPanel
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        user={user}
        supabase={supabase}
        onRestoreItem={handleRestoreHistoryItem}
        openAuthModal={openAuthModal}
      />
    </div>
  );
}
