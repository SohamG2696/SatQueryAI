import React, { useState, useRef } from "react";
import ImageUploader from "./ImageUploader";
import QueryBox from "./QueryBox";
import ResultPanel from "./ResultPanel";
import ExecutionSummary from "./ExecutionSummary";
import Spotlight from "./Spotlight";
import { Globe2, ShieldCheck, Terminal, AlertCircle } from "lucide-react";
import { executeQuery } from "@/services/api";
import "@/styles/analysis.css";

export default function AnalysisWorkspace({ supabase, user }) {
  const [images, setImages] = useState([]);           // multi-image array
  const [query, setQuery] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [executionTime, setExecutionTime] = useState(null);
  const [error, setError] = useState(null);

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

      const modelsInvoked = response.execution_summary?.models_invoked || [];
      const modelName = modelsInvoked.length > 0
        ? modelsInvoked.join(", ")
        : response.execution_summary?.route_selected || "SatQuery Controller";

      const keyFindings = [];
      if (response.synthesis?.key_findings?.length) {
        keyFindings.push(...response.synthesis.key_findings);
      } else {
        keyFindings.push(
          `Task Identified: ${response.task_detected.toUpperCase()}`,
          `Route Invoked: ${response.execution_summary?.route_selected || "Standard Pipeline"}`,
          `Processing Latency: ${response.execution_summary?.processing_time_seconds || elapsed}`
        );
      }

      const result = {
        rawResponse: response,
        sceneName: images[0]?.name || "Uploaded Satellite Imagery",
        imageCount: images.length,
        query: query.trim(),
        task_detected: response.task_detected,
        model: modelName,
        detectedObjects: response.task_detected === "grounding" ? "Localized Region" : response.task_detected.replace("_", " ").toUpperCase(),
        objectType: response.execution_summary?.route_selected || "Geospatial Target",
        changesDetected: response.task_detected === "change_vqa" ? "Change Detection Applied" : "Single Scene Inspection",
        confidence: response.confidence != null ? `${Math.round(response.confidence * 100)}%` : "Not available",
        message: response.answer,
        keyFindings: keyFindings,
        recommendation: response.verification?.notes || "Analysis generated and verified by SatQuery AI pipeline.",
        visual_evidence: response.visual_evidence,
        spectralData: {
          taskDetected: response.task_detected,
          modelsInvoked: modelsInvoked.join(", ") || "Active Module",
          routeSelected: response.execution_summary?.route_selected || "Standard",
          imageCount: `${images.length} asset${images.length !== 1 ? "s" : ""}`,
        },
      };

      setAnalysisResult(result);
      setIsAnalyzing(false);

      if (supabase && user?.id) {
        try {
          await supabase.from("analyses").insert({
            user_id: user.id,
            query: query || "Satellite Scene Inspection",
            ai_response: result.message,
            created_at: new Date().toISOString(),
          });
        } catch (dbErr) {
          console.warn("Supabase analyses table insert note:", dbErr.message);
        }
      }

      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (err) {
      setIsAnalyzing(false);
      setError(err.message || "Unable to connect to SatQuery AI backend. Please ensure the backend server is running on port 8000.");
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    }
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
                  <div className="rounded-lg border border-cyan-400/20 bg-cyan-400/5 p-2 text-cyan-400">
                    <Globe2 className="h-5 w-5" />
                  </div>
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
    </div>
  );
}
