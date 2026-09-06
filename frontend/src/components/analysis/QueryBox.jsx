import React from "react";
import { Sparkles, ArrowRight, Cpu } from "lucide-react";

const MIN_IMAGES = 1;

function QueryBox({
  query,
  setQuery,
  images = [],
  handleAnalyze,
  isAnalyzing,
}) {
  const suggestions = [
    "Detect changes between these satellite images",
    "Identify land use shifts across the image set",
    "Compute thermal anomalies and heat plumes",
    "Detect vessels, infrastructure, or urban expansion",
  ];

  const imageCount = images.length;
  const hasImages = imageCount > 0;
  const hasEnoughImages = imageCount >= MIN_IMAGES;
  const isFormValid = hasImages && query.trim().length > 0;

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      if (isFormValid && !isAnalyzing) handleAnalyze();
    }
  };

  return (
    <div className="space-y-4">
      {/* TEXTAREA INPUT */}
      <div className="relative">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={4}
          placeholder={
            hasImages
              ? `Ask anything about your ${imageCount > 1 ? `${imageCount} uploaded images` : "uploaded image"}... (e.g., "Detect deforestation changes", "Count vessels in harbor")`
              : "Upload satellite images above, then enter your analysis query..."
          }
          className="w-full resize-none rounded-2xl border border-white/15 bg-black/40 p-4 text-sm text-white placeholder:text-slate-500 outline-none transition duration-200 focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/50"
        />

        <div className="absolute bottom-3 right-3 text-[10px] text-slate-500 font-mono pointer-events-none">
          Ctrl+Enter to run
        </div>
      </div>

      {/* QUICK SUGGESTIONS */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium tracking-wider text-slate-400 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-violet-400" />
            RECOMMENDED QUERIES
          </span>
          <span className="text-[11px] text-slate-500">Click to apply</span>
        </div>

        <div className="flex flex-col gap-2">
          {suggestions.map((suggestion, idx) => {
            const isCurrent = query === suggestion;
            return (
              <button
                key={idx}
                type="button"
                onClick={() => setQuery(suggestion)}
                className={`group flex items-center justify-between rounded-xl border px-3.5 py-2.5 text-left text-xs transition-all duration-200 ${
                  isCurrent
                    ? "border-violet-400/60 bg-violet-950/40 text-violet-200 shadow-[0_0_15px_rgba(167,139,250,0.15)]"
                    : "border-white/10 bg-white/[0.02] text-slate-300 hover:border-violet-400/30 hover:bg-white/[0.05] hover:text-white"
                }`}
              >
                <span className="truncate">{suggestion}</span>
                <span className="ml-2 text-violet-400 opacity-0 transition-opacity group-hover:opacity-100">
                  ↵
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* SUBMIT BUTTON */}
      <div className="pt-2">
        <button
          type="button"
          onClick={handleAnalyze}
          disabled={!isFormValid || isAnalyzing}
          className={`relative group flex w-full items-center justify-center gap-2.5 rounded-2xl py-3.5 px-6 font-semibold text-sm transition-all duration-300 ${
            !isFormValid
              ? "cursor-not-allowed border border-white/10 bg-white/5 text-slate-500 opacity-60"
              : isAnalyzing
              ? "cursor-wait border border-cyan-400/40 bg-cyan-950/50 text-cyan-300 shadow-[0_0_25px_rgba(34,211,238,0.2)]"
              : "border border-cyan-400/40 bg-gradient-to-r from-cyan-500 via-blue-500 to-violet-500 text-white shadow-[0_0_25px_rgba(34,211,238,0.25)] hover:shadow-[0_0_35px_rgba(34,211,238,0.4)] hover:scale-[1.01] active:scale-[0.99]"
          }`}
        >
          {isAnalyzing ? (
            <>
              <Cpu className="h-4 w-4 animate-spin text-cyan-400" />
              <span>Running Agentic Analysis Pipeline...</span>
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4 text-cyan-200" />
              <span>Analyze Satellite Scene{imageCount > 1 ? `s (${imageCount})` : ""}</span>
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </>
          )}
        </button>

        {/* Contextual hint below button */}
        {!hasImages ? (
          <p className="mt-2 text-center text-[11px] text-slate-500">
            Upload at least 1 satellite image above to enable analysis
          </p>
        ) : !query.trim() ? (
          <p className="mt-2 text-center text-[11px] text-slate-500">
            Enter a natural-language query above to analyze the scene
          </p>
        ) : null}
      </div>
    </div>
  );
}

export default QueryBox;
