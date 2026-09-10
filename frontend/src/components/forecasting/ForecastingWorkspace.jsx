import React from "react";
import { ArrowLeft, TrendingUp, Sparkles, ShieldCheck } from "lucide-react";
import DynamicFuturePrediction from "../DynamicFuturePrediction";

export default function ForecastingWorkspace({ onBackToAnalysis }) {
  return (
    <div className="relative min-h-screen text-slate-100 selection:bg-violet-500/30 selection:text-violet-200">
      <div className="relative z-10 mx-auto max-w-7xl px-6 py-8 md:px-10">
        
        {/* Navigation / Header Bar */}
        <div className="mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-white/10 pb-6">
          <button
            type="button"
            onClick={onBackToAnalysis}
            className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-300 backdrop-blur-md transition-all hover:border-cyan-400/50 hover:bg-cyan-500/10 hover:text-cyan-200 hover:scale-105 cursor-pointer w-fit shadow-lg"
          >
            <ArrowLeft className="h-4 w-4" />
            <span>Back to Analysis Workspace</span>
          </button>

          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            <span className="inline-block h-2 w-2 rounded-full bg-violet-400 animate-pulse" />
            <span>MULTI-YEAR PREDICTIVE AI ENGINE</span>
          </div>
        </div>

        {/* Page Hero Section */}
        <section className="mb-10 text-center">
          <div className="mx-auto mb-4 inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-500/10 px-4 py-1.5 text-xs font-semibold tracking-wider text-violet-300 backdrop-blur-md shadow-[0_0_15px_rgba(167,139,250,0.15)]">
            <Sparkles className="h-3.5 w-3.5 text-violet-400" />
            <span>EXTRAPOLATIVE GEO-INTELLIGENCE</span>
          </div>

          <h1 className="mx-auto max-w-4xl text-3xl font-extrabold leading-tight tracking-tight sm:text-5xl md:text-6xl text-white">
            Multi-Year Land-Cover
            <br />
            <span className="bg-gradient-to-r from-cyan-300 via-violet-400 to-fuchsia-400 bg-clip-text text-transparent">
              Forecasting Engine.
            </span>
          </h1>

          <p className="mx-auto mt-4 max-w-2xl text-xs leading-relaxed text-slate-400 sm:text-sm md:text-base">
            Upload multi-temporal Sentinel-2 GeoTIFF datasets across multiple years. Extrapolate future land-cover dynamics, urban sprawl, vegetation shifts, and water coverage with AI trend analytics.
          </p>
        </section>

        {/* Main Dynamic Timeline Predictor */}
        <section className="pb-16">
          <DynamicFuturePrediction />
        </section>

        {/* Footer */}
        <footer className="border-t border-white/10 pt-6 pb-10 text-xs text-slate-500 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="text-slate-300 font-semibold">SatQuery AI</span>
            <span>— Predictive Land-Cover Modeling & Change Analysis</span>
          </div>
          <div className="flex items-center gap-4 font-mono text-[11px] text-slate-400">
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              SCL Band Validated
            </span>
            <span>•</span>
            <span>Sub-pixel Trend Extrapolation</span>
          </div>
        </footer>

      </div>
    </div>
  );
}
