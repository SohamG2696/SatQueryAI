import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { supabase } from '../lib/supabase';
import AnalysisWorkspace from './analysis/AnalysisWorkspace';
import ForecastingWorkspace from './forecasting/ForecastingWorkspace';
import ScrollBackgroundVideo from './ScrollBackgroundVideo';
import { Layers, TrendingUp } from 'lucide-react';

export default function Dashboard({ onBackToLanding, openAuthModal }) {
  const { user, profile, signOut } = useAuth();
  const [activeTab, setActiveTab] = useState('analysis'); // 'analysis' | 'forecasting'

  const displayName = profile?.full_name || user?.email?.split('@')[0] || 'Researcher';

  return (
    <div className="dashboard-page flex flex-col min-h-screen bg-transparent relative">
      {/* Global Scroll-Driven Background Video */}
      <ScrollBackgroundVideo
        src="/video/background_video.mp4"
        fallbackSrc="/background_video.mp4"
        overlayOpacity={0.35}
      />

      {/* Dashboard Top Header Navigation */}
      <header className="dashboard-nav z-50">
        <div className="logo" onClick={onBackToLanding} style={{ cursor: 'pointer' }}>
          <span className="logo-dot"></span>
          SatQuery <span>AI</span>
        </div>

        {/* Center Mode Switcher Tabs */}
        <div className="dash-nav-tabs hidden sm:flex items-center bg-[#070d18]/80 border border-white/10 p-1 rounded-full text-xs backdrop-blur-md shadow-lg">
          <button
            type="button"
            onClick={() => setActiveTab('analysis')}
            className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full font-medium transition-all cursor-pointer ${
              activeTab === 'analysis'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-[0_0_12px_rgba(34,211,238,0.2)]'
                : 'text-slate-400 hover:text-white border border-transparent'
            }`}
          >
            <Layers className="h-3.5 w-3.5" />
            <span>Analysis Workspace</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('forecasting')}
            className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full font-medium transition-all cursor-pointer ${
              activeTab === 'forecasting'
                ? 'bg-violet-500/20 text-violet-300 border border-violet-500/40 shadow-[0_0_12px_rgba(167,139,250,0.2)]'
                : 'text-slate-400 hover:text-white border border-transparent'
            }`}
          >
            <TrendingUp className="h-3.5 w-3.5" />
            <span>Land-Cover Forecaster</span>
          </button>
        </div>

        <div className="dash-user-section">
          {user ? (
            <>
              <div className="user-badge">
                <span className="user-dot"></span>
                <span className="user-name">{displayName}</span>
              </div>

              <button className="secondary-cta dash-landing-btn" onClick={onBackToLanding}>
                Back
              </button>

              <button className="start-btn dash-logout-btn" onClick={signOut}>
                Log Out
              </button>
            </>
          ) : (
            <>
              <button className="secondary-cta dash-landing-btn" onClick={onBackToLanding}>
                 Back
              </button>

              <button className="start-btn" onClick={() => openAuthModal && openAuthModal('login')}>
                Sign In
              </button>
            </>
          )}
        </div>
      </header>

      {/* Main Integrated Workspace / Forecaster */}
      <main className="flex-1">
        {activeTab === 'forecasting' ? (
          <ForecastingWorkspace onBackToAnalysis={() => setActiveTab('analysis')} />
        ) : (
          <AnalysisWorkspace
            supabase={supabase}
            user={user}
            openAuthModal={openAuthModal}
            onNavigateToForecasting={() => setActiveTab('forecasting')}
          />
        )}
      </main>
    </div>
  );
}
