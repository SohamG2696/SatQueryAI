import React, { useState } from 'react';
import './App.css';
import HeroVideo from './components/HeroVideo';
import AudioPlayer from './components/AudioPlayer';
import AuthModal from './components/AuthModal';
import Dashboard from './components/Dashboard';
import IntroVideo from './components/IntroVideo';
import ScrollBackgroundVideo from './components/ScrollBackgroundVideo';
import { AuthProvider, useAuth } from './context/AuthContext';

function MainAppContent() {
  const { user, profile, signOut, loading } = useAuth();

  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [authModalMode, setAuthModalMode] = useState('login');
  // 'intro' → 'landing' → 'dashboard'
  const [viewMode, setViewMode] = useState('intro');

  const openAuthModal = (mode = 'login') => {
    setAuthModalMode(mode);
    setAuthModalOpen(true);
  };

  const handleStartAnalysis = () => {
    setViewMode('dashboard');
  };

  const handleExplorePlatform = () => {
    if (user) {
      setViewMode('dashboard');
    } else {
      openAuthModal('login');
    }
  };

  // If viewing intro video
  if (viewMode === 'intro') {
    return <IntroVideo onFinish={() => setViewMode('landing')} />;
  }

  // If viewing analysis workspace / dashboard
  if (viewMode === 'dashboard') {
    return <Dashboard onBackToLanding={() => setViewMode('landing')} openAuthModal={openAuthModal} />;
  }

  const displayName =
    profile?.full_name ||
    user?.email?.split('@')[0] ||
    'User';

  return (
    <>
      {/* ── LANDING PAGE & EVERYTHING BELOW ── */}
      <div className="app">

        {/* Global Scroll-Driven Background Video */}
        <ScrollBackgroundVideo
          src="/video/background_video.mp4"
          fallbackSrc="/background_video.mp4"
        />

        {/* Non-intrusive Audio Player — DO NOT REMOVE */}
        <AudioPlayer />

        {/* Auth Modal for Login / Signup / Password Reset */}
        <AuthModal
          isOpen={authModalOpen}
          onClose={() => setAuthModalOpen(false)}
          initialMode={authModalMode}
        />

        {/* Navigation — Entrance animation via CSS */}
        <nav className="navbar">
          <div
            className="logo"
            onClick={() => setViewMode('landing')}
          >
            <span className="logo-dot"></span>
            SatQuery <span>AI</span>
          </div>

          <div className="nav-links">
            <a
              href="#platform"
              onClick={() => setViewMode('landing')}
            >
              Platform
            </a>

            <a
              href="#capabilities"
              onClick={() => setViewMode('landing')}
            >
              Capabilities
            </a>

            <a
              href="#how-it-works"
              onClick={() => setViewMode('landing')}
            >
              How It Works
            </a>

            <a
              href="#about"
              onClick={() => setViewMode('landing')}
            >
              About
            </a>
          </div>

          <div className="nav-actions">
            {user ? (
              <>
                <span
                  className="nav-user-greeting"
                  title={user.email}
                >
                  Hi, {displayName}
                </span>

                <button
                  className="start-btn"
                  onClick={signOut}
                >
                  Log Out
                </button>
              </>
            ) : (
              <>
                <button
                  className="login-btn"
                  onClick={() => openAuthModal('login')}
                >
                  Login
                </button>

                <button
                  className="start-btn"
                  onClick={() => openAuthModal('signup')}
                >
                  Get Started
                </button>
              </>
            )}
          </div>
        </nav>

        {/* =====================================================
          HERO SECTION
          ===================================================== */}

        <main className="hero-section">

          {/* Full-screen cinematic video */}
          <HeroVideo />

          {/* Subtle readability gradient over video */}
          <div className="hero-readability-overlay" />

          {/* Hero content */}
          <div className="hero-content">

            <p className="eyebrow">
              INTELLIGENT EARTH OBSERVATION
            </p>

            <h1>
              ASK THE EARTH.
              <br />
              <span>GET THE EVIDENCE.</span>
            </h1>

            <p className="hero-description">
              An intelligent vision-language assistant for
              satellite image analysis.
            </p>

            <div className="hero-buttons">

              <button
                className="primary-cta"
                onClick={handleStartAnalysis}
              >
                START ANALYSIS <span>→</span>
              </button>

            </div>
          </div>
        </main>

        {/* =====================================================
          CAPABILITIES
          ===================================================== */}

        <section
          id="capabilities"
          className="capabilities-section"
        >
          <p className="section-label">
            CAPABILITIES
          </p>

          <h2>
            Understand satellite imagery
            <br />
            <span>through natural language.</span>
          </h2>

          <div className="capability-grid">

            <div
              className="capability-card"
              onClick={handleStartAnalysis}
              style={{ cursor: 'pointer' }}
            >
              <span>01</span>

              <h3>
                Visual Question Answering
              </h3>

              <p>
                Ask questions about satellite imagery
                and receive evidence-grounded answers.
              </p>
            </div>

            <div
              className="capability-card"
              onClick={handleStartAnalysis}
              style={{ cursor: 'pointer' }}
            >
              <span>02</span>

              <h3>
                Change Detection
              </h3>

              <p>
                Compare imagery across time and
                understand what has changed.
              </p>
            </div>

            <div
              className="capability-card"
              onClick={handleStartAnalysis}
              style={{ cursor: 'pointer' }}
            >
              <span>03</span>

              <h3>
                Grounding & Spatial Reasoning
              </h3>

              <p>
                Identify and visually locate objects,
                regions and changes in imagery.
              </p>
            </div>

            <div
              className="capability-card"
              onClick={handleStartAnalysis}
              style={{ cursor: 'pointer' }}
            >
              <span>04</span>

              <h3>
                Optical + SAR Analysis
              </h3>

              <p>
                Combine multiple satellite modalities
                for deeper Earth observation.
              </p>
            </div>

          </div>
        </section>

        {/* =====================================================
          HOW IT WORKS
          ===================================================== */}

        <section
          id="how-it-works"
          className="workflow-section"
        >
          <div className="workflow-header-wrap">
            <p className="section-label">
              HOW IT WORKS
            </p>

            <h2>
              One question.
              <br />
              <span>
                Multiple layers of intelligence.
              </span>
            </h2>

            <p className="workflow-subtext">
              An end-to-end intelligent pipeline that validates queries, orchestrates vision-language specialist models, and delivers evidence-grounded spatial intelligence.
            </p>
          </div>

          <div className="workflow-grid">
            <div className="workflow-card" onClick={handleStartAnalysis} style={{ cursor: 'pointer' }}>
              <div className="workflow-card-top">
                <span className="workflow-step-num">01</span>
                <span className="workflow-step-tag">INPUT</span>
              </div>
              <div className="workflow-icon-wrap">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
                  <path d="M12 12v9" />
                  <path d="m8 16 4-4 4 4" />
                </svg>
              </div>
              <h3>Multi-Modal Ingestion</h3>
              <p>Upload Optical (Sentinel-2, Landsat) or SAR (Sentinel-1) satellite images and multi-temporal image pairs.</p>
              <div className="workflow-card-footer">
                <span className="workflow-card-badge">GeoTIFF · PNG · WebP</span>
              </div>
            </div>

            <div className="workflow-card" onClick={handleStartAnalysis} style={{ cursor: 'pointer' }}>
              <div className="workflow-card-top">
                <span className="workflow-step-num">02</span>
                <span className="workflow-step-tag">VALIDATION</span>
              </div>
              <div className="workflow-icon-wrap">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m3 21 1.9-5.7a8.5 8.5 0 1 1 3.8 3.8z" />
                </svg>
              </div>
              <h3>Query Normalization</h3>
              <p>Natural language parsing checks domain relevance, autocorrects typos, and infers spatial and temporal intent.</p>
              <div className="workflow-card-footer">
                <span className="workflow-card-badge">Domain Guardrail</span>
              </div>
            </div>

            <div className="workflow-card" onClick={handleStartAnalysis} style={{ cursor: 'pointer' }}>
              <div className="workflow-card-top">
                <span className="workflow-step-num">03</span>
                <span className="workflow-step-tag">INFERENCE</span>
              </div>
              <div className="workflow-icon-wrap">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                  <line x1="8" y1="21" x2="16" y2="21" />
                  <line x1="12" y1="17" x2="12" y2="21" />
                </svg>
              </div>
              <h3>Autonomous Routing</h3>
              <p>Controller dynamically orchestrates VLM reasoning, ChangeFormer VQA, and Grounding models for query goals.</p>
              <div className="workflow-card-footer">
                <span className="workflow-card-badge">Model Orchestration</span>
              </div>
            </div>

            <div className="workflow-card" onClick={handleStartAnalysis} style={{ cursor: 'pointer' }}>
              <div className="workflow-card-top">
                <span className="workflow-step-num">04</span>
                <span className="workflow-step-tag">EVIDENCE</span>
              </div>
              <div className="workflow-icon-wrap">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                  <path d="m9 12 2 2 4-4" />
                </svg>
              </div>
              <h3>Verified Evidence</h3>
              <p>Receive evidence-backed answers, localized bounding regions, change heatmaps, and confidence scores.</p>
              <div className="workflow-card-footer">
                <span className="workflow-card-badge">Confidence & Grounding</span>
              </div>
            </div>
          </div>
        </section>

        {/* =====================================================
          ABOUT SATQUERY AI  — 3-column editorial layout
          ===================================================== */}

        {/* =====================================================
          ABOUT SATQUERY AI
          ===================================================== */}

        <section id="about" className="about-section">
          <div className="about-inner">

            {/* About Header */}
            <div className="about-header">
              <div className="about-label-wrapper">
                <span className="about-label-line"></span>
                <p className="about-label">ABOUT SATQUERY-AI</p>
                <span className="about-label-line"></span>
              </div>

              <h2 className="about-title">
                Making Earth<br />
                Observation <span>Conversational.</span>
              </h2>
            </div>

            {/* 3 Cards Grid */}
            <div className="about-grid">

              {/* Card 01 */}
              <div className="about-card">
                <span className="about-card-number">01</span>
                <h3 className="about-card-title">WHAT WE ARE</h3>
                <p className="about-card-text">
                  SatQuery AI is an interactive vision-language assistant designed to help users understand remote sensing imagery through natural-language queries.
                </p>
                <p className="about-card-text">
                  It brings together <span className="highlight-cyan">AI, satellite imagery, and geospatial intelligence</span> to create a more intuitive way to explore and interpret Earth observation data.
                </p>
              </div>

              {/* Card 02 */}
              <div className="about-card">
                <span className="about-card-number">02</span>
                <h3 className="about-card-title">WHAT WE DO</h3>
                <p className="about-card-text">
                  SatQuery AI can work with <span className="highlight-cyan">optical and SAR imagery</span> to answer visual questions, identify regions of interest, detect changes over time, and generate <span className="highlight-cyan">evidence-grounded insights with confidence-aware results.</span>
                </p>
                <p className="about-card-text">
                  Instead of navigating complex remote sensing workflows, users can interact with satellite imagery simply by <span className="highlight-cyan">asking questions in natural language.</span>
                </p>
              </div>

              {/* Card 03 */}
              <div className="about-card">
                <span className="about-card-number">03</span>
                <h3 className="about-card-title">WHY WE EXIST</h3>
                <p className="about-card-text">
                  Remote sensing analysis often requires specialized knowledge, technical tools, and complex workflows. This can make valuable satellite data difficult to interpret and access.
                </p>
                <p className="about-card-text">
                  <span className="highlight-cyan">SatQuery AI aims to bridge that gap</span>—making remote sensing analysis more intuitive, explainable, and accessible while keeping visual evidence at the center of every insight.
                </p>
              </div>

            </div>

            {/* Bottom Mission & Built-For Bar */}
            <div className="about-bottom-bar">

              <div className="about-mission-box">
                <div className="about-icon-wrap">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="9" />
                    <circle cx="12" cy="12" r="4" />
                    <line x1="12" y1="2" x2="12" y2="5" />
                    <line x1="12" y1="19" x2="12" y2="22" />
                    <line x1="2" y1="12" x2="5" y2="12" />
                    <line x1="19" y1="12" x2="22" y2="12" />
                  </svg>
                </div>
                <div className="about-mission-info">
                  <span className="about-bottom-label">OUR MISSION</span>
                  <p>To make satellite image analysis more accessible, intelligent, and evidence-driven.</p>
                </div>
              </div>

              <div className="about-built-box">
                <div className="about-icon-wrap">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.71.79-1.81.2-2.55L4.5 16.5z" />
                    <path d="M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-3.05 11a22.35 22.35 0 0 1-3.95 2z" />
                  </svg>
                </div>
                <div className="about-built-info">
                  <span className="about-bottom-label">BUILT FOR</span>
                  <h4>Smart India Hackathon 2026</h4>
                  <p>ISRO · Department of Space</p>
                </div>
              </div>

            </div>

          </div>
        </section>

        {/* =====================================================
            FOOTER
            ===================================================== */}

        <footer className="footer">
          <div className="footer-left">
            <div className="logo" onClick={() => setViewMode('landing')} style={{ cursor: 'pointer' }}>
              <span className="logo-dot"></span>
              SatQuery <span>AI</span>
            </div>
            <span className="footer-status-pill">
              <span className="footer-status-dot"></span>
              ALL SYSTEMS OPERATIONAL
            </span>
          </div>

          <div className="footer-links">
            <a href="#capabilities">Capabilities</a>
            <a href="#how-it-works">Architecture</a>
            <a href="#about">About</a>
            <span className="footer-copy">© 2026 SatQuery AI · ISRO SIH</span>
          </div>
        </footer>

      </div>
    </>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <MainAppContent />
    </AuthProvider>
  );
}