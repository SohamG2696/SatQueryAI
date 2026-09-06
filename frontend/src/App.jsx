import React, { useState } from 'react';
import './App.css';
import HeroVideo from './components/HeroVideo';
import AudioPlayer from './components/AudioPlayer';
import AuthModal from './components/AuthModal';
import Dashboard from './components/Dashboard';
import IntroVideo from './components/IntroVideo';
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
      {/* ── INTRO VIDEO — rendered on top until dismissed ── */}
      {viewMode === 'intro' && (
        <IntroVideo onFinish={() => setViewMode('landing')} />
      )}

      {/* ── LANDING PAGE & EVERYTHING BELOW ── */}
      <div className="app">

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

        <div className="workflow">

          <div className="workflow-step">
            <span>01</span>

            <h3>UPLOAD</h3>

            <p>
              Provide your satellite imagery.
            </p>
          </div>

          <div className="workflow-line"></div>

          <div className="workflow-step">
            <span>02</span>

            <h3>ASK</h3>

            <p>
              Describe what you want to understand.
            </p>
          </div>

          <div className="workflow-line"></div>

          <div className="workflow-step">
            <span>03</span>

            <h3>ANALYZE</h3>

            <p>
              AI selects the right analysis tools.
            </p>
          </div>

          <div className="workflow-line"></div>

          <div className="workflow-step">
            <span>04</span>

            <h3>VERIFY</h3>

            <p>
              Receive evidence and confidence.
            </p>
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