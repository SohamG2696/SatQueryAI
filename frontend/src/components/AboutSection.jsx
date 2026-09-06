import React from 'react';

export default function AboutSection() {
  return (
    <section id="about" className="about-section">
      {/* Background technical grid and subtle line details */}
      <div className="about-grid-bg" />
      <div className="about-tech-line top-line" />

      <div className="about-container">
        {/* Eyebrow and Headline */}
        <div className="about-header">
          <p className="eyebrow">
            ABOUT SATQUERY AI
          </p>

          <h2 className="about-headline">
            Making <span>Earth Observation</span>
            <br />
            Conversational.
          </h2>

          <p className="about-tagline">
            Ask questions. Explore imagery. Understand Earth.
          </p>
        </div>

        <div className="about-divider" />

        {/* Section 01: WHAT WE ARE */}
        <div className="about-block">
          <div className="about-block-label">
            <span className="num">01</span>
            <h3>WHAT WE ARE</h3>
          </div>
          <div className="about-block-content">
            <p className="lead-p">
              <strong>SatQuery AI</strong> is an interactive vision-language assistant designed to help users understand remote sensing imagery through natural-language queries.
            </p>
            <p className="body-p">
              It brings together AI, satellite imagery, and geospatial intelligence to create a more intuitive way to explore and interpret Earth observation data.
            </p>
          </div>
        </div>

        <div className="about-divider" />

        {/* Section 02: WHAT WE DO */}
        <div className="about-block">
          <div className="about-block-label">
            <span className="num">02</span>
            <h3>WHAT WE DO</h3>
          </div>
          <div className="about-block-content">
            <p className="body-p">
              SatQuery AI can work with optical and SAR imagery to answer visual questions, identify regions of interest, detect changes over time, and generate evidence-grounded insights with confidence-aware results.
            </p>
            <div className="highlight-box">
              Instead of navigating complex remote sensing workflows, users can interact with satellite imagery simply by asking questions in natural language.
            </div>

            {/* Mini Capability Cards */}
            <div className="about-cards-grid">
              <div className="about-mini-card">
                <div className="card-header">
                  <span className="card-indicator" />
                  <h4>OPTICAL IMAGERY</h4>
                </div>
                <p>Analyze visual information from optical satellite imagery.</p>
              </div>

              <div className="about-mini-card">
                <div className="card-header">
                  <span className="card-indicator" />
                  <h4>SAR IMAGERY</h4>
                </div>
                <p>Work with radar-based Earth observation data.</p>
              </div>

              <div className="about-mini-card">
                <div className="card-header">
                  <span className="card-indicator" />
                  <h4>VISUAL QUESTIONS</h4>
                </div>
                <p>Ask questions about what appears in satellite imagery.</p>
              </div>

              <div className="about-mini-card">
                <div className="card-header">
                  <span className="card-indicator" />
                  <h4>CHANGE DETECTION</h4>
                </div>
                <p>Identify changes across different points in time.</p>
              </div>

              <div className="about-mini-card">
                <div className="card-header">
                  <span className="card-indicator" />
                  <h4>REGION OF INTEREST</h4>
                </div>
                <p>Focus analysis on specific geographic areas.</p>
              </div>

              <div className="about-mini-card">
                <div className="card-header">
                  <span className="card-indicator" />
                  <h4>EVIDENCE-GROUNDED RESULTS</h4>
                </div>
                <p>Present insights supported by visual evidence and confidence.</p>
              </div>
            </div>
          </div>
        </div>

        <div className="about-divider" />

        {/* Section 03: WHY WE EXIST */}
        <div className="about-block">
          <div className="about-block-label">
            <span className="num">03</span>
            <h3>WHY WE EXIST</h3>
          </div>
          <div className="about-block-content">
            <p className="body-p">
              Remote sensing analysis often requires specialized knowledge, technical tools, and complex workflows. This can make valuable satellite data difficult to interpret and access.
            </p>
            <blockquote className="why-hero-quote">
              SatQuery AI aims to bridge that gap—making remote sensing analysis more intuitive, explainable, and accessible while keeping the underlying visual evidence at the center of every insight.
            </blockquote>
          </div>
        </div>

        {/* Mission Statement */}
        <div className="about-mission-card">
          <span className="mission-tag">OUR MISSION</span>
          <h2>
            "To make satellite image analysis more accessible, intelligent, and evidence-driven."
          </h2>
        </div>

        {/* Credibility Footer */}
        <div className="about-credibility-footer">
          <div className="cred-badge">
            <span className="cred-label">BUILT FOR</span>
            <span className="cred-title">Smart India Hackathon 2026</span>
          </div>
          <div className="cred-org">
            <span>ISRO · Department of Space</span>
          </div>
        </div>
      </div>
    </section>
  );
}
