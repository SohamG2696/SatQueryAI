import React, { useRef, useState, useEffect, useCallback } from 'react';

/**
 * IntroVideo — Full-screen cinematic intro that plays once per page load.
 *
 * Props:
 *   onFinish — called when video ends or user skips
 */
function IntroVideo({ onFinish }) {
  const videoRef = useRef(null);
  const [fadingOut, setFadingOut] = useState(false);

  // Trigger the fade-out → call onFinish
  const handleExit = useCallback(() => {
    if (fadingOut) return; // prevent double-fire
    setFadingOut(true);
    setTimeout(() => {
      onFinish();
    }, 700); // matches CSS transition duration
  }, [fadingOut, onFinish]);

  // Auto-transition when video ends
  const handleEnded = useCallback(() => {
    handleExit();
  }, [handleExit]);

  // Attempt autoplay (browsers require muted for autoplay)
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;
    vid.muted = true;
    const playPromise = vid.play();
    if (playPromise !== undefined) {
      playPromise.catch(() => {
        // Autoplay blocked — video stays paused; user can still click Skip
      });
    }
  }, []);

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        background: '#000',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        opacity: fadingOut ? 0 : 1,
        transition: 'opacity 0.7s ease',
        pointerEvents: fadingOut ? 'none' : 'auto',
      }}
    >
      {/* ── Cinematic intro video ── */}
      <video
        ref={videoRef}
        src="/Before Landing Page Final 1.mp4"
        autoPlay
        muted
        playsInline
        onEnded={handleEnded}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100vw',
          height: '100vh',
          objectFit: 'cover',
          display: 'block',
        }}
      />

      {/* ── Skip Intro button — top-right corner ── */}
      <button
        onClick={handleExit}
        aria-label="Skip intro video"
        style={{
          position: 'absolute',
          top: '1.5rem',
          right: '1.75rem',
          zIndex: 10000,
          display: 'flex',
          alignItems: 'center',
          gap: '0.4rem',
          padding: '0.5rem 1.1rem',
          background: 'rgba(3, 7, 13, 0.65)',
          border: '1px solid rgba(97, 216, 255, 0.35)',
          borderRadius: '999px',
          color: 'rgba(242, 247, 250, 0.85)',
          fontSize: '0.75rem',
          fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          backdropFilter: 'blur(8px)',
          cursor: 'pointer',
          transition: 'background 0.2s, border-color 0.2s, color 0.2s',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(97, 216, 255, 0.15)';
          e.currentTarget.style.borderColor = 'rgba(97, 216, 255, 0.7)';
          e.currentTarget.style.color = '#61d8ff';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(3, 7, 13, 0.65)';
          e.currentTarget.style.borderColor = 'rgba(97, 216, 255, 0.35)';
          e.currentTarget.style.color = 'rgba(242, 247, 250, 0.85)';
        }}
      >
        Skip Intro
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <polyline points="13 17 18 12 13 7" />
          <polyline points="6 17 11 12 6 7" />
        </svg>
      </button>
    </div>
  );
}

export default IntroVideo;
