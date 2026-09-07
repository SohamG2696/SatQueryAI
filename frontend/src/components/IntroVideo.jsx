import React, { useRef, useState, useEffect, useCallback } from 'react';
import { SkipForward } from 'lucide-react';

/**
 * IntroVideo — Full-screen cinematic intro video with unmuted audio playing from start.
 * No prompts, no mute/unmute buttons, directly plays audio with video.
 */
function IntroVideo({ onFinish }) {
  const videoRef = useRef(null);
  const [fadingOut, setFadingOut] = useState(false);

  // Trigger the fade-out → transition to landing page
  const handleExit = useCallback(() => {
    if (fadingOut) return;
    setFadingOut(true);
    setTimeout(() => {
      onFinish();
    }, 600);
  }, [fadingOut, onFinish]);

  // Auto-transition when video finishes
  const handleEnded = useCallback(() => {
    handleExit();
  }, [handleExit]);

  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;

    vid.volume = 1.0;
    vid.playbackRate = 1.25;
    vid.muted = false;

    // Attempt direct unmuted playback
    vid.play().catch(() => {
      // Fallback to muted playback only if browser security prevents unmuted autoplay
      vid.muted = true;
      vid.playbackRate = 1.25;
      vid.play().catch(() => {});
    });

    // Unmute on genuine user gesture (click, tap, keypress)
    const handleGesture = () => {
      if (vid && vid.muted) {
        vid.muted = false;
        vid.volume = 1.0;
        vid.playbackRate = 1.25;
      }
      removeListeners();
    };

    const removeListeners = () => {
      window.removeEventListener('pointerdown', handleGesture);
      window.removeEventListener('click', handleGesture);
      window.removeEventListener('keydown', handleGesture);
      window.removeEventListener('touchstart', handleGesture);
    };

    window.addEventListener('pointerdown', handleGesture);
    window.addEventListener('click', handleGesture);
    window.addEventListener('keydown', handleGesture);
    window.addEventListener('touchstart', handleGesture);

    return () => {
      removeListeners();
    };
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
        transition: 'opacity 0.6s ease',
        pointerEvents: fadingOut ? 'none' : 'auto',
      }}
    >
      {/* ── Cinematic intro video with audio ── */}
      <video
        ref={videoRef}
        src="/Before%20Landing%20Page%20Final%201.mp4"
        autoPlay
        playsInline
        preload="auto"
        onPlay={(e) => {
          e.currentTarget.playbackRate = 1.25;
        }}
        onEnded={handleEnded}
        onError={handleExit}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100vw',
          height: '100vh',
          objectFit: 'cover',
          display: 'block',
        }}
      />

      {/* ── Skip Intro button ── */}
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
          gap: '0.45rem',
          padding: '0.55rem 1.25rem',
          background: 'rgba(3, 7, 18, 0.75)',
          border: '1px solid rgba(97, 216, 255, 0.35)',
          borderRadius: '999px',
          color: 'rgba(242, 247, 250, 0.9)',
          fontSize: '0.78rem',
          fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
          fontWeight: 600,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          backdropFilter: 'blur(12px)',
          cursor: 'pointer',
          boxShadow: '0 0 20px rgba(0, 0, 0, 0.5)',
          transition: 'all 0.2s ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = 'rgba(34, 211, 238, 0.2)';
          e.currentTarget.style.borderColor = 'rgba(34, 211, 238, 0.8)';
          e.currentTarget.style.color = '#22d3ee';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = 'rgba(3, 7, 18, 0.75)';
          e.currentTarget.style.borderColor = 'rgba(97, 216, 255, 0.35)';
          e.currentTarget.style.color = 'rgba(242, 247, 250, 0.9)';
        }}
      >
        <span>Skip Intro</span>
        <SkipForward size={14} />
      </button>
    </div>
  );
}

export default IntroVideo;
