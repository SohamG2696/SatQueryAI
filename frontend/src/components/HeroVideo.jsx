import React, { useRef, useEffect } from 'react';

export default function HeroVideo() {
  const videoRef = useRef(null);

  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;

    vid.muted = false;
    vid.volume = 0.8;

    const playPromise = vid.play();
    if (playPromise !== undefined) {
      playPromise.catch(() => {
        // If autoplay unmuted is restricted by browser policy, play muted and unmute on first gesture
        vid.muted = true;
        vid.play().catch(() => {});
      });
    }

    const handleGesture = () => {
      if (vid && vid.muted) {
        vid.muted = false;
        vid.volume = 0.8;
        vid.play().catch(() => {});
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
    <div className="hero-video-bg relative">
      <video
        ref={videoRef}
        className="hero-video-el"
        autoPlay
        loop
        playsInline
        preload="auto"
      >
        <source src="/video/Final-Vdo-SIH.mp4" type="video/mp4" />
      </video>
    </div>
  );
}
