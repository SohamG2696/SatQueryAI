import React, { useRef, useEffect } from 'react';

export default function HeroVideo() {
  const videoRef = useRef(null);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.play().catch(err => {
        // Autoplay restriction — browser will typically play muted video without issue
        console.log('Hero video play deferred:', err.message);
      });
    }
  }, []);

  return (
    <div className="hero-video-bg">
      <video
        ref={videoRef}
        className="hero-video-el"
        autoPlay
        loop
        muted
        playsInline
        preload="auto"
      >
        <source src="/video/Final-Vdo-SIH.mp4" type="video/mp4" />
      </video>
    </div>
  );
}
