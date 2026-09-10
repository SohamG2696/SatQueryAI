import React, { useRef, useEffect } from 'react';

export default function HeroVideo() {
  const videoRef = useRef(null);

  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;

    vid.muted = true;
    vid.defaultMuted = true;
    vid.playbackRate = 1.0;

    const playVideo = () => {
      const playPromise = vid.play();
      if (playPromise !== undefined) {
        playPromise.catch(() => {});
      }
    };

    if (vid.readyState >= 2) {
      playVideo();
    } else {
      vid.addEventListener('loadeddata', playVideo, { once: true });
      vid.addEventListener('canplay', playVideo, { once: true });
    }

    // Seamless loop: reset right before EOF stall to eliminate loop stutter/latency
    const onTimeUpdate = () => {
      if (vid.duration && vid.currentTime >= vid.duration - 0.2) {
        vid.currentTime = 0;
        vid.play().catch(() => {});
      }
    };

    vid.addEventListener('timeupdate', onTimeUpdate);
    playVideo();

    const handleGesture = () => {
      if (vid) {
        vid.muted = false;
        vid.volume = 0.8;
      }
      removeListeners();
    };

    const removeListeners = () => {
      window.removeEventListener('pointerdown', handleGesture);
      window.removeEventListener('click', handleGesture);
      window.removeEventListener('keydown', handleGesture);
      window.removeEventListener('touchstart', handleGesture);
    };

    window.addEventListener('pointerdown', handleGesture, { once: true });
    window.addEventListener('click', handleGesture, { once: true });
    window.addEventListener('keydown', handleGesture, { once: true });
    window.addEventListener('touchstart', handleGesture, { once: true });

    return () => {
      removeListeners();
      vid.removeEventListener('loadeddata', playVideo);
      vid.removeEventListener('canplay', playVideo);
      vid.removeEventListener('timeupdate', onTimeUpdate);
    };
  }, []);

  return (
    <div className="hero-video-bg relative">
      <video
        ref={videoRef}
        className="hero-video-el"
        autoPlay
        loop
        muted
        playsInline
        preload="auto"
        disablePictureInPicture
        disableRemotePlayback
      >
        <source src="/video/Final-Vdo-SIH.mp4" type="video/mp4" />
      </video>
    </div>
  );
}
