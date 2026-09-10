import React, { useRef, useEffect } from 'react';

export default function ScrollBackgroundVideo({
  src = '/video/background_video.mp4',
  fallbackSrc = '/background_video.mp4',
  overlayOpacity = 0.35,
}) {
  const videoRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    video.muted = true;
    video.defaultMuted = true;
    video.pause();

    let rafId = null;
    let targetTime = 0;
    let isSeeking = false;

    const onSeeked = () => {
      isSeeking = false;
    };
    video.addEventListener('seeked', onSeeked);

    const calcProgress = () => {
      const scrollHeight = document.documentElement.scrollHeight;
      const clientHeight = window.innerHeight;
      const maxScroll = scrollHeight - clientHeight;
      if (maxScroll <= 0) return 0;
      const currentScroll = window.scrollY || window.pageYOffset || 0;
      return Math.max(0, Math.min(1, currentScroll / maxScroll));
    };

    const updateTargetTime = () => {
      if (!video || !video.duration) return;
      const progress = calcProgress();
      // Clamp end time to duration - 0.08s to prevent EOF buffer stalls/latency at the end of scroll
      const maxUsableTime = Math.max(0, video.duration - 0.08);
      targetTime = progress * maxUsableTime;
    };

    const onMeta = () => {
      video.pause();
      updateTargetTime();
    };

    video.addEventListener('loadedmetadata', onMeta);
    video.addEventListener('canplay', onMeta);
    if (video.readyState >= 1) {
      onMeta();
    }

    const onScroll = () => {
      updateTargetTime();
    };

    // Smooth GPU sync loop
    const syncLoop = () => {
      if (video && video.duration && !isSeeking) {
        const diff = targetTime - video.currentTime;
        if (Math.abs(diff) > 0.015) {
          isSeeking = true;
          const safeTime = Math.max(0, Math.min(targetTime, Math.max(0, video.duration - 0.08)));
          try {
            if ('fastSeek' in video && typeof video.fastSeek === 'function') {
              video.fastSeek(safeTime);
            } else {
              video.currentTime = safeTime;
            }
          } catch {
            video.currentTime = safeTime;
          }
        }
      }
      rafId = requestAnimationFrame(syncLoop);
    };

    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    updateTargetTime();
    rafId = requestAnimationFrame(syncLoop);

    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      if (rafId) cancelAnimationFrame(rafId);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('loadedmetadata', onMeta);
      video.removeEventListener('canplay', onMeta);
    };
  }, [src, fallbackSrc]);

  return (
    <div className="fixed-scroll-video-wrapper" aria-hidden="true">
      <video
        ref={videoRef}
        className="fixed-scroll-video-el"
        muted
        playsInline
        preload="auto"
        disablePictureInPicture
        disableRemotePlayback
      >
        <source src={src} type="video/mp4" />
        {fallbackSrc && <source src={fallbackSrc} type="video/mp4" />}
      </video>

      {/* Cinematic dark overlay & subtle vignette for contrast */}
      <div
        className="fixed-scroll-video-overlay"
        style={{ backgroundColor: `rgba(3, 7, 13, ${overlayOpacity})` }}
      />
      <div className="fixed-scroll-video-vignette" />
    </div>
  );
}

