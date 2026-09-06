import { useEffect, useRef } from 'react';

export default function AudioPlayer() {
  const audioRef = useRef(null);

  useEffect(() => {
    const audio = new Audio('/audio/satquery_reference_audio.mp3');
    audio.loop = true;
    audio.volume = 0.45;
    audioRef.current = audio;

    const attemptPlay = () => {
      audio.play().then(() => {
        removeListeners();
      }).catch((err) => {
        console.log('Audio autoplay deferred until user interaction:', err.message);
      });
    };

    const handleUserGesture = () => {
      if (audioRef.current && audioRef.current.paused) {
        audioRef.current.play().then(() => {
          removeListeners();
        }).catch(e => console.log('Audio play error:', e));
      }
    };

    const removeListeners = () => {
      window.removeEventListener('click', handleUserGesture);
      window.removeEventListener('keydown', handleUserGesture);
      window.removeEventListener('touchstart', handleUserGesture);
      window.removeEventListener('pointerdown', handleUserGesture);
    };

    attemptPlay();

    window.addEventListener('click', handleUserGesture);
    window.addEventListener('keydown', handleUserGesture);
    window.addEventListener('touchstart', handleUserGesture);
    window.addEventListener('pointerdown', handleUserGesture);

    return () => {
      removeListeners();
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);

  return null;
}
