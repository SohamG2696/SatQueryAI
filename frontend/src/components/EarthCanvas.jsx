import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import Earth from './Earth';
import Satellite from './Satellite';
import SpaceEnvironment from './SpaceEnvironment';

export default function EarthCanvas() {
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        camera={{
          position: [0, 0, 6.8],
          fov: 42,
          near: 0.1,
          far: 1000
        }}
        dpr={[1, 2]}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: 'high-performance'
        }}
        style={{ background: 'transparent' }}
      >
        <Suspense fallback={null}>
          <SpaceEnvironment />
          <Earth />
          <Satellite />
        </Suspense>
      </Canvas>
    </div>
  );
}
