import React, { useMemo } from 'react';
import { Stars } from '@react-three/drei';
import * as THREE from 'three';

// Generate High-Res Deep Space Milky Way & Cosmic Dust Background Texture (2048x1024)
function generateMilkyWayTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 2048;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d');

  // Deep space base
  ctx.fillStyle = '#02050b';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Diagonal Milky Way Galactic Dust Band (Top-Left to Center-Right)
  ctx.save();
  ctx.translate(canvas.width * 0.35, canvas.height * 0.35);
  ctx.rotate(-Math.PI * 0.22); // Diagonal angle matching reference frame

  // Galactic Core Glow
  const coreGrad = ctx.createRadialGradient(0, 0, 0, 0, 0, 600);
  coreGrad.addColorStop(0, 'rgba(215, 185, 235, 0.45)');
  coreGrad.addColorStop(0.25, 'rgba(120, 150, 220, 0.28)');
  coreGrad.addColorStop(0.55, 'rgba(40, 80, 150, 0.15)');
  coreGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = coreGrad;
  ctx.fillRect(-1000, -350, 2000, 700);

  // Interstellar Dust Veins & Nebula Pockets
  for (let i = 0; i < 180; i++) {
    const x = (Math.random() - 0.5) * 1600;
    const y = (Math.random() - 0.5) * 300;
    const r = Math.random() * 80 + 20;

    const dustGrad = ctx.createRadialGradient(x, y, 0, x, y, r);
    dustGrad.addColorStop(0, `rgba(${160 + Math.random() * 80}, ${180 + Math.random() * 70}, 240, ${Math.random() * 0.25})`);
    dustGrad.addColorStop(1, 'rgba(0,0,0,0)');

    ctx.fillStyle = dustGrad;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();

  // Draw Faint Embedded Galactic Micro-Stars
  ctx.fillStyle = '#ffffff';
  for (let i = 0; i < 1200; i++) {
    const x = Math.random() * canvas.width;
    const y = Math.random() * canvas.height;
    const size = Math.random() * 1.5 + 0.3;
    const opacity = Math.random() * 0.8 + 0.2;

    ctx.globalAlpha = opacity;
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1.0;

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// Generate Texture for Lower-Left Spiral Galaxy Celestial Object
function generateSpiralGalaxyTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  const cx = 256;
  const cy = 256;

  ctx.fillStyle = 'rgba(0,0,0,0)';
  ctx.clearRect(0, 0, 512, 512);

  // Bright Nucleus
  const coreGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 60);
  coreGrad.addColorStop(0, 'rgba(255, 245, 210, 0.95)');
  coreGrad.addColorStop(0.3, 'rgba(180, 210, 255, 0.6)');
  coreGrad.addColorStop(0.7, 'rgba(80, 140, 220, 0.25)');
  coreGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = coreGrad;
  ctx.beginPath();
  ctx.arc(cx, cy, 60, 0, Math.PI * 2);
  ctx.fill();

  // Spiral Arms
  ctx.strokeStyle = 'rgba(140, 190, 255, 0.35)';
  ctx.lineWidth = 12;
  
  for (let arm = 0; arm < 2; arm++) {
    ctx.beginPath();
    const offset = arm * Math.PI;
    for (let theta = 0; theta < Math.PI * 3; theta += 0.08) {
      const r = 18 + theta * 22;
      const angle = theta + offset;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;
      if (theta === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

export default function SpaceEnvironment({ earthCenter = [1.55, -0.15, 0] }) {
  const milkyWayTexture = useMemo(() => generateMilkyWayTexture(), []);
  const galaxyTexture = useMemo(() => generateSpiralGalaxyTexture(), []);

  // Thin cyan orbital path curve around Earth
  const lineGeometry = useMemo(() => {
    const points = [];
    const radiusX = 2.75;
    const radiusZ = 2.3;
    const segments = 128;

    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const x = earthCenter[0] + Math.cos(angle) * radiusX - 1.25;
      const y = earthCenter[1] + 1.58;
      const z = earthCenter[2] + Math.sin(angle) * radiusZ;
      points.push(new THREE.Vector3(x, y, z));
    }
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [earthCenter]);

  return (
    <group>
      {/* Primary Directional Sunlight matching Reference Image (From top-left) */}
      <directionalLight
        position={[-7, 4.5, 6]}
        intensity={3.6}
        color="#ffffff"
      />

      {/* Ambient Deep Space Soft Light */}
      <ambientLight color="#0a1829" intensity={0.4} />

      {/* Cyan Limb & Atmosphere Accent Fill */}
      <directionalLight
        position={[6, -3, -4]}
        intensity={0.9}
        color="#38bdf8"
      />

      {/* Milky Way Deep-Space Backdrop Plane */}
      <mesh position={[0, 0, -25]}>
        <planeGeometry args={[70, 40]} />
        <meshBasicMaterial map={milkyWayTexture} transparent={true} opacity={0.88} />
      </mesh>

      {/* Distinct Spiral Galaxy Celestial Object in Lower-Left Space Region */}
      <mesh position={[-2.8, -2.1, -12]} rotation={[0.4, 0.2, -0.6]} scale={[1.8, 1.2, 1]}>
        <planeGeometry args={[2, 2]} />
        <meshBasicMaterial
          map={galaxyTexture}
          transparent={true}
          opacity={0.75}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* 3D Deep Space Stars Field */}
      <Stars
        radius={70}
        depth={35}
        count={3200}
        factor={3.8}
        saturation={0.6}
        fade={true}
        speed={0.4}
      />

      {/* Subtle Cyan Orbital Path Line around Earth */}
      <line geometry={lineGeometry}>
        <lineBasicMaterial
          color="#38bdf8"
          transparent={true}
          opacity={0.25}
          linewidth={1}
        />
      </line>
    </group>
  );
}
