import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Custom Photorealistic Atmosphere Fresnel Shader
function createAtmosphereShader() {
  return new THREE.ShaderMaterial({
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vWorldPosition;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPos.xyz;
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      varying vec3 vWorldPosition;
      uniform vec3 color;
      uniform vec3 sunPosition;

      void main() {
        vec3 viewDir = normalize(cameraPosition - vWorldPosition);
        vec3 sunDir = normalize(sunPosition - vWorldPosition);
        
        // Atmosphere Fresnel rim reflection
        float fresnel = pow(1.0 - max(0.0, dot(vNormal, viewDir)), 3.0);
        
        // Sun illumination factor on atmosphere
        float sunDot = max(0.0, dot(vNormal, sunDir));
        
        // Electric cyan-blue atmosphere scatter on daylight limb
        vec3 atmosphereColor = mix(color * 0.5, vec3(0.38, 0.85, 1.0), sunDot * 0.85);
        float alpha = fresnel * (0.2 + 0.8 * sunDot);
        
        gl_FragColor = vec4(atmosphereColor, max(0.0, alpha * 0.95));
      }
    `,
    uniforms: {
      color: { value: new THREE.Color('#25a0e8') },
      sunPosition: { value: new THREE.Vector3(-7, 4.5, 6) }
    },
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false
  });
}

// Convert Lat/Lon coordinates to Canvas X/Y pixels (Equirectangular)
function latLonToCanvas(lat, lon, width = 2048, height = 1024) {
  const x = ((lon + 180) / 360) * width;
  const y = ((90 - lat) / 180) * height;
  return [x, y];
}

// Generate High-Resolution Photorealistic Satellite-Grade Earth Day Map (2048x1024)
function generatePhotorealisticDayTexture() {
  const width = 2048;
  const height = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  // 1. Deep Ocean Base with Bathymetric Gradients
  const oceanGrad = ctx.createLinearGradient(0, 0, 0, height);
  oceanGrad.addColorStop(0, '#030a16');
  oceanGrad.addColorStop(0.35, '#071d38');
  oceanGrad.addColorStop(0.5, '#041326');
  oceanGrad.addColorStop(0.7, '#071d38');
  oceanGrad.addColorStop(1, '#030a16');
  ctx.fillStyle = oceanGrad;
  ctx.fillRect(0, 0, width, height);

  // Helper to draw realistic geographical landmass polygons with smooth curves & shallow coastal shelves
  const drawGeographicLand = (latLonPoints, baseColor, desertColor = null) => {
    const canvasPoints = latLonPoints.map(([lat, lon]) => latLonToCanvas(lat, lon, width, height));

    // Coastal Shallow Turquoise Shelf Halo
    ctx.beginPath();
    ctx.moveTo(canvasPoints[0][0], canvasPoints[0][1]);
    for (let i = 1; i < canvasPoints.length; i++) {
      ctx.lineTo(canvasPoints[i][0], canvasPoints[i][1]);
    }
    ctx.closePath();

    ctx.lineWidth = 16;
    ctx.strokeStyle = 'rgba(10, 110, 150, 0.35)';
    ctx.stroke();
    ctx.lineWidth = 6;
    ctx.strokeStyle = 'rgba(25, 160, 195, 0.45)';
    ctx.stroke();

    // Fill Land Base
    ctx.fillStyle = baseColor;
    ctx.fill();

    // Desert / Dry Biome Overlay
    if (desertColor) {
      ctx.fillStyle = desertColor;
      ctx.globalAlpha = 0.7;
      ctx.fill();
      ctx.globalAlpha = 1.0;
    }
  };

  // Africa
  drawGeographicLand(
    [[37, 10], [32, 32], [12, 44], [11, 51], [-12, 40], [-35, 20], [-34, 18], [5, 9], [15, -17], [36, -5]],
    '#214224', '#a68757'
  );
  // Sahara Desert Detail
  const [sx, sy] = latLonToCanvas(24, 14, width, height);
  const saharaGrad = ctx.createRadialGradient(sx, sy, 10, sx, sy, 180);
  saharaGrad.addColorStop(0, '#bd9b68');
  saharaGrad.addColorStop(0.7, '#a68552');
  saharaGrad.addColorStop(1, 'rgba(33, 66, 36, 0)');
  ctx.fillStyle = saharaGrad;
  ctx.beginPath();
  ctx.arc(sx, sy, 180, 0, Math.PI * 2);
  ctx.fill();

  // Europe
  drawGeographicLand(
    [[36, -9], [43, -9], [48, -4], [51, 2], [54, 8], [58, 6], [71, 25], [60, 30], [45, 35], [40, 26], [38, 24], [36, 14]],
    '#294a2e'
  );

  // Middle East & Arabia
  drawGeographicLand(
    [[30, 32], [30, 48], [25, 55], [12, 44], [15, 53], [22, 59], [30, 48]],
    '#ad8e5c', '#c29d66'
  );

  // India & South Asia (Prominent in reference image 1)
  drawGeographicLand(
    [[25, 62], [30, 70], [35, 76], [28, 88], [22, 89], [16, 82], [8, 77], [13, 74], [19, 72], [23, 68]],
    '#1e4522', '#3d5432'
  );
  // Western Ghats & Deccan Plateau Detail
  const [ix, iy] = latLonToCanvas(18, 77, width, height);
  const indiaGrad = ctx.createRadialGradient(ix, iy, 5, ix, iy, 90);
  indiaGrad.addColorStop(0, '#224a25');
  indiaGrad.addColorStop(0.8, '#1b3b1e');
  indiaGrad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = indiaGrad;
  ctx.beginPath();
  ctx.arc(ix, iy, 90, 0, Math.PI * 2);
  ctx.fill();

  // East Asia & Eurasia
  drawGeographicLand(
    [[70, 70], [70, 170], [60, 165], [50, 140], [40, 120], [30, 122], [22, 114], [10, 105], [10, 98], [20, 95], [28, 88]],
    '#264728', '#8c754d'
  );

  // SE Asia & Indonesia Islands
  drawGeographicLand([[-10, 114], [-12, 130], [-12, 142], [-25, 153], [-38, 145], [-32, 115], [-22, 114]], '#1b3b1e');

  // Australia
  drawGeographicLand(
    [[-12, 130], [-12, 142], [-25, 153], [-38, 145], [-32, 115], [-22, 114]],
    '#8f5633', '#a8663b'
  );

  // North America
  drawGeographicLand(
    [[70, -165], [70, -70], [45, -64], [25, -80], [15, -90], [30, -115], [48, -125], [60, -140]],
    '#2b4c2e', '#998057'
  );

  // South America
  drawGeographicLand(
    [[12, -72], [5, -52], [-10, -35], [-23, -42], [-55, -68], [-40, -73], [-18, -70], [0, -80]],
    '#184020'
  );

  // Himalayas & Alps Snow Peaks
  const [hx, hy] = latLonToCanvas(30, 84, width, height);
  ctx.fillStyle = '#e2e8f0';
  ctx.beginPath();
  ctx.ellipse(hx, hy, 55, 12, -0.1, 0, Math.PI * 2);
  ctx.fill();

  // Polar Ice Caps
  ctx.fillStyle = '#f1f5f9';
  ctx.fillRect(0, 0, width, 55); // North Pole
  ctx.fillRect(0, height - 65, width, 65); // South Pole

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// Generate High-Resolution Night City Lights Texture (2048x1024)
function generatePhotorealisticNightTexture() {
  const width = 2048;
  const height = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, width, height);

  const addCityLightNode = (lat, lon, radius, intensity = 1.0) => {
    const [cx, cy] = latLonToCanvas(lat, lon, width, height);
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    grad.addColorStop(0, `rgba(255, 220, 130, ${1.0 * intensity})`);
    grad.addColorStop(0.3, `rgba(255, 160, 40, ${0.75 * intensity})`);
    grad.addColorStop(0.7, `rgba(210, 100, 15, ${0.35 * intensity})`);
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();
  };

  // Dense Golden City Networks matching Reference Image 1 (India, East Asia, Middle East, Europe)
  // India (Dense golden cluster as in reference image)
  for (let i = 0; i < 350; i++) {
    const lat = 8 + Math.random() * 24;
    const lon = 68 + Math.random() * 22;
    addCityLightNode(lat, lon, Math.random() * 4 + 1.2, 1.25);
  }
  // Major Metros
  addCityLightNode(19, 72.8, 14, 1.6); // Mumbai
  addCityLightNode(28.6, 77.2, 16, 1.6); // Delhi
  addCityLightNode(12.9, 77.5, 12, 1.5); // Bengaluru
  addCityLightNode(22.5, 88.3, 13, 1.5); // Kolkata
  addCityLightNode(13.0, 80.2, 12, 1.5); // Chennai

  // East Asia & Japan
  for (let i = 0; i < 400; i++) {
    const lat = 20 + Math.random() * 25;
    const lon = 100 + Math.random() * 40;
    addCityLightNode(lat, lon, Math.random() * 3.5 + 1.0, 1.15);
  }
  addCityLightNode(35.6, 139.6, 18, 1.7); // Tokyo
  addCityLightNode(31.2, 121.4, 15, 1.5); // Shanghai

  // Middle East & Nile River
  for (let i = 0; i < 160; i++) {
    const lat = 15 + Math.random() * 20;
    const lon = 32 + Math.random() * 25;
    addCityLightNode(lat, lon, Math.random() * 3 + 1, 1.0);
  }
  // Nile River Ribbon
  for (let lat = 24; lat < 31; lat += 0.5) {
    addCityLightNode(lat, 31.2, 4.5, 1.35);
  }

  // Europe
  for (let i = 0; i < 350; i++) {
    const lat = 36 + Math.random() * 24;
    const lon = -9 + Math.random() * 45;
    addCityLightNode(lat, lon, Math.random() * 3.5 + 1.0, 1.1);
  }

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

// Generate Ocean Specular Texture (Oceans reflect sunlight, land is matte)
function generateSpecularTexture() {
  const width = 2048;
  const height = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = '#050505';

  const maskLand = (latLonPoints) => {
    const canvasPoints = latLonPoints.map(([lat, lon]) => latLonToCanvas(lat, lon, width, height));
    ctx.beginPath();
    ctx.moveTo(canvasPoints[0][0], canvasPoints[0][1]);
    for (let i = 1; i < canvasPoints.length; i++) ctx.lineTo(canvasPoints[i][0], canvasPoints[i][1]);
    ctx.closePath();
    ctx.fill();
  };

  maskLand([[37, 10], [32, 32], [12, 44], [-35, 20], [15, -17]]); // Africa
  maskLand([[36, -9], [54, 8], [71, 25], [45, 35]]); // Europe
  maskLand([[30, 32], [25, 55], [12, 44]]); // Middle East
  maskLand([[25, 62], [35, 76], [8, 77]]); // India
  maskLand([[70, 70], [70, 170], [22, 114]]); // Asia
  maskLand([[-12, 130], [-38, 145], [-22, 114]]); // Australia
  maskLand([[70, -165], [45, -64], [15, -90]]); // Americas

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

// Generate Realistic Photorealistic Cloud Cover Texture (2048x1024)
function generatePhotorealisticCloudTexture() {
  const width = 2048;
  const height = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = 'rgba(0,0,0,0)';
  ctx.clearRect(0, 0, width, height);

  // Swirling Equatorial & Storm Cloud Bands
  for (let i = 0; i < 480; i++) {
    const x = Math.random() * width;
    const y = Math.random() * height;
    const rx = Math.random() * 120 + 30;
    const ry = Math.random() * 28 + 8;
    const angle = (Math.random() - 0.5) * 0.4;

    const grad = ctx.createRadialGradient(x, y, 0, x, y, rx);
    grad.addColorStop(0, 'rgba(255, 255, 255, 0.72)');
    grad.addColorStop(0.45, 'rgba(240, 248, 255, 0.42)');
    grad.addColorStop(1, 'rgba(255, 255, 255, 0)');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, angle, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

export default function Earth({ earthRadius = 2.25 }) {
  const earthRef = useRef();
  const cloudsRef = useRef();
  const atmosphereMaterial = useMemo(() => createAtmosphereShader(), []);

  const dayTexture = useMemo(() => generatePhotorealisticDayTexture(), []);
  const nightTexture = useMemo(() => generatePhotorealisticNightTexture(), []);
  const specularTexture = useMemo(() => generateSpecularTexture(), []);
  const cloudTexture = useMemo(() => generatePhotorealisticCloudTexture(), []);

  useFrame((state, delta) => {
    // Continuous slow Y rotation matching reference image
    if (earthRef.current) {
      earthRef.current.rotation.y += delta * 0.025;
    }
    if (cloudsRef.current) {
      cloudsRef.current.rotation.y += delta * 0.032;
    }
  });

  return (
    // Positioned in right hero region matching reference composition (right 45-50%)
    <group position={[1.5, -0.15, 0]}>
      {/* Main Photorealistic Earth Sphere */}
      <mesh ref={earthRef} rotation={[0.22, 3.8, 0]}>
        <sphereGeometry args={[earthRadius, 96, 96]} />
        <meshStandardMaterial
          map={dayTexture}
          roughnessMap={specularTexture}
          roughness={0.45}
          metalness={0.1}
          emissiveMap={nightTexture}
          emissive={new THREE.Color('#ffaa33')}
          emissiveIntensity={2.3}
        />
      </mesh>

      {/* Realistic Swirling Cloud Layer */}
      <mesh ref={cloudsRef} rotation={[0.22, 3.8, 0]}>
        <sphereGeometry args={[earthRadius * 1.012, 96, 96]} />
        <meshStandardMaterial
          map={cloudTexture}
          transparent={true}
          opacity={0.42}
          blending={THREE.NormalBlending}
          depthWrite={false}
        />
      </mesh>

      {/* Atmospheric Scattering Fresnel Outer Glow */}
      <mesh scale={[1.08, 1.08, 1.08]}>
        <sphereGeometry args={[earthRadius, 96, 96]} />
        <primitive object={atmosphereMaterial} attach="material" />
      </mesh>
    </group>
  );
}
