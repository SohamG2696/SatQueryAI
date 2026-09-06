import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import ScanningBeam from './ScanningBeam';

// Solar Panel Cell Grid Texture
function generateSolarPanelTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#0a1d33';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2;

  const cols = 8;
  const rows = 4;
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;

  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      ctx.strokeRect(c * cellW + 2, r * cellH + 2, cellW - 4, cellH - 4);
      ctx.fillStyle = '#0f2f52';
      ctx.fillRect(c * cellW + 4, r * cellH + 4, cellW - 8, cellH - 8);
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

export default function Satellite({ earthCenter = [1.55, -0.15, 0] }) {
  const satGroupRef = useRef();
  const satMeshRef = useRef();

  const solarTexture = useMemo(() => generateSolarPanelTexture(), []);

  const [satPos, setSatPos] = React.useState([-0.25, 1.65, 1.2]);
  const [targetPos, setTargetPos] = React.useState([1.2, 0.4, 1.8]);

  useFrame((state) => {
    const t = state.clock.getElapsedTime() * 0.12; // Slow, calm cinematic orbit speed

    const radiusX = 2.75;
    const radiusZ = 2.3;
    const heightY = 1.58;

    const angle = t + 0.85; // Starting position matching reference image
    const x = earthCenter[0] + Math.cos(angle) * radiusX - 1.25;
    const y = earthCenter[1] + heightY + Math.sin(t * 0.4) * 0.12;
    const z = earthCenter[2] + Math.sin(angle) * radiusZ;

    const newSatPos = [x, y, z];
    setSatPos(newSatPos);

    if (satGroupRef.current) {
      satGroupRef.current.position.set(x, y, z);

      // Orient satellite dish/sensor towards Earth center
      const earthVec = new THREE.Vector3(...earthCenter);
      satGroupRef.current.lookAt(earthVec);
      satGroupRef.current.rotateZ(0.12);

      // Earth surface target point for observation beam
      const dir = new THREE.Vector3().subVectors(earthVec, new THREE.Vector3(x, y, z)).normalize();
      const earthSurfacePoint = new THREE.Vector3(...earthCenter).add(dir.multiplyScalar(2.2));
      setTargetPos([earthSurfacePoint.x, earthSurfacePoint.y, earthSurfacePoint.z]);
    }
  });

  return (
    <group>
      <group ref={satGroupRef} position={satPos}>
        {/* Scaled small relative to Earth (~10% of Earth diameter) */}
        <group ref={satMeshRef} scale={[0.26, 0.26, 0.26]}>
          
          {/* Main Body Chassis - Gold Thermal Foil Wrapped Octagonal Payload Bus */}
          <mesh position={[0, 0, 0]}>
            <cylinderGeometry args={[0.5, 0.5, 0.75, 8]} />
            <meshStandardMaterial
              color="#e6b800"
              metalness={0.9}
              roughness={0.2}
              emissive="#261a00"
              emissiveIntensity={0.15}
            />
          </mesh>

          {/* Top & Bottom Gold Insulation Covers */}
          <mesh position={[0, 0.4, 0]}>
            <boxGeometry args={[0.68, 0.08, 0.68]} />
            <meshStandardMaterial color="#d4af37" metalness={0.92} roughness={0.18} />
          </mesh>
          <mesh position={[0, -0.4, 0]}>
            <boxGeometry args={[0.68, 0.08, 0.68]} />
            <meshStandardMaterial color="#d4af37" metalness={0.92} roughness={0.18} />
          </mesh>

          {/* Left Solar Panel Array Wing */}
          <group position={[-1.65, 0, 0]}>
            <mesh position={[0, 0, 0]}>
              <boxGeometry args={[2.1, 0.7, 0.04]} />
              <meshStandardMaterial
                map={solarTexture}
                metalness={0.75}
                roughness={0.25}
              />
            </mesh>
            {/* Structural Truss Arm */}
            <mesh position={[1.15, 0, 0]}>
              <cylinderGeometry args={[0.035, 0.035, 0.35, 8]} rotation={[0, 0, Math.PI / 2]} />
              <meshStandardMaterial color="#94a3b8" metalness={0.95} roughness={0.15} />
            </mesh>
          </group>

          {/* Right Solar Panel Array Wing */}
          <group position={[1.65, 0, 0]}>
            <mesh position={[0, 0, 0]}>
              <boxGeometry args={[2.1, 0.7, 0.04]} />
              <meshStandardMaterial
                map={solarTexture}
                metalness={0.75}
                roughness={0.25}
              />
            </mesh>
            {/* Structural Truss Arm */}
            <mesh position={[-1.15, 0, 0]}>
              <cylinderGeometry args={[0.035, 0.035, 0.35, 8]} rotation={[0, 0, Math.PI / 2]} />
              <meshStandardMaterial color="#94a3b8" metalness={0.95} roughness={0.15} />
            </mesh>
          </group>

          {/* Parabolic Dish Antenna facing Earth */}
          <group position={[0, -0.22, 0.52]} rotation={[Math.PI / 4, 0, 0]}>
            <mesh>
              <sphereGeometry args={[0.38, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.35]} />
              <meshStandardMaterial
                color="#f8fafc"
                metalness={0.95}
                roughness={0.12}
                side={THREE.DoubleSide}
              />
            </mesh>
            {/* Feed Horn Assembly */}
            <mesh position={[0, 0, 0.22]}>
              <cylinderGeometry args={[0.02, 0.02, 0.28, 8]} rotation={[Math.PI / 2, 0, 0]} />
              <meshStandardMaterial color="#334155" metalness={0.9} />
            </mesh>
          </group>

          {/* Earth Observation Optical Sensor Lens Unit */}
          <mesh position={[0, -0.42, 0.18]}>
            <cylinderGeometry args={[0.14, 0.16, 0.22, 16]} />
            <meshStandardMaterial color="#0f172a" metalness={0.95} roughness={0.08} />
          </mesh>

        </group>
      </group>

      {/* Observation Beam */}
      <ScanningBeam satellitePos={satPos} targetPos={targetPos} />
    </group>
  );
}
