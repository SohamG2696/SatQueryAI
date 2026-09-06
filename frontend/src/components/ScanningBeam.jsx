import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export default function ScanningBeam({ satellitePos, targetPos }) {
  const beamGroupRef = useRef();
  const coneRef = useRef();
  const targetSpotRef = useRef();

  useFrame((state) => {
    const time = state.clock.getElapsedTime();

    if (beamGroupRef.current && satellitePos && targetPos) {
      const origin = new THREE.Vector3(...satellitePos);
      const target = new THREE.Vector3(...targetPos);
      
      const direction = new THREE.Vector3().subVectors(target, origin);
      const height = direction.length();

      const midpoint = new THREE.Vector3().addVectors(origin, target).multiplyScalar(0.5);
      beamGroupRef.current.position.copy(midpoint);

      beamGroupRef.current.lookAt(target);
      beamGroupRef.current.rotateX(Math.PI / 2);

      // Subtle, refined scanning pulse
      if (coneRef.current) {
        coneRef.current.material.opacity = 0.2 + Math.sin(time * 3.0) * 0.08;
      }

      if (targetSpotRef.current) {
        targetSpotRef.current.position.copy(target);
        targetSpotRef.current.lookAt(origin);
        const scale = 1.0 + Math.sin(time * 3.5) * 0.1;
        targetSpotRef.current.scale.set(scale, scale, scale);
      }
    }
  });

  const origin = new THREE.Vector3(...satellitePos);
  const target = new THREE.Vector3(...targetPos);
  const height = origin.distanceTo(target);

  return (
    <group>
      {/* Thin, Focused, Semi-Transparent Cyan Scanning Beam */}
      <group ref={beamGroupRef}>
        <mesh ref={coneRef}>
          {/* Narrow cone parameters: top radius 0.03, bottom radius 0.32 */}
          <cylinderGeometry args={[0.03, 0.32, height, 16, 1, true]} />
          <meshBasicMaterial
            color="#38bdf8"
            transparent={true}
            opacity={0.25}
            side={THREE.DoubleSide}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
        
        {/* Core Line Beam */}
        <mesh>
          <cylinderGeometry args={[0.008, 0.08, height, 8, 1, true]} />
          <meshBasicMaterial
            color="#ffffff"
            transparent={true}
            opacity={0.4}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
      </group>

      {/* Subtle Ground Contact Point on Earth Surface */}
      <group ref={targetSpotRef}>
        <mesh>
          <ringGeometry args={[0.04, 0.08, 24]} />
          <meshBasicMaterial
            color="#38bdf8"
            side={THREE.DoubleSide}
            transparent={true}
            opacity={0.7}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
        <mesh>
          <circleGeometry args={[0.025, 16]} />
          <meshBasicMaterial
            color="#ffffff"
            transparent={true}
            opacity={0.85}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      </group>
    </group>
  );
}
