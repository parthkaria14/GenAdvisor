"use client"

import { Canvas } from "@react-three/fiber"
import { Environment, Float, OrbitControls } from "@react-three/drei"

type AdvisorProps = {
  // when true, fills viewport with fixed positioning (old behavior). When false, fills parent (absolute).
  fixed?: boolean
}

function ThinkingCore({ color = "#7c3aed" }: { color?: string }) {
  return (
    <Float speed={2} rotationIntensity={0.8} floatIntensity={1.2}>
      <mesh castShadow receiveShadow>
        <icosahedronGeometry args={[1.2, 3]} />
        <meshPhysicalMaterial
          color={color}
          transmission={0.92}
          roughness={0.12}
          thickness={0.6}
          metalness={0.1}
          clearcoat={1}
          clearcoatRoughness={0.05}
        />
      </mesh>
    </Float>
  )
}

function OrbitRings() {
  return (
    <>
      <mesh rotation={[Math.PI / 2, 0, 0.5]}>
        <torusGeometry args={[2.2, 0.025, 16, 200]} />
        <meshStandardMaterial color="#60a5fa" emissive="#3b82f6" emissiveIntensity={0.25} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, -0.35]}>
        <torusGeometry args={[2.5, 0.02, 16, 220]} />
        <meshStandardMaterial color="#ec4899" emissive="#db2777" emissiveIntensity={0.2} />
      </mesh>
    </>
  )
}

export function AI3DAdvisor({ fixed = false }: AdvisorProps) {
  return (
    <div
      className={fixed ? "pointer-events-none fixed inset-0 -z-10" : "pointer-events-none absolute inset-0 -z-10"}
      aria-hidden="true"
    >
      <Canvas camera={{ position: [0, 0, 6], fov: 55 }} shadows>
        <color attach="background" args={["transparent"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[6, 6, 6]} intensity={0.7} castShadow />
        <ThinkingCore color="#7c3aed" />
        <OrbitRings />
        <Environment preset="studio" />
        <OrbitControls enableZoom={false} enablePan={false} enableRotate={false} />
      </Canvas>
    </div>
  )
}
