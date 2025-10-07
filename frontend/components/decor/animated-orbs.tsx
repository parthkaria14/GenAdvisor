"use client"

export function AnimatedOrbs() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-20 overflow-hidden" aria-hidden="true">
      <div className="orb orb-blue" />
      <div className="orb orb-pink" />
      <div className="orb orb-orange" />
    </div>
  )
}
