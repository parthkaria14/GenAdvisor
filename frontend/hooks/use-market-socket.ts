"use client"

import { useEffect } from "react"

type UpdateHandler = (payload: any) => void

export function useMarketSocket(onUpdate?: UpdateHandler) {
  useEffect(() => {
    let isActive = true
    try {
      const scheme = typeof window !== "undefined" && window.location.protocol === "https:" ? "wss" : "ws"
      const base =
        (typeof window !== "undefined" && (window as any).NEXT_PUBLIC_API_BASE_URL) ||
        process.env.NEXT_PUBLIC_API_BASE_URL ||
        ""
      const url = base ? `${base.replace(/^http/, "ws")}/ws` : `${scheme}://${window.location.host}/ws`

      const ws = new WebSocket(url)
      ws.onmessage = (e) => {
        if (!isActive) return
        try {
          const data = JSON.parse(e.data)
          if (onUpdate) onUpdate(data)
        } catch {
          // ignore non-JSON
        }
      }
      ws.onerror = () => {
        // silent
      }
      return () => {
        isActive = false
        ws.close()
      }
    } catch {
      return () => {
        isActive = false
      }
    }
  }, [onUpdate])
}



