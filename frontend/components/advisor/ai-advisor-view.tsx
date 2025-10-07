"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Spinner } from "@/components/ui/spinner"
import { useChatStore } from "@/stores/chat-store"
import { getAdvisorMockResponse, postQuery } from "@/lib/api"
import { AI3DAdvisor } from "@/components/advisor/ai-3d-advisor"

type Msg = { role: "user" | "assistant"; content: string }

export default function AIAdvisorView() {
  const { messages, append } = useChatStore()
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  async function onSend() {
    if (!input.trim()) return
    const userMsg: Msg = { role: "user", content: input.trim() }
    append(userMsg)
    setInput("")
    setLoading(true)
    try {
      // Prefer real endpoint
      const res = await postQuery(userMsg.content, false)
      const text = typeof res === "string" ? res : JSON.stringify(res)
      append({ role: "assistant", content: text })
    } catch {
      // Fallback to mock
      const assistant = await getAdvisorMockResponse(userMsg.content)
      append({ role: "assistant", content: assistant })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative grid gap-4">
      {/* 3D decorative background scoped to this view */}
      <AI3DAdvisor fixed={false} />
      <Card className="card-glass glass-hover">
        <CardHeader className="pb-2">
          {/* Gradient headline for stronger visual hierarchy */}
          <CardTitle className="text-lg font-semibold brand-gradient-text">AI Advisor</CardTitle>
        </CardHeader>
        <CardContent>
          {/* Slightly more transparent to let the 3D peek through subtly */}
          <div className="min-h-[65svh] sm:min-h-[70svh] overflow-y-auto border rounded-md p-3 space-y-3 bg-background/50 backdrop-blur-[1px]">
            {messages.map((m, i) => (
              <div key={i} className="text-sm leading-relaxed" aria-live="polite">
                <span className="font-medium">{m.role === "user" ? "You" : "Advisor"}: </span>
                <span className="text-pretty">{m.content}</span>
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-2 text-muted-foreground text-sm">
                <Spinner className="size-4" />
                Thinking...
              </div>
            )}
            <div ref={bottomRef} />
          </div>
          <div className="mt-3 flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about Indian markets, portfolios, or strategies..."
              className="resize-none h-20 glass"
              aria-label="Message"
            />
            <Button onClick={onSend} disabled={loading}>
              Send
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
