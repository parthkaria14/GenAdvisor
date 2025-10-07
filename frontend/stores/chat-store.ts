"use client"

import { create } from "zustand"

type Msg = { role: "user" | "assistant"; content: string }

type ChatState = {
  messages: Msg[]
  append: (m: Msg) => void
}

export const useChatStore = create<ChatState>()((set) => ({
  messages: [
    { role: "assistant", content: "Hi! I am your AI Advisor. Ask me about Indian markets, portfolios, or any stock." },
  ],
  append: (m) => set((s) => ({ messages: [...s.messages, m] })),
}))
