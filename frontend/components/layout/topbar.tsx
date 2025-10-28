"use client"

import Link from "next/link"
import { useState, useEffect } from "react"
import { useTheme } from "next-themes"
import { Search, Sun, Moon } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

export default function Topbar() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)
  const [query, setQuery] = useState("")
  
  // Only show theme toggle after mounting to avoid hydration mismatch
  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <header className="sticky top-0 z-40 w-full topbar-glass">
      <div className="container-fluid h-16 flex items-center gap-3">
        <Link href="/app" className="font-semibold tracking-tight">
          <span className="brand-gradient-text text-lg">GenAdvisor</span>
        </Link>

        {/* Mobile search trigger */}
        <Button variant="secondary" size="sm" className="sm:hidden w-8 h-8 px-0" aria-label="Open search">
          <Search className="h-5 w-5" />
        </Button>

        {/* Full search from sm and up */}
        <form
          className="hidden sm:flex items-center gap-2 flex-1"
          onSubmit={(e) => {
            e.preventDefault()
            console.log("[v0] Search:", query)
          }}
        >
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search stocks, indices, news..."
              className="pl-9 w-full"
            />
          </div>
          <Button type="submit" className="neon-blue">
            Search
          </Button>
        </form>

        <Button
          variant="secondary"
          size="sm"
          aria-label="Toggle theme"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          className="w-8 h-8 px-0"
        >
          {mounted ? (
            theme === "dark" ? (
              <Sun className="h-5 w-5" />
            ) : (
              <Moon className="h-5 w-5" />
            )
          ) : (
            <div className="h-5 w-5" /> // Placeholder of same size while loading
          )}
        </Button>

        <Avatar className="h-8 w-8">
          <AvatarImage alt="You" />
          <AvatarFallback>U</AvatarFallback>
        </Avatar>
      </div>
    </header>
  )
}
