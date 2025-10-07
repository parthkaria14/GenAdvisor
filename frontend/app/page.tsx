"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ThemeToggle } from "@/components/theme-toggle"
import { LineChart, Shield, Sparkles } from "lucide-react"

export default function LandingPage() {
  return (
    <div data-landing-root="true" className="min-h-svh text-foreground bg-transparent">
      {/* Header */}
      <header className="border-b">
        <div className="mx-auto max-w-6xl px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="size-8 rounded bg-primary/10 flex items-center justify-center">
              <span className="text-primary text-sm font-bold">GA</span>
            </div>
            <span className="text-base font-semibold">GenAdvisor</span>
          </div>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Link href="/app">
              <Button variant="default">Open App</Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <main className="mx-auto max-w-6xl px-4 py-12 md:py-16">
        <section className="grid items-center gap-8 md:grid-cols-2">
          <div className="space-y-6">
            <h1 className="font-serif text-balance text-4xl md:text-5xl lg:text-6xl leading-tight">
              <span className="text-brand-gradient">Do more with your money.</span>
            </h1>
            <p className="text-muted-foreground text-pretty text-base md:text-lg leading-relaxed">
              AI-guided insights, real-time dashboards, and powerful screeners for India’s markets—built for clarity,
              speed, and confident decisions.
            </p>
            <div className="flex flex-col sm:flex-row gap-3">
              <Link href="/app">
                <Button size="lg" className="w-full sm:w-auto">
                  Launch GenAdvisor
                </Button>
              </Link>
              <a
                href="#features"
                className="w-full sm:w-auto inline-flex items-center justify-center text-sm font-medium underline-offset-4 hover:underline"
              >
                Explore features
              </a>
            </div>
          </div>

          {/* Visual */}
          <div className="rounded-2xl border card-glass">
            <div className="aspect-[16/10] w-full rounded-2xl bg-brand-gradient p-1" aria-hidden="true">
              <img
                src="/images/design-inspiration.png"
                alt="Vibrant finance design inspiration"
                className="h-full w-full rounded-xl object-cover"
              />
            </div>
          </div>
        </section>

        {/* Features */}
        <section id="features" className="mt-16 md:mt-20 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <LineChart className="size-5 text-primary" />
                Market Dashboard
              </CardTitle>
            </CardHeader>
            <CardContent className="text-muted-foreground">
              Live watchlists, sector performance, and overview cards with compact charts to monitor breadth and
              momentum at a glance.
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="size-5 text-primary" />
                AI Advisor
              </CardTitle>
            </CardHeader>
            <CardContent className="text-muted-foreground">
              Chat with an AI that understands Indian equities. Ask for analyses, risk, and portfolio suggestions with
              clear, well-formatted answers.
            </CardContent>
          </Card>

          <Card className="sm:col-span-2 lg:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="size-5 text-primary" />
                Portfolio & Risk
              </CardTitle>
            </CardHeader>
            <CardContent className="text-muted-foreground">
              Optimize allocations, visualize exposure, and get risk insights—then act confidently with data-backed
              guidance.
            </CardContent>
          </Card>
        </section>

        {/* CTA */}
        <section className="mt-12 md:mt-16">
          <div className="rounded-lg border bg-card p-6 md:p-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h2 className="font-serif text-2xl md:text-3xl">Ready to invest smarter?</h2>
              <p className="text-muted-foreground mt-1">
                Join traders and investors who rely on GenAdvisor for clarity and speed.
              </p>
            </div>
            <Link href="/app">
              <Button size="lg">Open the App</Button>
            </Link>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t">
        <div className="mx-auto max-w-6xl px-4 py-6 text-xs text-muted-foreground">
          © {new Date().getFullYear()} GenAdvisor. For informational purposes only. Not investment advice.
        </div>
      </footer>
    </div>
  )
}
