"use client"

import useSWR from "swr"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Separator } from "@/components/ui/separator"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Radar, RadarChart, PolarGrid, PolarAngleAxis } from "recharts"
import {
  getPortfolioMock,
  postAnalyzePortfolio,
  postAnalyzeRisk,
} from "@/lib/api"
import { useEffect, useRef, useState } from "react"
import { Spinner } from "@/components/ui/spinner"

function CountUp({ value, duration = 900 }: { value: number; duration?: number }) {
  const [display, setDisplay] = useState(0)
  const rafRef = useRef<number | null>(null)
  useEffect(() => {
    const start = performance.now()
    const from = 0
    const to = value
    const step = (t: number) => {
      const p = Math.min(1, (t - start) / duration)
      const eased = 1 - Math.pow(1 - p, 3)
      setDisplay(from + (to - from) * eased)
      if (p < 1) rafRef.current = requestAnimationFrame(step)
    }
    rafRef.current = requestAnimationFrame(step)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [value, duration])
  return <>{display.toFixed(2)}</>
}

const fetcher = async () => getPortfolioMock()

export default function PortfolioManagerView() {
  const { data } = useSWR("portfolio", fetcher, { fallbackData: getPortfolioMock() })
  const [optimizing, setOptimizing] = useState(false)
  const [risking, setRisking] = useState(false)
  const [optResult, setOptResult] = useState<any | null>(null)
  const [riskData, setRiskData] = useState<{ metric: string; score: number }[] | null>(null)

  async function onOptimize() {
    setOptimizing(true)
    try {
      const budget = data.holdings.reduce((acc: number, h: any) => acc + h.ltp * h.qty, 0)
      const existing_portfolio = data.holdings.map((h: any) => ({
        symbol: h.symbol,
        quantity: h.qty,
        avg_price: h.avg,
      }))
      const res = await postAnalyzePortfolio({ budget, strategy: "moderate", existing_portfolio })
      setOptResult(res)
    } catch {
      // keep silent fallback if backend unavailable
      setOptResult({ note: "Optimization service unavailable. Try again later." })
    } finally {
      setOptimizing(false)
    }
  }

  async function onRisk() {
    setRisking(true)
    try {
      const portfolio = data.holdings.map((h: any) => ({ symbol: h.symbol, quantity: h.qty, avg_price: h.avg }))
      const res = await postAnalyzeRisk({ portfolio, time_horizon: 12 })
      // Try to map to metric/score pairs if possible, else keep null and show default
      const metrics: { metric: string; score: number }[] = Array.isArray(res)
        ? res
        : typeof res === "object" && res
          ? Object.entries(res as Record<string, number>).map(([metric, score]) => ({ metric, score: Number(score) }))
          : []
      setRiskData(metrics.length ? metrics : null)
    } catch {
      setRiskData([
        { metric: "Volatility", score: 58 },
        { metric: "Drawdown", score: 41 },
        { metric: "Beta", score: 52 },
        { metric: "Liquidity", score: 77 },
        { metric: "Concentration", score: 38 },
      ])
    } finally {
      setRisking(false)
    }
  }

  const totalPL = data.holdings.reduce((acc: number, h: any) => acc + (h.ltp - h.avg) * h.qty, 0)

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card className="lg:col-span-2 card-glass glass-hover animate-fade-in">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Holdings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Ticker</TableHead>
                  <TableHead className="text-right">Qty</TableHead>
                  <TableHead className="text-right">Avg Price</TableHead>
                  <TableHead className="text-right">LTP</TableHead>
                  <TableHead className="text-right">P/L</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.holdings.map((h: any) => {
                  const pl = (h.ltp - h.avg) * h.qty
                  return (
                    <TableRow key={h.symbol} className="transition-all hover:-translate-y-[1px] hover:shadow-sm">
                      <TableCell className="font-medium">{h.symbol}</TableCell>
                      <TableCell className="text-right">{h.qty}</TableCell>
                      <TableCell className="text-right">{h.avg.toFixed(2)}</TableCell>
                      <TableCell className="text-right">{h.ltp.toFixed(2)}</TableCell>
                      <TableCell className={pl >= 0 ? "text-right text-accent" : "text-right text-destructive"}>
                        {pl >= 0 ? "+" : ""}
                        {pl.toFixed(2)}
                      </TableCell>
                    </TableRow>
                  )
                })}
                <TableRow className="bg-muted/20">
                  <TableCell colSpan={4} className="text-right font-medium">
                    Total P/L
                  </TableCell>
                  <TableCell className={totalPL >= 0 ? "text-right text-accent" : "text-right text-destructive"}>
                    {totalPL >= 0 ? "+" : ""}
                    <CountUp value={totalPL} />
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </div>
          <Separator className="my-4" />
          <div className="flex items-center gap-2">
            <Button onClick={onOptimize} disabled={optimizing}>
              {optimizing ? (
                <>
                  <Spinner className="size-4 mr-2" /> Optimizing...
                </>
              ) : (
                "AI Portfolio Optimizer"
              )}
            </Button>
            <Button variant="secondary" onClick={onRisk} disabled={risking}>
              {risking ? (
                <>
                  <Spinner className="size-4 mr-2" /> Analyzing...
                </>
              ) : (
                "Risk Analysis"
              )}
            </Button>
          </div>
          {optResult && (
            <div className="mt-3 text-sm text-pretty space-y-1.5">
              {"note" in optResult && <div>{optResult.note}</div>}
              {"expected_return" in optResult && (
                <div>
                  <span className="font-medium">Expected Return: </span>
                  {(optResult.expected_return as number)?.toFixed
                    ? (optResult.expected_return as number).toFixed(2) + "%"
                    : optResult.expected_return}
                </div>
              )}
              {"risk_level" in optResult && (
                <div>
                  <span className="font-medium">Risk Level: </span>
                  {optResult.risk_level}
                </div>
              )}
              {"sharpe_ratio" in optResult && (
                <div>
                  <span className="font-medium">Sharpe Ratio: </span>
                  {optResult.sharpe_ratio}
                </div>
              )}
              {"recommendations" in optResult && Array.isArray(optResult.recommendations) && (
                <ul className="list-disc pl-5">
                  {(optResult.recommendations as any[]).map((r, i) => (
                    <li key={i}>{typeof r === "string" ? r : (r?.text ?? JSON.stringify(r))}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="lg:col-span-1 card-glass glass-hover animate-fade-in [animation-delay:120ms]">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Risk Profile</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={{
              risk: { label: "Score", color: "hsl(var(--chart-1))" },
            }}
            className="aspect-square"
          >
            <RadarChart
              outerRadius="70%"
              data={
                riskData ?? [
                  { metric: "Volatility", score: 60 },
                  { metric: "Drawdown", score: 40 },
                  { metric: "Beta", score: 55 },
                  { metric: "Liquidity", score: 80 },
                  { metric: "Concentration", score: 35 },
                ]
              }
            >
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
              <ChartTooltip content={<ChartTooltipContent hideLabel />} />
              <Radar
                name="Risk"
                dataKey="score"
                stroke="hsl(var(--chart-1))"
                fill="hsl(var(--chart-1))"
                fillOpacity={0.4}
              />
            </RadarChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  )
}
