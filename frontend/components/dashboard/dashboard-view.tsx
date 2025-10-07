"use client"

import { useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { Pie, PieChart, Cell, Line, LineChart, XAxis, YAxis, ResponsiveContainer } from "recharts"
import { cn } from "@/lib/utils"
import useSWR from "swr"
import { getDashboardData } from "@/lib/api"
import { useMarketSocket } from "@/hooks/use-market-socket"
import { Badge } from "@/components/ui/badge"

const fetcher = async () => getDashboardData()

export default function DashboardView() {
  const { data } = useSWR("dashboard", fetcher)
  useMarketSocket() // hook connects to /ws

  const marketOverview = data?.marketOverview ?? []
  const donutData = useMemo(() => data?.portfolioBreakdown ?? [], [data])
  const watchlist = data?.watchlist ?? []

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      {/* Market Overview Cards */}
      <div className="grid grid-cols-2 gap-4 lg:col-span-2 md:grid-cols-4">
        {marketOverview.map((item) => (
          <Card key={item.label} className="bg-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm text-muted-foreground">{item.label}</CardTitle>
            </CardHeader>
            <CardContent className="flex items-end justify-between">
              <div className="text-2xl font-semibold">{item.value}</div>
              <Badge
                variant={item.delta >= 0 ? "default" : "secondary"}
                className={cn(item.delta >= 0 ? "text-accent" : "text-destructive")}
              >
                {item.delta >= 0 ? "+" : ""}
                {item.delta.toFixed(2)}%
              </Badge>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Portfolio Donut */}
      <Card className="lg:col-span-1">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Portfolio Allocation</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartContainer
            config={{
              equity: { label: "Equity", color: "hsl(var(--chart-1))" },
              debt: { label: "Debt", color: "hsl(var(--chart-2))" },
              gold: { label: "Gold", color: "hsl(var(--chart-3))" },
              cash: { label: "Cash", color: "hsl(var(--chart-4))" },
            }}
            className="aspect-square"
          >
            <PieChart>
              <ChartTooltip content={<ChartTooltipContent hideLabel />} />
              <ChartLegend verticalAlign="bottom" content={<ChartLegendContent />} />
              <Pie data={donutData} dataKey="value" nameKey="name" innerRadius={55} outerRadius={80} strokeWidth={1}>
                {donutData.map((entry) => (
                  <Cell key={entry.name} fill={`var(--color-${entry.key})`} />
                ))}
              </Pie>
            </PieChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {/* Watchlist */}
      <Card className="lg:col-span-3">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Watchlist</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {watchlist.map((stock) => (
              <div key={stock.symbol} className="rounded-md border p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-sm font-medium">{stock.symbol}</div>
                  <div className={cn("text-xs", stock.change >= 0 ? "text-accent" : "text-destructive")}>
                    {stock.price.toFixed(2)} ({stock.change >= 0 ? "+" : ""}
                    {stock.change.toFixed(2)}%)
                  </div>
                </div>
                <div className="h-14">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={stock.spark}>
                      <XAxis dataKey="t" hide />
                      <YAxis domain={["dataMin", "dataMax"]} hide />
                      <Line
                        type="monotone"
                        dataKey="p"
                        stroke={stock.change >= 0 ? "hsl(var(--chart-2))" : "hsl(var(--destructive))"}
                        strokeWidth={1.5}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
