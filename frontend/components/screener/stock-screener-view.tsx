"use client"

import { useMemo, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { getScreenerMock, postScreener, getPricePrediction } from "@/lib/api"
import { PredictionDialog } from "./prediction-dialog"

type Row = {
  symbol: string
  sector: string
  marketCap: "Small" | "Mid" | "Large"
  pe: number
  price: number
}

export default function StockScreenerView() {
  const base = useMemo(() => getScreenerMock(), [])
  const [cap, setCap] = useState<"All" | Row["marketCap"]>("All")
  const [sector, setSector] = useState<"All" | string>("All")
  const [maxPE, setMaxPE] = useState<string>("")
  const [rows, setRows] = useState<Row[]>(base)
  const [loading, setLoading] = useState<boolean>(false)
  const [selectedPrediction, setSelectedPrediction] = useState<{ symbol: string; price: number; predicted: number | null } | null>(null)
  const [market, setMarket] = useState<"NSE" | "BSE">("NSE")

  const sectors = useMemo(() => Array.from(new Set(base.map((r) => r.sector))).sort(), [base])

  async function onSearch() {
    setLoading(true)
    try {
      const payload: any = {}
      if (cap !== "All") {
        // Only a coarse mapping is possible without exact caps; leaving market cap range unset for "All"
        if (cap === "Large") payload.market_cap_min = 100000000000 // nominal threshold
        if (cap === "Small") payload.market_cap_max = 50000000000
      }
      if (sector !== "All") payload.sector = sector
      if (maxPE) payload.pe_max = Number(maxPE)
      payload.include_predictions = false // Never include predictions in screener
      const res = await postScreener(payload)
      console.log("Screener response:", res)
      const mapped: Row[] = res.stocks.map((s) => {
        return {
          symbol: s.symbol,
          sector: s.sector || "Unknown",
          marketCap: (s.market_cap ?? 0) > 1e11 ? "Large" : (s.market_cap ?? 0) > 5e10 ? "Mid" : "Small",
          pe: s.pe_ratio ?? 0,
          price: s.price ?? 0,
        }
      })
      setRows(mapped)
    } catch (error) {
      console.error("Screener error:", error)
      // fallback to local filter on base
      const filtered = base.filter((r) => {
        if (cap !== "All" && r.marketCap !== cap) return false
        if (sector !== "All" && r.sector !== sector) return false
        if (maxPE && r.pe > Number(maxPE)) return false
        return true
      })
      setRows(filtered)
    } finally {
      setLoading(false)
    }
  }

  async function handlePredictPrice(symbol: string, price: number) {
    try {
      // Normalize symbol format for API
      const normalizedSymbol = symbol.includes('.') ? symbol : `${symbol}.NS`
      
      // Get prediction
      const prediction = await getPricePrediction(normalizedSymbol, 5)
      
      // Open dialog with prediction
      setSelectedPrediction({
        symbol: symbol,
        price: price,
        predicted: prediction.predicted_next_price
      })
    } catch (error) {
      console.error("Error getting prediction:", error)
      // Still open dialog with null prediction so user sees error
      setSelectedPrediction({
        symbol: symbol,
        price: price,
        predicted: null
      })
    }
  }

  return (
    <div className="grid gap-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Filters</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-5">
          <div className="grid gap-1.5">
            <label className="text-xs text-muted-foreground">Market</label>
            <Select value={market} onValueChange={(v) => setMarket(v as "NSE" | "BSE")}>
              <SelectTrigger>
                <SelectValue placeholder="NSE" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="NSE">NSE</SelectItem>
                <SelectItem value="BSE">BSE</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="grid gap-1.5">
            <label className="text-xs text-muted-foreground">Market Cap</label>
            <Select value={cap} onValueChange={(v) => setCap(v as any)}>
              <SelectTrigger>
                <SelectValue placeholder="All" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="All">All</SelectItem>
                <SelectItem value="Large">Large</SelectItem>
                <SelectItem value="Mid">Mid</SelectItem>
                <SelectItem value="Small">Small</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="grid gap-1.5">
            <label className="text-xs text-muted-foreground">Sector</label>
            <Select value={sector} onValueChange={(v) => setSector(v)}>
              <SelectTrigger>
                <SelectValue placeholder="All" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="All">All</SelectItem>
                {sectors.map((s) => (
                  <SelectItem key={s} value={s}>
                    {s}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="grid gap-1.5">
            <label className="text-xs text-muted-foreground">Max P/E</label>
            <Input
              value={maxPE}
              onChange={(e) => setMaxPE(e.target.value)}
              placeholder="e.g., 30"
              inputMode="numeric"
            />
          </div>
          <div className="flex items-end gap-2">
            <Button type="button" onClick={onSearch} disabled={loading}>
              {loading ? "Loading..." : "Search"}
            </Button>
            <Button
              type="button"
              variant="secondary"
              onClick={() => {
                setCap("All")
                setSector("All")
                setMaxPE("")
                setRows(base)
              }}
              disabled={loading}
            >
              Reset
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Results</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto min-h-[60svh]">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Sector</TableHead>
                <TableHead className="text-right">MCap</TableHead>
                <TableHead className="text-right">P/E</TableHead>
                <TableHead className="text-right">Current Price</TableHead>
                <TableHead className="text-center">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r) => {
                return (
                  <TableRow key={r.symbol}>
                    <TableCell className="font-medium">{r.symbol}</TableCell>
                    <TableCell>{r.sector}</TableCell>
                    <TableCell className="text-right">{r.marketCap}</TableCell>
                    <TableCell className="text-right">{r.pe.toFixed(1)}</TableCell>
                    <TableCell className="text-right">{r.price.toFixed(2)}</TableCell>
                    <TableCell className="text-center">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handlePredictPrice(r.symbol, r.price)}
                      >
                        Predict Price
                      </Button>
                    </TableCell>
                  </TableRow>
                )
              })}
              {rows.length === 0 && (
                <TableRow>
                  <TableCell
                    colSpan={6}
                    className="text-center text-sm text-muted-foreground"
                  >
                    No results. Adjust filters.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {selectedPrediction && (
        <PredictionDialog
          symbol={selectedPrediction.symbol}
          currentPrice={selectedPrediction.price}
          predictedPrice={selectedPrediction.predicted}
          open={!!selectedPrediction}
          onOpenChange={(open) => !open && setSelectedPrediction(null)}
        />
      )}
    </div>
  )
}
