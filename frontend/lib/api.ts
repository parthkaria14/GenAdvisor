export function getDashboardMock() {
  return {
    marketOverview: [
      { label: "NIFTY 50", value: "24,075", delta: 0.62 },
      { label: "SENSEX", value: "79,645", delta: 0.45 },
      { label: "USD/INR", value: "83.11", delta: -0.08 },
      { label: "Crude", value: "72.4", delta: 1.12 },
    ],
    portfolioBreakdown: [
      { name: "Equity", key: "equity", value: 55 },
      { name: "Debt", key: "debt", value: 20 },
      { name: "Gold", key: "gold", value: 15 },
      { name: "Cash", key: "cash", value: 10 },
    ],
    watchlist: [
      {
        symbol: "RELIANCE",
        price: 2893.25,
        change: 0.85,
        spark: Array.from({ length: 24 }, (_, i) => ({ t: i, p: 2800 + Math.sin(i / 2) * 25 + i })),
      },
      {
        symbol: "TCS",
        price: 3998.8,
        change: -0.42,
        spark: Array.from({ length: 24 }, (_, i) => ({ t: i, p: 4050 - Math.cos(i / 2) * 20 - i })),
      },
      {
        symbol: "HDFCBANK",
        price: 1642.5,
        change: 0.21,
        spark: Array.from({ length: 24 }, (_, i) => ({ t: i, p: 1600 + Math.sin(i / 3) * 12 })),
      },
      {
        symbol: "INFY",
        price: 1620.1,
        change: 1.1,
        spark: Array.from({ length: 24 }, (_, i) => ({ t: i, p: 1580 + Math.sin(i / 2.8) * 10 + i * 0.5 })),
      },
      {
        symbol: "ITC",
        price: 456.7,
        change: -0.65,
        spark: Array.from({ length: 24 }, (_, i) => ({ t: i, p: 465 - Math.sin(i / 1.5) * 5 })),
      },
      {
        symbol: "LT",
        price: 3732.0,
        change: 0.35,
        spark: Array.from({ length: 24 }, (_, i) => ({ t: i, p: 3700 + Math.cos(i / 2.2) * 15 })),
      },
    ],
  }
}

export function getPortfolioMock() {
  return {
    holdings: [
      { symbol: "RELIANCE", qty: 15, avg: 2550.0, ltp: 2893.25 },
      { symbol: "TCS", qty: 8, avg: 3600.0, ltp: 3998.8 },
      { symbol: "HDFCBANK", qty: 20, avg: 1520.0, ltp: 1642.5 },
      { symbol: "INFY", qty: 12, avg: 1500.0, ltp: 1620.1 },
      { symbol: "ITC", qty: 100, avg: 420.0, ltp: 456.7 },
    ],
  }
}

export async function runPortfolioOptimizationMock(
  _holdings: Array<{ symbol: string; qty: number; avg: number; ltp: number }>,
) {
  await wait(800)
  return {
    note: "Increase allocation to large-cap IT by 2-3% and reduce concentration in single-stock positions to lower idiosyncratic risk.",
  }
}

export async function runRiskAnalysisMock(_holdings: Array<{ symbol: string; qty: number; avg: number; ltp: number }>) {
  await wait(700)
  return [
    { metric: "Volatility", score: 58 },
    { metric: "Drawdown", score: 41 },
    { metric: "Beta", score: 52 },
    { metric: "Liquidity", score: 77 },
    { metric: "Concentration", score: 38 },
  ]
}

export async function getAdvisorMockResponse(prompt: string) {
  await wait(600)
  return `Based on your query "${prompt}", NIFTY breadth looks stable. Consider a staggered approach into high-quality large caps and maintain 10-15% cash for tactical entries.`
}

export function getScreenerMock() {
  return [
    { symbol: "RELIANCE", sector: "Energy", marketCap: "Large", pe: 28.3, price: 2893.25 },
    { symbol: "TCS", sector: "IT", marketCap: "Large", pe: 30.5, price: 3998.8 },
    { symbol: "HDFCBANK", sector: "Financials", marketCap: "Large", pe: 22.1, price: 1642.5 },
    { symbol: "INFY", sector: "IT", marketCap: "Large", pe: 26.7, price: 1620.1 },
    { symbol: "ITC", sector: "FMCG", marketCap: "Large", pe: 28.9, price: 456.7 },
    { symbol: "LT", sector: "Industrials", marketCap: "Large", pe: 32.2, price: 3732.0 },
    { symbol: "PIDILITIND", sector: "Chemicals", marketCap: "Large", pe: 52.3, price: 2665.0 },
    { symbol: "POLYCAB", sector: "Industrials", marketCap: "Mid", pe: 35.2, price: 5360.0 },
    { symbol: "DEEPAKNTR", sector: "Chemicals", marketCap: "Mid", pe: 24.3, price: 2345.5 },
    { symbol: "COFORGE", sector: "IT", marketCap: "Mid", pe: 26.1, price: 6050.0 },
    { symbol: "INDIGO", sector: "Airlines", marketCap: "Large", pe: 18.7, price: 4430.0 },
    { symbol: "CAMS", sector: "Financials", marketCap: "Mid", pe: 45.5, price: 3600.0 },
  ] as Array<{
    symbol: string
    sector: string
    marketCap: "Small" | "Mid" | "Large"
    pe: number
    price: number
  }>
}

function wait(ms: number) {
  return new Promise((res) => setTimeout(res, ms))
}

const API_BASE =
  (typeof window !== "undefined" ? (window as any).NEXT_PUBLIC_API_BASE_URL : undefined) ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://localhost:8000"

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  })
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    throw new Error(`API ${path} ${res.status}: ${text}`)
  }
  // try json parse
  return (await res.json()) as T
}

export type RAGResponse = unknown // backend may return string or structured JSON
export async function postQuery(query: string, stream = false): Promise<RAGResponse> {
  return apiFetch<RAGResponse>("/api/v1/query", {
    method: "POST",
    body: JSON.stringify({ query, stream }),
  })
}

export type AnalyzePortfolioInput = {
  budget?: number
  strategy: string
  existing_portfolio?: Array<{ symbol: string; quantity?: number; avg_price?: number }>
  constraints?: Record<string, any>
}
export type AnalyzePortfolioResult = {
  allocation?: Record<string, number>
  strategy?: string
  expected_return?: number
  risk_level?: string
  sharpe_ratio?: number
  recommendations?: string[] | Array<{ text?: string }>
  rebalancing_needed?: boolean
}
export async function postAnalyzePortfolio(input: AnalyzePortfolioInput): Promise<AnalyzePortfolioResult> {
  return apiFetch<AnalyzePortfolioResult>("/api/v1/analyze/portfolio", {
    method: "POST",
    body: JSON.stringify(input),
  })
}

export type AnalyzeRiskInput = {
  portfolio: Array<{ symbol: string; quantity?: number; avg_price?: number }>
  time_horizon?: number
}
export type AnalyzeRiskResult = any
export async function postAnalyzeRisk(input: AnalyzeRiskInput): Promise<AnalyzeRiskResult> {
  return apiFetch<AnalyzeRiskResult>("/api/v1/analyze/risk", {
    method: "POST",
    body: JSON.stringify(input),
  })
}

export type ScreenerInput = {
  market_cap_min?: number
  market_cap_max?: number
  pe_min?: number
  pe_max?: number
  sector?: string
  min_volume?: number
  include_predictions?: boolean
}
export type ScreenerResult = {
  count: number
  stocks: Array<{
    symbol: string
    price: number
    pe_ratio?: number
    market_cap?: number
    sector?: string
    volume?: number
    predicted_price?: number | null
    predicted_prices?: number[]
  }>
  timestamp?: string
}
export async function postScreener(input: ScreenerInput): Promise<ScreenerResult> {
  return apiFetch<ScreenerResult>("/api/v1/screener", {
    method: "POST",
    body: JSON.stringify(input),
  })
}

export type MarketOverview = {
  market_breadth?: Record<string, any>
  sector_performance?: Record<string, number>
  top_gainers?: Array<{ symbol: string; change: number }>
  top_losers?: Array<{ symbol: string; change: number }>
  timestamp?: string
}
export async function getMarketOverview(): Promise<MarketOverview> {
  return apiFetch<MarketOverview>("/api/v1/market/overview")
}

// Stock details
export type StockDetails = Record<string, any>
export async function getStock(symbol: string): Promise<StockDetails> {
  return apiFetch<StockDetails>(`/api/v1/market/stock/${encodeURIComponent(symbol)}`)
}

// Health check
export type Health = { status: string; components?: Record<string, boolean> }
export async function getHealth(): Promise<Health> {
  return apiFetch<Health>("/health")
}

// Price Prediction
export type PredictionResult = {
  symbol: string
  current_price: number
  predicted_prices: number[]
  forecast_horizon: number
  predicted_next_price: number
  timestamp?: string
}
export async function getPricePrediction(symbol: string, forecast_horizon: number = 5): Promise<PredictionResult> {
  return apiFetch<PredictionResult>(`/api/v1/predict/${encodeURIComponent(symbol)}?forecast_horizon=${forecast_horizon}`)
}

// High-level aggregator for dashboard to keep UI stable
export async function getDashboardData() {
  const overview = await getMarketOverview()
  // Map backend overview into existing UI expectations
  const marketOverview = [
    { label: "Advances", value: String(overview.market_breadth?.advances ?? "-"), delta: 0 },
    { label: "Declines", value: String(overview.market_breadth?.declines ?? "-"), delta: 0 },
    { label: "Unchanged", value: String(overview.market_breadth?.unchanged ?? "-"), delta: 0 },
    { label: "Top Gainer", value: overview.top_gainers?.[0]?.symbol ?? "-", delta: overview.top_gainers?.[0]?.change ?? 0 },
  ]

  const sectorPerformance = overview.sector_performance ?? {}
  const portfolioBreakdown = Object.entries(sectorPerformance).slice(0, 4).map(([name, value], idx) => ({
    name,
    key: ["equity", "debt", "gold", "cash"][idx] ?? `k${idx}`,
    value: typeof value === "number" ? Math.max(0, Math.min(100, Math.round(value))) : 0,
  }))
  if (portfolioBreakdown.length === 0) {
    // fallback proportions
    portfolioBreakdown.push(
      { name: "Equity", key: "equity", value: 60 },
      { name: "Debt", key: "debt", value: 20 },
      { name: "Gold", key: "gold", value: 10 },
      { name: "Cash", key: "cash", value: 10 },
    )
  }

  const watchlist = (overview.top_gainers ?? []).slice(0, 6).map((g, i) => ({
    symbol: g.symbol,
    price: 0,
    change: g.change ?? 0,
    spark: Array.from({ length: 24 }, (_, t) => ({ t, p: 100 + Math.sin((t + i) / 3) * 5 })),
  }))

  return { marketOverview, portfolioBreakdown, watchlist }
}
