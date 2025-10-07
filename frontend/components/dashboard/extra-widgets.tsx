"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"

export function ExtraWidgets() {
  return (
    <div className="grid gap-4 md:grid-cols-3 animate-float-in">
      <Card className="card-glass">
        <CardHeader>
          <CardTitle className="text-pretty">Market Movers</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-2 text-sm">
          <div className="flex items-center justify-between">
            <span>NIFTY BANK</span>
            <Badge variant="secondary" className="text-green-600">
              +1.24%
            </Badge>
          </div>
          <Separator />
          <div className="flex items-center justify-between">
            <span>IT Index</span>
            <Badge variant="secondary" className="text-pink-600">
              -0.32%
            </Badge>
          </div>
          <Separator />
          <div className="flex items-center justify-between">
            <span>Auto Index</span>
            <Badge variant="secondary" className="text-green-600">
              +0.67%
            </Badge>
          </div>
        </CardContent>
      </Card>

      <Card className="card-glass">
        <CardHeader>
          <CardTitle className="text-pretty">Today’s Highlights</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3 text-sm">
          <div className="flex items-center justify-between">
            <span>Adv/Dec</span>
            <span className="font-medium">32 / 18</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Turnover</span>
            <span className="font-medium">₹29,274 Cr</span>
          </div>
          <div className="flex items-center justify-between">
            <span>VIX</span>
            <Badge className="bg-muted">12.3</Badge>
          </div>
        </CardContent>
      </Card>

      <Card className="card-glass">
        <CardHeader>
          <CardTitle className="text-pretty">News</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <a className="underline underline-offset-4 hover:opacity-80" href="#">
            RBI policy preview: What to expect this week
          </a>
          <a className="underline underline-offset-4 hover:opacity-80" href="#">
            PSU banks extend rally on credit growth optimism
          </a>
          <a className="underline underline-offset-4 hover:opacity-80" href="#">
            IT services outlook stabilizes as deal wins rise
          </a>
        </CardContent>
      </Card>
    </div>
  )
}
