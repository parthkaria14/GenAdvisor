"use client"

import type React from "react"

import { useState } from "react"
import {
  SidebarProvider,
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarSeparator,
} from "@/components/ui/sidebar"
import { Home, MessageSquare, PieChart, Filter, Settings } from "lucide-react"
import DashboardView from "@/components/dashboard/dashboard-view"
import AIAdvisorView from "@/components/advisor/ai-advisor-view"
import PortfolioManagerView from "@/components/portfolio/portfolio-manager-view"
import StockScreenerView from "@/components/screener/stock-screener-view"
import Topbar from "@/components/layout/topbar"
import { AnimatedOrbs } from "@/components/decor/animated-orbs" // add animated background

type ViewKey = "dashboard" | "advisor" | "portfolio" | "screener" | "settings"

const NAV: Array<{ key: ViewKey; label: string; icon: React.ElementType }> = [
  { key: "dashboard", label: "Dashboard", icon: Home },
  { key: "advisor", label: "AI Advisor", icon: MessageSquare },
  { key: "portfolio", label: "Portfolio", icon: PieChart },
  { key: "screener", label: "Screener", icon: Filter },
  { key: "settings", label: "Settings", icon: Settings },
]

export default function Page() {
  const [view, setView] = useState<ViewKey>("dashboard")

  return (
    <SidebarProvider>
      <div className="flex min-h-svh">
        <Sidebar variant="sidebar" collapsible="icon" className="border-r">
          <SidebarHeader>
            <div className="flex items-center gap-2 px-2 h-10">
              <div className="size-6 rounded bg-primary/10 flex items-center justify-center">
                <span className="text-primary text-xs font-bold">GA</span>
              </div>
              <span className="text-sm font-semibold">GenAdvisor</span>
            </div>
          </SidebarHeader>
          <SidebarSeparator />
          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupLabel>Navigation</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {NAV.map((item) => (
                    <SidebarMenuItem key={item.key}>
                      <SidebarMenuButton
                        isActive={view === item.key}
                        onClick={() => setView(item.key)}
                        tooltip={item.label}
                      >
                        <item.icon />
                        <span>{item.label}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>
          <SidebarFooter>
            <div className="px-2 py-1 text-xs text-muted-foreground">Markets • NSE/BSE</div>
          </SidebarFooter>
          <SidebarRail />
        </Sidebar>

        <SidebarInset>
          <AnimatedOrbs />
          {/* Top Bar */}
          <Topbar />

          <main className="flex-1 p-4 md:p-6 container-fluid">
            {view === "dashboard" && <DashboardView />}
            {view === "advisor" && <AIAdvisorView />}
            {view === "portfolio" && <PortfolioManagerView />}
            {view === "screener" && <StockScreenerView />}
            {view === "settings" && <div className="text-muted-foreground text-sm">Settings coming soon.</div>}
          </main>
        </SidebarInset>
      </div>
    </SidebarProvider>
  )
}
