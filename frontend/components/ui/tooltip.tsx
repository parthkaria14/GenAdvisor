"use client"

import * as React from "react"

export const TooltipProvider = ({ children }: { children: React.ReactNode; delayDuration?: number }) => <>{children}</>

export const Tooltip = ({ children }: { children: React.ReactNode }) => <>{children}</>

export const TooltipTrigger = ({ children, asChild }: { children: React.ReactNode; asChild?: boolean }) => <>{children}</>

export const TooltipContent = ({ children, hidden }: { children: React.ReactNode; hidden?: boolean }) => (
  <>{!hidden && children}</>
)


