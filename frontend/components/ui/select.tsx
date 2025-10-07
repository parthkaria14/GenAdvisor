"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

export function Select({ children, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cn("bg-background border-input text-foreground h-9 rounded-md border px-3 text-sm" )} {...props}>
      {children}
    </select>
  )
}

export function SelectTrigger({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("h-9 rounded-md border px-3 text-sm flex items-center justify-between", className)} {...props} />
}

export function SelectValue(props: { placeholder?: string }) {
  return <span className="text-muted-foreground">{props.placeholder}</span>
}

export const SelectContent = (props: React.HTMLAttributes<HTMLDivElement>) => <div {...props} />
export const SelectItem = (props: React.HTMLAttributes<HTMLDivElement>) => <div {...props} />


