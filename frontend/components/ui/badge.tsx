import type * as React from "react"
import { cn } from "@/lib/utils"

export function Badge({
  className,
  variant = "default",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { variant?: "default" | "secondary" | "outline" }) {
  const base = "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium"
  const variants: Record<string, string> = {
    default: "bg-primary text-primary-foreground border-transparent",
    secondary: "bg-muted text-foreground border-transparent",
    outline: "border-border text-muted-foreground",
  }
  return <span className={cn(base, variants[variant], className)} {...props} />
}
