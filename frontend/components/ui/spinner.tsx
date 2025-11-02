import * as React from "react"
import { cn } from "@/lib/utils"

export function Spinner({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("inline-block size-4 animate-spin rounded-full border-2 border-muted border-t-foreground", className)}
      {...props}
    />
  )
}








