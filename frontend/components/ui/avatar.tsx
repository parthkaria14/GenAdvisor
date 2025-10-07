import type * as React from "react"
import { cn } from "@/lib/utils"

export function Avatar({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("inline-flex h-8 w-8 items-center justify-center rounded-full bg-muted", className)}
      {...props}
    />
  )
}
export function AvatarFallback({ children }: { children?: React.ReactNode }) {
  return <span className="text-xs text-muted-foreground">{children}</span>
}

export function AvatarImage({ className, ...props }: React.ImgHTMLAttributes<HTMLImageElement>) {
  return <img className={cn("h-full w-full rounded-full object-cover", className)} {...props} />
}
