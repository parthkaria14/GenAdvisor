"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

export function Sheet({ children, open, onOpenChange, ...props }: React.ComponentProps<"div"> & { open?: boolean; onOpenChange?: (open: boolean) => void }) {
  React.useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onOpenChange?.(false)
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onOpenChange])
  return (
    <div data-slot="sheet" aria-hidden={!open} className={cn("fixed inset-0 z-50", !open && "hidden")} {...props}>
      <div className="bg-black/40 absolute inset-0" onClick={() => onOpenChange?.(false)} />
      {children}
    </div>
  )
}

export function SheetContent({ className, side = "left", ...props }: React.ComponentProps<"div"> & { side?: "left" | "right" }) {
  return (
    <div
      className={cn(
        "bg-background text-foreground fixed top-0 h-full w-80 shadow-lg transition-transform",
        side === "left" ? "left-0" : "right-0",
        className,
      )}
      {...props}
    />
  )
}

export const SheetHeader = (props: React.ComponentProps<"div">) => <div {...props} />
export const SheetTitle = (props: React.ComponentProps<"div">) => <div {...props} />
export const SheetDescription = (props: React.ComponentProps<"div">) => <div {...props} />



