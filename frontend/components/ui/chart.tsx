import * as React from "react"
import { cn } from "@/lib/utils"

export function ChartContainer({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("relative", className)} {...props}>
      {children}
    </div>
  )
}

export const ChartTooltip = ({ content }: { content: React.ReactNode }) => <>{content}</>
export const ChartTooltipContent = (props: { hideLabel?: boolean }) => <div />
export const ChartLegend = (props: any) => null
export const ChartLegendContent = () => <div />


