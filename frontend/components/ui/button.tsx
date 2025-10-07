import * as React from "react"
import { cn } from "@/lib/utils"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "outline" | "secondary"
  size?: "sm" | "md"
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", ...props }, ref) => {
    const base =
      "inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 disabled:opacity-50 disabled:pointer-events-none"
    const variants: Record<string, string> = {
      default: "bg-primary text-primary-foreground hover:bg-primary/90",
      outline: "border border-input bg-background hover:bg-muted",
      secondary: "bg-muted text-foreground hover:bg-muted/80",
    }
    const sizes: Record<string, string> = {
      sm: "h-8 px-3 text-xs",
      md: "h-9 px-4 text-sm",
    }
    return <button ref={ref} className={cn(base, variants[variant], sizes[size], className)} {...props} />
  },
)
Button.displayName = "Button"
