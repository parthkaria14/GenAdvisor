import * as React from "react"
import { cn } from "@/lib/utils"

export const Table = (props: React.TableHTMLAttributes<HTMLTableElement>) => (
  <table className={cn("w-full text-sm", props.className)} {...props} />
)
export const TableHeader = (props: React.HTMLAttributes<HTMLTableSectionElement>) => <thead {...props} />
export const TableBody = (props: React.HTMLAttributes<HTMLTableSectionElement>) => <tbody {...props} />
export const TableRow = (props: React.HTMLAttributes<HTMLTableRowElement>) => <tr {...props} />
export const TableHead = (props: React.ThHTMLAttributes<HTMLTableCellElement>) => (
  <th className={cn("px-2 py-1 text-left font-medium text-muted-foreground", props.className)} {...props} />
)
export const TableCell = (props: React.TdHTMLAttributes<HTMLTableCellElement>) => (
  <td className={cn("px-2 py-1", props.className)} {...props} />
)








