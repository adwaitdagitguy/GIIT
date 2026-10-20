import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatScientific(val: number | null | undefined, precision = 4): string {
  if (val === null || val === undefined || isNaN(val)) return "N/A"
  if (Math.abs(val) < 0.0001 || Math.abs(val) >= 10000) {
    return val.toExponential(precision)
  }
  return val.toFixed(precision)
}

export function formatPercent(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) return "N/A"
  return `${(val * 100).toFixed(1)}%`
}
