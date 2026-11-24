import type { DatasetHistogram } from "../../types/dataset.ts";

export const count = (value: number) => value.toLocaleString("en-US");
export const decimal = (value: number | null, digits = 1) =>
  value === null
    ? "—"
    : value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
export const percent = (value: number | null) =>
  value === null ? "No labels" : `${decimal(value * 100)}%`;

// Targets have different histogram bounds. Align by bin edges, never by index.
export function combineHistograms(
  histograms: DatasetHistogram[],
): DatasetHistogram {
  const binWidth = histograms[0].binWidth;
  const first = Math.min(...histograms.map((h) => h.edges[0]));
  const last = Math.max(...histograms.map((h) => h.edges.at(-1)!));
  const bins = Math.round((last - first) / binWidth);
  const counts = Array<number>(bins).fill(0);
  for (const histogram of histograms) {
    const offset = Math.round((histogram.edges[0] - first) / binWidth);
    histogram.counts.forEach((value, index) => {
      counts[offset + index] += value;
    });
  }
  return {
    binWidth,
    edges: Array.from({ length: bins + 1 }, (_, i) => first + i * binWidth),
    counts,
  };
}

export function histogramAt(histograms: DatasetHistogram[], width: number) {
  return histograms.find((h) => h.binWidth === width) ?? histograms[0];
}

export function elementColor(element: string) {
  if (["C", "N", "H"].includes(element)) return "var(--blue-500)";
  if (element === "O") return "var(--explorer-orange)";
  if (element === "S") return "var(--amber-500)";
  if (["F", "Cl", "Br", "I", "At", "Ts"].includes(element))
    return "var(--red-500)";
  return "var(--slate-400)";
}
