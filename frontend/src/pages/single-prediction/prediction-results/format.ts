import type { PanelTone } from "../../../components/common/Panel.tsx";
import type { Output } from "../../../types/predict.ts";

// tinted panels go green at or above this probability
const FAVORABLE = 0.5;

export function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function tone(output: Output): PanelTone {
  return output.value !== null && output.value >= FAVORABLE
    ? "good"
    : "neutral";
}
