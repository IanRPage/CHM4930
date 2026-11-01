import type { Output } from "../../../types/predict.ts";

// tinted panels go green at or above this probability
const FAVORABLE = 0.5;

export function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function tone(output: Output) {
  return output.value !== null && output.value >= FAVORABLE
    ? "predict-panel-good"
    : "predict-panel-neutral";
}
