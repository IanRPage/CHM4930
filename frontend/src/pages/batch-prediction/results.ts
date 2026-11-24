import type { BatchResult } from "../../types/batch.ts";
import type { Output, Target } from "../../types/predict.ts";
import { NR_TASKS, SR_TASKS } from "../../types/predict.ts";
import { HIGH_RISK } from "../single-prediction/prediction-results/format.ts";

export const ACTIVE_PIC50 = 6;

export const TOX_GROUPS = [
  { label: "Tox21 · NR", tasks: NR_TASKS },
  { label: "Tox21 · SR", tasks: SR_TASKS },
  { label: "ClinTox", tasks: ["CT_TOX"] },
] as const;

export function toxOutputs(r: BatchResult): [task: string, output: Output][] {
  if (!r.prediction) return [];
  const { toxicity, clinicalToxicity } = r.prediction;
  return [
    ...NR_TASKS.map(
      (t) => [t, toxicity.nuclearReceptor[t]] as [string, Output],
    ),
    ...SR_TASKS.map((t) => [t, toxicity.stressResponse[t]] as [string, Output]),
    ["CT_TOX", clinicalToxicity],
  ];
}

export function isActive(r: BatchResult, target: Target) {
  const pic50 = r.prediction?.bioactivity[target].pic50.value;
  return pic50 != null && pic50 >= ACTIVE_PIC50;
}

export function isHighTox(r: BatchResult) {
  return toxOutputs(r).some(
    ([, o]) => o.value !== null && o.value >= HIGH_RISK,
  );
}

export function ciHalfWidth(r: BatchResult, target: Target) {
  const pic50 = r.prediction?.bioactivity[target].pic50;
  return pic50 && pic50.value !== null ? (pic50.hi - pic50.lo) / 2 : null;
}

export function median(values: number[]) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
