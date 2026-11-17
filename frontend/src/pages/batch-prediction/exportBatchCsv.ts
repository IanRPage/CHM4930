import type { BatchResult } from "../../types/batch.ts";
import type { Target } from "../../types/predict.ts";
import { NR_TASKS, SR_TASKS } from "../../types/predict.ts";
import {
  csvCell,
  downloadCsv,
} from "../single-prediction/prediction-results/exportCsv.ts";
import type { Cell } from "../single-prediction/prediction-results/exportCsv.ts";
import { ciHalfWidth, toxOutputs } from "./results.ts";

export function batchToCsv(results: BatchResult[], target: Target): string {
  const header = [
    "#",
    "SMILES",
    `${target} pIC50`,
    `${target} 95% CI half-width`,
    `${target} IC50 (nM)`,
    `${target} P(active) (0-1)`,
    ...NR_TASKS.map((t) => `Tox21 ${t} probability (0-1)`),
    ...SR_TASKS.map((t) => `Tox21 ${t} probability (0-1)`),
    "Clinical toxicity probability (0-1)",
    "Error",
  ];
  const lines = results.map((r) => {
    const bio = r.prediction?.bioactivity[target];
    const cells: Cell[] = [
      r.row,
      r.smiles,
      bio?.pic50.value ?? null,
      ciHalfWidth(r, target),
      bio?.ic50Nm.value ?? null,
      bio?.pActive.value ?? null,
      ...(r.prediction
        ? toxOutputs(r).map(([, o]) => o.value)
        : Array<null>(NR_TASKS.length + SR_TASKS.length + 1).fill(null)),
      r.error,
    ];
    return cells.map(csvCell).join(",");
  });
  return [header.map(csvCell).join(","), ...lines].join("\r\n") + "\r\n";
}

export function downloadBatchCsv(results: BatchResult[], target: Target) {
  downloadCsv(batchToCsv(results, target), "mufu-batch-predictions.csv");
}
