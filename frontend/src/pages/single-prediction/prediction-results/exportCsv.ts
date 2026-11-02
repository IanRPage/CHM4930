import type {
  Estimate,
  Output,
  PredictResponse,
} from "../../../types/predict.ts";
import { NR_TASKS, SR_TASKS } from "../../../types/predict.ts";

type Cell = string | number | null;
type Column = readonly [header: string, value: Cell];

function addEstimate(
  columns: Column[],
  label: string,
  output: Output<Estimate>,
) {
  columns.push([label, output.value]);
  columns.push([`${label} CI lower`, output.value === null ? null : output.lo]);
  columns.push([`${label} CI upper`, output.value === null ? null : output.hi]);
}

// Quote CSV special characters and prevent spreadsheets from executing formulas
// when a SMILES value starts with =, +, -, or @ (including after whitespace).
function csvCell(value: Cell): string {
  if (value === null) return "";
  let text = String(value);
  if (typeof value === "string" && /^\s*[=+@-]/u.test(text)) {
    text = `'${text}`;
  }
  return /[",\r\n]/u.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function predictionToCsv(result: PredictResponse): string {
  const columns: Column[] = [["SMILES", result.smiles]];

  // Export both targets, regardless of which toggle is active in the UI.
  for (const target of ["EGFR", "BACE1"] as const) {
    const bio = result.bioactivity[target];
    addEstimate(columns, `${target} pIC50`, bio.pic50);
    addEstimate(columns, `${target} IC50 (nM)`, bio.ic50Nm);
    columns.push([`${target} P(active) (0-1)`, bio.pActive.value]);
  }

  for (const task of NR_TASKS) {
    columns.push([
      `Tox21 ${task} probability (0-1)`,
      result.toxicity.nuclearReceptor[task].value,
    ]);
  }
  for (const task of SR_TASKS) {
    columns.push([
      `Tox21 ${task} probability (0-1)`,
      result.toxicity.stressResponse[task].value,
    ]);
  }

  columns.push(["FDA approval probability (0-1)", result.fdaApproved.value]);
  columns.push(["Model version", result.metadata.modelVersion]);
  columns.push(["MC-Dropout passes", result.metadata.mcDropoutPasses]);
  columns.push(["CI coverage (0-1)", result.metadata.ciCoverage]);

  return (
    columns.map(([header]) => csvCell(header)).join(",") +
    "\r\n" +
    columns.map(([, value]) => csvCell(value)).join(",") +
    "\r\n"
  );
}

export function downloadPredictionCsv(result: PredictResponse): void {
  const csv = predictionToCsv(result);
  // UTF-8 BOM helps spreadsheet applications recognize non-ASCII units.
  const blob = new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "mufu-prediction.csv";
  link.hidden = true;
  document.body.appendChild(link);
  try {
    link.click();
  } finally {
    link.remove();
    URL.revokeObjectURL(url);
  }
}
