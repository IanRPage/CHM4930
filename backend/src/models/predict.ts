import { httpError } from "../errors.ts";
import { isRecord } from "./util.ts";

export const TARGETS = ["BACE1", "EGFR"] as const;
export const NR_TASKS = [
  "NR-AR",
  "NR-AR-LBD",
  "NR-AhR",
  "NR-Aromatase",
  "NR-ER",
  "NR-ER-LBD",
  "NR-PPAR-gamma",
] as const;
export const SR_TASKS = [
  "SR-ARE",
  "SR-ATAD5",
  "SR-HSE",
  "SR-MMP",
  "SR-p53",
] as const;
export const SMILES_MAX_LENGTH = 1000;

export type Target = (typeof TARGETS)[number];
export type NrTask = (typeof NR_TASKS)[number];
export type SrTask = (typeof SR_TASKS)[number];

export interface PredictRequest {
  smiles: string;
}

export interface Unavailable {
  value: null;
  reason: string;
}

export interface Value {
  value: number;
}

export interface Estimate extends Value {
  lo: number;
  hi: number;
}

export type Output<T extends Value = Value> = T | Unavailable;

export interface TargetBioactivity {
  pic50: Output<Estimate>;
  ic50Nm: Output<Estimate>;
  pActive: Output;
}

export interface PredictResponse {
  smiles: string;
  bioactivity: Record<Target, TargetBioactivity>;
  toxicity: {
    nuclearReceptor: Record<NrTask, Output>;
    stressResponse: Record<SrTask, Output>;
  };
  fdaApproved: Output;
  metadata: {
    modelVersion: string;
    mcDropoutPasses: number;
    ciCoverage: number;
  };
}

export function parsePredictRequest(body: unknown): PredictRequest {
  if (!isRecord(body)) {
    throw httpError(400, "Request body must be a JSON object");
  }
  if (typeof body.smiles !== "string") {
    throw httpError(400, "smiles must be a string");
  }
  const smiles = body.smiles.trim();
  if (!smiles) {
    throw httpError(400, "smiles must not be empty");
  }
  if (smiles.length > SMILES_MAX_LENGTH) {
    throw httpError(
      400,
      `smiles must be at most ${SMILES_MAX_LENGTH} characters`,
    );
  }
  if (/\s/.test(smiles)) {
    throw httpError(400, "smiles must not contain whitespace");
  }
  return { smiles };
}
