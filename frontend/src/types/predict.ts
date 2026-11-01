// mirrors backend/src/models/predict.ts. missing metadata fields are null since
// the client treats anything the server leaves out as unavailable

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
  smiles: string | null;
  bioactivity: Record<Target, TargetBioactivity>;
  toxicity: {
    nuclearReceptor: Record<NrTask, Output>;
    stressResponse: Record<SrTask, Output>;
  };
  fdaApproved: Output;
  metadata: {
    modelVersion: string | null;
    mcDropoutPasses: number | null;
    ciCoverage: number | null;
  };
}
