// mirrors backend/src/models/batch.ts
import type { PredictResponse } from "./predict.ts";

export interface BatchRowError {
  index: number;
  smiles: string;
  reason: string;
}

export interface BatchChunkResponse {
  results: (PredictResponse | null)[];
  errors: BatchRowError[];
}

export interface BatchResult {
  row: number;
  smiles: string;
  prediction: PredictResponse | null;
  error: string | null;
}
