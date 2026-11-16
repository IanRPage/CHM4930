import { httpError } from "../errors.ts";
import type { PredictResponse } from "./predict.ts";
import { parseSmiles } from "./predict.ts";
import { isRecord } from "./util.ts";

export const MAX_CHUNK_SIZE = 10;

export interface BatchRowError {
  index: number;
  smiles: string;
  reason: string;
}

export interface BatchPredictResponse {
  results: (PredictResponse | null)[];
  errors: BatchRowError[];
}

export interface ParsedBatchRequest {
  valid: Map<number, string>;
  errors: BatchRowError[];
  inputLength: number;
}

export function parseBatchPredictRequest(body: unknown): ParsedBatchRequest {
  if (!isRecord(body)) {
    throw httpError(400, "Request body must be a JSON object");
  }
  if (!Array.isArray(body.smiles)) {
    throw httpError(400, "smiles must be an array");
  }
  if (body.smiles.length === 0) {
    throw httpError(400, "smiles must not be empty");
  }
  if (body.smiles.length > MAX_CHUNK_SIZE) {
    throw httpError(400, `smiles must have at most ${MAX_CHUNK_SIZE} entries`);
  }

  const rows: unknown[] = body.smiles;
  const valid = new Map<number, string>();
  const errors: BatchRowError[] = [];

  for (let i = 0; i < rows.length; i++) {
    try {
      valid.set(i, parseSmiles(rows[i]));
    } catch (err) {
      const reason = err instanceof Error ? err.message : "Invalid SMILES";
      const row = rows[i];
      let raw: string;
      try {
        raw = typeof row === "string" ? row : String(row);
      } catch {
        raw = "[invalid]";
      }
      errors.push({ index: i, smiles: raw, reason });
    }
  }

  return { valid, errors, inputLength: rows.length };
}
