import type {
  BatchChunkResponse,
  BatchResult,
  BatchRowError,
} from "../types/batch.ts";
import { PredictError, errorFromStatus } from "./errors.ts";
import {
  isNumber,
  isRecord,
  parsePredictResponse,
  readJson,
} from "./predict.ts";

export const CHUNK_SIZE = 10;
const BASE_DELAY_MS = 1000;
const MAX_DELAY_MS = 30_000;
const MAX_BUSY_RETRIES = 5;

export interface BatchProgress {
  processed: number;
  total: number;
  invalid: number;
}

export interface BatchCallbacks {
  onProgress: (progress: BatchProgress) => void;
  onComplete: (results: BatchResult[]) => void;
  onError: (error: Error) => void;
}

function parseBatchChunkResponse(body: unknown): BatchChunkResponse {
  if (
    !isRecord(body) ||
    !Array.isArray(body.results) ||
    !Array.isArray(body.errors)
  ) {
    throw new PredictError("invalid-response");
  }
  const errors = body.errors.map((e: unknown): BatchRowError => {
    if (!isRecord(e) || !isNumber(e.index) || typeof e.reason !== "string") {
      throw new PredictError("invalid-response");
    }
    return {
      index: e.index,
      smiles: typeof e.smiles === "string" ? e.smiles : "",
      reason: e.reason,
    };
  });
  const results = body.results.map((r: unknown) =>
    r === null ? null : parsePredictResponse(r),
  );
  return { results, errors };
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function retryDelay(res: Response, attempt: number) {
  const retryAfter = Number(res.headers.get("Retry-After"));
  const header = Number.isFinite(retryAfter) ? retryAfter * 1000 : 0;
  return Math.min(Math.max(header, BASE_DELAY_MS * 2 ** attempt), MAX_DELAY_MS);
}

async function postChunk(
  smiles: string[],
  signal: AbortSignal,
): Promise<BatchChunkResponse> {
  let gatewayRetried = false;
  for (let attempt = 0; ; attempt++) {
    let res: Response;
    try {
      res = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
        signal,
      });
    } catch {
      throw new PredictError("network");
    }

    if (res.ok) {
      return parseBatchChunkResponse(await readJson(res));
    }

    const busy = res.status === 503 || res.status === 429;
    const gateway = res.status === 502 || res.status === 504;
    if (busy && attempt < MAX_BUSY_RETRIES) {
      await sleep(retryDelay(res, attempt));
    } else if (gateway && !gatewayRetried) {
      gatewayRetried = true;
      await sleep(BASE_DELAY_MS);
    } else {
      const body = await readJson(res);
      const message =
        isRecord(body) && typeof body.error === "string"
          ? body.error
          : undefined;
      throw errorFromStatus(res.status, message);
    }
  }
}

// Predicts each unique SMILES once, in chunks, then fans the results back out
// to every input row. An aborted run resolves without calling onComplete.
export async function batchPredict(
  smiles: string[],
  { onProgress, onComplete, onError }: BatchCallbacks,
  signal: AbortSignal,
): Promise<void> {
  const unique = [...new Set(smiles)];
  const outcomes = new Map<string, Pick<BatchResult, "prediction" | "error">>();
  let invalid = 0;

  try {
    for (let start = 0; start < unique.length; start += CHUNK_SIZE) {
      if (signal.aborted) return;
      const chunk = unique.slice(start, start + CHUNK_SIZE);
      const { results, errors } = await postChunk(chunk, signal);
      if (results.length !== chunk.length) {
        throw new PredictError("invalid-response");
      }
      chunk.forEach((s, i) => {
        const reason = errors.find((e) => e.index === i)?.reason;
        const prediction = results[i];
        const error = prediction ? null : reason ?? "Prediction failed";
        if (error) invalid++;
        outcomes.set(s, { prediction, error });
      });
      onProgress({
        processed: start + chunk.length,
        total: unique.length,
        invalid,
      });
    }
  } catch (err) {
    if (signal.aborted) return;
    onError(err as Error);
    return;
  }

  onComplete(
    smiles.map((s, i) => ({ row: i + 1, smiles: s, ...outcomes.get(s)! })),
  );
}
