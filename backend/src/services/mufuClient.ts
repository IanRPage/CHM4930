import { httpError } from "../errors.ts";
import {
  parseInferenceResponse,
  toPredictResponse,
} from "../models/inference.ts";
import type { PredictResponse } from "../models/predict.ts";
import { isRecord } from "../models/util.ts";

// One request runs many MC-Dropout passes on the inference server plus a tunnel
// round trip, and must finish inside the Vercel function duration
export const MUFU_TIMEOUT_MS = 30_000;
// A batch must finish before the Vercel function duration (40 s)
export const BATCH_TIMEOUT_MS = 35_000;

const INVALID_SMILES = "Invalid SMILES";
const BUSY = "Model server is busy, try again shortly";
const UNAVAILABLE = "Model server is unavailable";
const TIMED_OUT = "Model server timed out";
const BAD_RESPONSE = "Model server returned an invalid response";

export interface MuFuClientOptions {
  url: string | undefined;
  apiKey: string | undefined;
  timeoutMs?: number;
}

function errorName(err: unknown) {
  return err instanceof Error ? err.name : typeof err;
}

function isTimeout(err: unknown) {
  return err instanceof Error && err.name === "TimeoutError";
}

function upstreamError(status: number, message: string, reason: string) {
  console.error(`MuFuClient: ${reason}`);
  return httpError(status, message);
}

function timedOut() {
  return upstreamError(504, TIMED_OUT, "request timed out");
}

function statusError(status: number, retryAfter?: string) {
  if (status === 422) return httpError(400, INVALID_SMILES);
  if (status === 429 || status === 503) {
    return httpError(503, BUSY, retryAfter);
  }
  if (status === 502 || status === 504 || status === 530) {
    return upstreamError(503, UNAVAILABLE, `gateway status ${status}`);
  }
  return upstreamError(502, BAD_RESPONSE, `unexpected status ${status}`);
}

export class MuFuClient {
  private readonly url: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;

  constructor({ url, apiKey, timeoutMs = MUFU_TIMEOUT_MS }: MuFuClientOptions) {
    if (!url || !apiKey) {
      throw new Error("MUFU_API_URL and MUFU_API_KEY must be set");
    }
    this.url = url.replace(/\/+$/, "");
    this.apiKey = apiKey;
    this.timeoutMs = timeoutMs;
  }

  async modelVersion(): Promise<string | null> {
    try {
      const res = await fetch(`${this.url}/health`, {
        redirect: "error",
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      if (!res.ok) {
        await res.body?.cancel().catch(() => {});
        return null;
      }
      const body: unknown = await res.json();
      return isRecord(body) &&
        typeof body.model_version === "string" &&
        body.model_version
        ? body.model_version
        : null;
    } catch {
      return null;
    }
  }

  async predict(smiles: string): Promise<PredictResponse> {
    let res: Response;
    try {
      res = await fetch(`${this.url}/predict`, {
        method: "POST",
        redirect: "error",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": this.apiKey,
        },
        body: JSON.stringify({ smiles }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (err) {
      throw isTimeout(err)
        ? timedOut()
        : upstreamError(503, UNAVAILABLE, `request failed (${errorName(err)})`);
    }

    if (!res.ok) {
      const retryAfter = res.headers.get("Retry-After") ?? undefined;
      await res.body?.cancel().catch(() => {});
      throw statusError(res.status, retryAfter);
    }

    let body: unknown;
    try {
      body = await res.json();
    } catch (err) {
      if (isTimeout(err)) throw timedOut();
      if (err instanceof SyntaxError) {
        throw upstreamError(502, BAD_RESPONSE, "response body is not JSON");
      }
      throw upstreamError(
        503,
        UNAVAILABLE,
        `response body failed (${errorName(err)})`,
      );
    }

    try {
      return toPredictResponse(parseInferenceResponse(body));
    } catch (err) {
      const reason = err instanceof Error ? err.message : "invalid response";
      throw upstreamError(502, BAD_RESPONSE, reason);
    }
  }

  async predictBatch(
    smilesList: string[],
    signal?: AbortSignal,
  ): Promise<PredictResponse[]> {
    const deadline = Date.now() + BATCH_TIMEOUT_MS;
    const results: PredictResponse[] = [];
    for (const smiles of smilesList) {
      if (signal?.aborted) throw httpError(499, "Client closed request");
      if (Date.now() >= deadline) throw timedOut();
      results.push(await this.predict(smiles));
    }
    return results;
  }
}
