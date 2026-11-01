import type {
  Estimate,
  Output,
  PredictResponse,
  TargetBioactivity,
  Unavailable,
  Value,
} from "../types/predict.ts";
import { NR_TASKS, SR_TASKS, TARGETS } from "../types/predict.ts";
import { PredictError, errorFromStatus } from "./errors.ts";

// Above the backend's 30 s inference timeout, so its 504 normally arrives first
export const PREDICT_TIMEOUT_MS = 35_000;

export const NOT_RETURNED = "Not returned by the server";

export interface PredictOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

type Json = Record<string, unknown>;

function isRecord(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function record(raw: unknown): Json {
  return isRecord(raw) ? raw : {};
}

function unavailable(raw: unknown): Unavailable {
  if (isRecord(raw) && raw.value === null && typeof raw.reason === "string") {
    return { value: null, reason: raw.reason };
  }
  return { value: null, reason: NOT_RETURNED };
}

function parseValue(raw: unknown): Output<Value> {
  return isRecord(raw) && isNumber(raw.value)
    ? { value: raw.value }
    : unavailable(raw);
}

function parseEstimate(raw: unknown): Output<Estimate> {
  return isRecord(raw) &&
    isNumber(raw.value) &&
    isNumber(raw.lo) &&
    isNumber(raw.hi)
    ? { value: raw.value, lo: raw.lo, hi: raw.hi }
    : unavailable(raw);
}

function fromKeys<K extends string, V>(
  keys: readonly K[],
  value: (key: K) => V,
): Record<K, V> {
  return Object.fromEntries(keys.map((key) => [key, value(key)])) as Record<
    K,
    V
  >;
}

function parseBioactivity(raw: unknown): TargetBioactivity {
  const target = record(raw);
  return {
    pic50: parseEstimate(target.pic50),
    ic50Nm: parseEstimate(target.ic50Nm),
    pActive: parseValue(target.pActive),
  };
}

export function parsePredictResponse(body: unknown): PredictResponse {
  if (!isRecord(body)) throw new PredictError("invalid-response");
  const bioactivity = record(body.bioactivity);
  const toxicity = record(body.toxicity);
  const nr = record(toxicity.nuclearReceptor);
  const sr = record(toxicity.stressResponse);
  const metadata = record(body.metadata);
  return {
    smiles: typeof body.smiles === "string" ? body.smiles : null,
    bioactivity: fromKeys(TARGETS, (t) => parseBioactivity(bioactivity[t])),
    toxicity: {
      nuclearReceptor: fromKeys(NR_TASKS, (t) => parseValue(nr[t])),
      stressResponse: fromKeys(SR_TASKS, (t) => parseValue(sr[t])),
    },
    fdaApproved: parseValue(body.fdaApproved),
    metadata: {
      modelVersion:
        typeof metadata.modelVersion === "string"
          ? metadata.modelVersion
          : null,
      mcDropoutPasses: isNumber(metadata.mcDropoutPasses)
        ? metadata.mcDropoutPasses
        : null,
      ciCoverage: isNumber(metadata.ciCoverage) ? metadata.ciCoverage : null,
    },
  };
}

async function readJson(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch (err) {
    if (err instanceof SyntaxError) return undefined;
    throw err;
  }
}

async function send(smiles: string, signal: AbortSignal) {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
    signal,
  });
  return { res, body: await readJson(res) };
}

export async function predict(
  smiles: string,
  { signal, timeoutMs = PREDICT_TIMEOUT_MS }: PredictOptions = {},
): Promise<PredictResponse> {
  const timeout = AbortSignal.timeout(timeoutMs);
  let result: Awaited<ReturnType<typeof send>>;
  try {
    result = await send(
      smiles,
      signal ? AbortSignal.any([signal, timeout]) : timeout,
    );
  } catch (err) {
    if (signal?.aborted) throw err;
    if (timeout.aborted) throw new PredictError("timeout");
    throw new PredictError("network");
  }

  const { res, body } = result;
  if (!res.ok) {
    const message =
      isRecord(body) && typeof body.error === "string" ? body.error : undefined;
    throw errorFromStatus(res.status, message);
  }
  if (body === undefined) throw new PredictError("invalid-response");
  return parsePredictResponse(body);
}
