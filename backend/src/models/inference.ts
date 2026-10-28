import {
  NR_TASKS,
  SR_TASKS,
  TARGETS,
  type Output,
  type Unavailable,
  type PredictResponse,
  type Target,
  type TargetBioactivity,
} from "./predict.ts";
import { fromKeys, isRecord } from "./util.ts";

export const TOX_TASKS = [
  ...NR_TASKS,
  ...SR_TASKS,
  "FDA_APPROVED",
  "CT_TOX",
] as const;

export type ToxTask = (typeof TOX_TASKS)[number];

type Pic50 =
  | { pic50: number; pic50_lo: number; pic50_hi: number }
  | { pic50: null; pic50_lo: null; pic50_hi: null };

export type InferenceBioactivity = Pic50 & { p_active: number | null };

export interface InferenceResponse {
  smiles: string;
  model_version: string;
  n_samples: number;
  coverage: number;
  bioactivity: Record<Target, InferenceBioactivity>;
  toxicity: Record<ToxTask, number | null>;
}

const NOT_RETURNED = "Not returned by the model";

function fail(path: string, expected: string): never {
  throw new Error(`Invalid inference response: ${path} must be ${expected}`);
}

function record(value: unknown, path: string) {
  if (!isRecord(value)) fail(path, "an object");
  return value;
}

function nonEmptyString(value: unknown, path: string) {
  if (typeof value !== "string" || !value) fail(path, "a non-empty string");
  return value;
}

function nullableNumber(value: unknown, path: string) {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(path, "a finite number or null");
  }
  return value;
}

function nullableProbability(value: unknown, path: string) {
  const p = nullableNumber(value, path);
  if (p !== null && (p < 0 || p > 1)) fail(path, "in [0, 1]");
  return p;
}

function parseBioactivity(value: unknown, path: string): InferenceBioactivity {
  const bio = record(value, path);
  const pic50 = nullableNumber(bio.pic50, `${path}.pic50`);
  const lo = nullableNumber(bio.pic50_lo, `${path}.pic50_lo`);
  const hi = nullableNumber(bio.pic50_hi, `${path}.pic50_hi`);
  const p_active = nullableProbability(bio.p_active, `${path}.p_active`);

  if (pic50 === null) {
    if (lo !== null || hi !== null) {
      fail(`${path}.pic50_lo/pic50_hi`, "null when pic50 is null");
    }
    return { pic50, pic50_lo: null, pic50_hi: null, p_active };
  }
  if (lo === null || hi === null || lo > pic50 || pic50 > hi) {
    fail(path, "ordered as pic50_lo <= pic50 <= pic50_hi");
  }
  return { pic50, pic50_lo: lo, pic50_hi: hi, p_active };
}

export function parseInferenceResponse(body: unknown): InferenceResponse {
  const root = record(body, "body");
  const bioactivity = record(root.bioactivity, "bioactivity");
  const toxicity = record(root.toxicity, "toxicity");

  const n_samples = root.n_samples;
  if (typeof n_samples !== "number" || !Number.isInteger(n_samples)) {
    fail("n_samples", "an integer");
  }
  if (n_samples < 2) fail("n_samples", "at least 2");
  const coverage = root.coverage;
  if (typeof coverage !== "number" || !(coverage > 0 && coverage < 1)) {
    fail("coverage", "a number in (0, 1)");
  }

  return {
    smiles: nonEmptyString(root.smiles, "smiles"),
    model_version: nonEmptyString(root.model_version, "model_version"),
    n_samples,
    coverage,
    bioactivity: fromKeys(TARGETS, (target) =>
      parseBioactivity(bioactivity[target], `bioactivity.${target}`),
    ),
    toxicity: fromKeys(TOX_TASKS, (task) =>
      nullableProbability(toxicity[task], `toxicity.${task}`),
    ),
  };
}

function unavailable(): Unavailable {
  return { value: null, reason: NOT_RETURNED };
}

function output(value: number | null): Output {
  return value === null ? unavailable() : { value };
}

function ic50Nm(pic50: number) {
  return 10 ** (9 - pic50);
}

function toBioactivity(raw: InferenceBioactivity): TargetBioactivity {
  const pActive = output(raw.p_active);
  if (raw.pic50 === null) {
    return {
      pic50: unavailable(),
      ic50Nm: unavailable(),
      pActive,
    };
  }
  return {
    pic50: { value: raw.pic50, lo: raw.pic50_lo, hi: raw.pic50_hi },
    ic50Nm: {
      value: ic50Nm(raw.pic50),
      lo: ic50Nm(raw.pic50_hi),
      hi: ic50Nm(raw.pic50_lo),
    },
    pActive,
  };
}

export function toPredictResponse(raw: InferenceResponse): PredictResponse {
  return {
    smiles: raw.smiles,
    bioactivity: fromKeys(TARGETS, (target) =>
      toBioactivity(raw.bioactivity[target]),
    ),
    toxicity: {
      nuclearReceptor: fromKeys(NR_TASKS, (task) => output(raw.toxicity[task])),
      stressResponse: fromKeys(SR_TASKS, (task) => output(raw.toxicity[task])),
    },
    fdaApproved: output(raw.toxicity.FDA_APPROVED),
    metadata: {
      modelVersion: raw.model_version,
      mcDropoutPasses: raw.n_samples,
      ciCoverage: raw.coverage,
    },
  };
}
