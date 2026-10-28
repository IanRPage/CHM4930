import { describe, expect, it } from "vitest";

import {
  parseInferenceResponse,
  TOX_TASKS,
  toPredictResponse,
} from "../../src/models/inference.ts";
import { NR_TASKS, SR_TASKS } from "../../src/models/predict.ts";

type Json = Record<string, unknown>;

// Loosely typed so tests can corrupt it freely; the parser is what checks types.
function rawResponse() {
  return {
    smiles: "CCO",
    model_version: "mufu-test",
    n_samples: 50,
    coverage: 0.95,
    bioactivity: {
      BACE1: { pic50: 5, pic50_lo: 4, pic50_hi: 6, p_active: 0.2 },
      EGFR: { pic50: 7, pic50_lo: 6.5, pic50_hi: 8, p_active: 0.9 },
    } as Record<string, Json>,
    toxicity: Object.fromEntries(TOX_TASKS.map((task) => [task, 0.1])) as Json,
  };
}

describe("parseInferenceResponse", () => {
  it("accepts a valid response", () => {
    const raw = rawResponse();

    expect(parseInferenceResponse(raw)).toEqual(raw);
  });

  it("accepts null outputs", () => {
    const raw = rawResponse();
    raw.bioactivity.EGFR = {
      pic50: null,
      pic50_lo: null,
      pic50_hi: null,
      p_active: null,
    };
    raw.toxicity["SR-p53"] = null;

    expect(parseInferenceResponse(raw)).toEqual(raw);
  });

  it("rejects a non-object body", () => {
    expect(() => parseInferenceResponse(null)).toThrow(
      "Invalid inference response: body must be an object",
    );
  });

  it.each([
    [
      "a missing target",
      (raw) => delete raw.bioactivity.EGFR,
      "bioactivity.EGFR must be an object",
    ],
    [
      "a missing tox task",
      (raw) => delete raw.toxicity["NR-AR"],
      "toxicity.NR-AR must be a finite number or null",
    ],
    [
      "a non-finite pIC50",
      (raw) => (raw.bioactivity.BACE1!.pic50 = NaN),
      "bioactivity.BACE1.pic50 must be a finite number or null",
    ],
    [
      "a string probability",
      (raw) => (raw.toxicity.CT_TOX = "0.1"),
      "toxicity.CT_TOX must be a finite number or null",
    ],
    [
      "a probability above 1",
      (raw) => (raw.bioactivity.EGFR!.p_active = 1.5),
      "bioactivity.EGFR.p_active must be in [0, 1]",
    ],
    [
      "pIC50 outside its interval",
      (raw) => (raw.bioactivity.BACE1!.pic50_lo = 5.5),
      "bioactivity.BACE1 must be ordered as pic50_lo <= pic50 <= pic50_hi",
    ],
    [
      "an interval without pIC50",
      (raw) => (raw.bioactivity.BACE1!.pic50 = null),
      "bioactivity.BACE1.pic50_lo/pic50_hi must be null when pic50 is null",
    ],
    [
      "a missing model version",
      (raw) => (raw.model_version = ""),
      "model_version must be a non-empty string",
    ],
    [
      "a fractional n_samples",
      (raw) => (raw.n_samples = 2.5),
      "n_samples must be an integer",
    ],
    [
      "a single MC-Dropout pass",
      (raw) => (raw.n_samples = 1),
      "n_samples must be at least 2",
    ],
    [
      "coverage of 1",
      (raw) => (raw.coverage = 1),
      "coverage must be a number in (0, 1)",
    ],
  ] satisfies [
    string,
    (raw: ReturnType<typeof rawResponse>) => unknown,
    string,
  ][])("rejects %s", (_name, mutate, message) => {
    const raw = rawResponse();
    mutate(raw);

    expect(() => parseInferenceResponse(raw)).toThrow(
      `Invalid inference response: ${message}`,
    );
  });
});

describe("toPredictResponse", () => {
  it("derives IC50 in nM with swapped bounds", () => {
    const res = toPredictResponse(parseInferenceResponse(rawResponse()));

    expect(res.bioactivity.EGFR.pic50).toEqual({ value: 7, lo: 6.5, hi: 8 });
    expect(res.bioactivity.EGFR.ic50Nm).toEqual({
      value: 100,
      lo: 10,
      hi: 10 ** 2.5,
    });
    expect(res.bioactivity.EGFR.pActive).toEqual({ value: 0.9 });
  });

  it("groups Tox21 into NR/SR and hides CT_TOX", () => {
    const raw = rawResponse();
    raw.toxicity.FDA_APPROVED = 0.8;
    const res = toPredictResponse(parseInferenceResponse(raw));

    expect(Object.keys(res.toxicity.nuclearReceptor)).toEqual([...NR_TASKS]);
    expect(Object.keys(res.toxicity.stressResponse)).toEqual([...SR_TASKS]);
    expect(res.fdaApproved).toEqual({ value: 0.8 });
    expect(JSON.stringify(res)).not.toContain("CT_TOX");
    expect(res.metadata).toEqual({
      modelVersion: "mufu-test",
      mcDropoutPasses: 50,
      ciCoverage: 0.95,
    });
  });

  it("marks null outputs as unavailable with a reason", () => {
    const raw = rawResponse();
    raw.bioactivity.BACE1 = {
      pic50: null,
      pic50_lo: null,
      pic50_hi: null,
      p_active: 0.3,
    };
    raw.toxicity["NR-AhR"] = null;
    const res = toPredictResponse(parseInferenceResponse(raw));

    const unavailable = { value: null, reason: "Not returned by the model" };
    expect(res.bioactivity.BACE1).toEqual({
      pic50: unavailable,
      ic50Nm: unavailable,
      pActive: { value: 0.3 },
    });
    expect(res.toxicity.nuclearReceptor["NR-AhR"]).toEqual(unavailable);
  });
});
