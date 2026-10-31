import type { PredictResponse } from "../types/predict.ts";
import { NR_TASKS, SR_TASKS } from "../types/predict.ts";

// Example response from backend/docs/predict-api.md
export function predictResponse(): PredictResponse {
  const tox = { value: 0.1 };
  return {
    smiles: "CC(=O)Oc1ccccc1C(=O)O",
    bioactivity: {
      BACE1: {
        pic50: { value: 4.9, lo: 3.8, hi: 6 },
        ic50Nm: { value: 12589, lo: 1000, hi: 158489 },
        pActive: { value: 0.12 },
      },
      EGFR: {
        pic50: { value: null, reason: "Not returned by the model" },
        ic50Nm: { value: null, reason: "Not returned by the model" },
        pActive: { value: 0.08 },
      },
    },
    toxicity: {
      nuclearReceptor: Object.fromEntries(NR_TASKS.map((t) => [t, tox])),
      stressResponse: Object.fromEntries(SR_TASKS.map((t) => [t, tox])),
    } as PredictResponse["toxicity"],
    fdaApproved: { value: 0.91 },
    metadata: {
      modelVersion: "mufu-test",
      mcDropoutPasses: 50,
      ciCoverage: 0.95,
    },
  };
}
