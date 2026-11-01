import { TOX_TASKS } from "../src/models/inference.ts";

export type Json = Record<string, unknown>;

// Loosely typed so tests can corrupt it freely; the parser is what checks types.
export function rawResponse() {
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
