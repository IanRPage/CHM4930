import { readFileSync } from "node:fs";

import type {
  DatasetArtifact,
  DatasetMolecule,
} from "../src/models/dataset.ts";
import { TOX_TASKS } from "../src/models/inference.ts";

export type Json = Record<string, unknown>;

// Loosely typed so tests can corrupt it freely; the parser is what checks types.
export function rawResponse() {
  return {
    smiles: "CCO",
    model_version: "mufu-test",
    n_samples: 50,
    coverage: 0.95,
    nearest_train_tanimoto: 0.75,
    applicability_domain: "in_domain",
    bioactivity: {
      BACE1: { pic50: 5, pic50_lo: 4, pic50_hi: 6, p_active: 0.2 },
      EGFR: { pic50: 7, pic50_lo: 6.5, pic50_hi: 8, p_active: 0.9 },
    } as Record<string, Json>,
    toxicity: Object.fromEntries(TOX_TASKS.map((task) => [task, 0.1])) as Json,
  };
}

export function datasetFixture(): DatasetArtifact {
  const bundled = JSON.parse(
    readFileSync(
      new URL("../data/dataset-explorer.json", import.meta.url),
      "utf8",
    ),
  ) as DatasetArtifact;
  const molecule = bundled.molecules[0] as DatasetMolecule;
  return {
    ...bundled,
    molecules: Array.from(
      { length: 26 },
      (_, index): DatasetMolecule => ({
        ...molecule,
        id: `mol_${String(index).padStart(3, "0")}`,
        smiles: index === 0 ? "c1ccccc1" : index === 1 ? "[NH4+]" : "CCO",
        activity: index === 0 ? "inactive" : index === 1 ? "unknown" : "active",
        risk:
          index === 1 ? "unknown" : index % 2 === 0 ? "negative" : "positive",
        riskLabelsObserved: index === 1 ? 0 : 1,
      }),
    ),
  };
}
