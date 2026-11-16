import type {
  DatasetArtifact,
  DatasetMolecule,
} from "../src/models/dataset.ts";
import { TOX_TASKS } from "../src/models/inference.ts";
import { NR_TASKS, SR_TASKS } from "../src/models/predict.ts";
import { fromKeys } from "../src/models/util.ts";

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
  const molecule: DatasetMolecule = {
    id: "mol_template",
    smiles: "CCO",
    sources: ["bace1", "tox21"],
    bioactivity: {
      BACE1: { pic50: 6, active: true },
      EGFR: { pic50: null, active: null },
    },
    tox21: { ...fromKeys([...NR_TASKS, ...SR_TASKS], () => null), "NR-AR": 0 },
    clintox: { FDA_APPROVED: null, CT_TOX: null },
    molecularWeight: 46.069,
    elementCounts: { C: 2, H: 6, O: 1 },
    atomCount: 9,
    heavyAtomCount: 3,
    wildcardAtomCount: 0,
    activity: "active",
    risk: "negative",
    riskLabelsObserved: 1,
  };
  return {
    // These tests only need a summary payload to check that it passes through.
    summary: {
      all: {
        counts: {
          total: 26,
          withBioactivityLabels: 25,
          withToxicityLabels: 25,
        },
      },
    } as DatasetArtifact["summary"],
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
