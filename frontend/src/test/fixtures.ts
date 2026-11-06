import type { MolInfo } from "../types/molInfo.ts";
import type { PredictResponse } from "../types/predict.ts";
import { NR_TASKS, SR_TASKS } from "../types/predict.ts";

export function predictResponse(): PredictResponse {
  const tox = { value: 0.1 };
  return {
    smiles: "CC(=O)Oc1ccccc1C(=O)O",
    nearestTrainTanimoto: 0.72,
    applicabilityDomain: "in_domain",
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

export function molInfoResponse(): MolInfo {
  return {
    canonicalSmiles: "CC(=O)Oc1ccccc1C(=O)O",
    inchiKey: "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
    formula: "C9H8O4",
    molecularWeight: 180.159,
    exactMass: 180.042259,
    logP: 1.3101,
    tpsa: 63.6,
    molarRefractivity: 44.71,
    hBondDonors: 1,
    hBondAcceptors: 3,
    rotatableBonds: 2,
    heavyAtoms: 13,
    aromaticRings: 1,
    fractionCsp3: 0.1111,
    lipinskiViolations: 0,
    svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"></svg>',
    molblock:
      "aspirin\n  RDKit          3D\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END",
  };
}
