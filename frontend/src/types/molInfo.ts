// mirrors backend/src/models/molInfo.ts

export interface MolInfo {
  canonicalSmiles: string;
  inchiKey: string | null;
  formula: string;
  molecularWeight: number;
  exactMass: number;
  logP: number;
  tpsa: number;
  molarRefractivity: number;
  hBondDonors: number;
  hBondAcceptors: number;
  rotatableBonds: number;
  heavyAtoms: number;
  aromaticRings: number;
  fractionCsp3: number;
  lipinskiViolations: number;
  svg: string;
  molblock: string | null;
}
