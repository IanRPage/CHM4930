import initRDKitModule, { type MainModule } from "@rdkit/rdkit";
import { ConformerGenerator, Molecule, Resources } from "openchemlib";

import { httpError } from "../errors.ts";

export const MAX_3D_HEAVY_ATOMS = 100;
const CONFORMER_SEED = 42;

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

interface Descriptors {
  CrippenClogP: number;
  tpsa: number;
  CrippenMR: number;
  lipinskiHBD: number;
  lipinskiHBA: number;
  NumRotatableBonds: number;
  NumHeavyAtoms: number;
  NumAromaticRings: number;
  FractionCSP3: number;
}

let rdkit: Promise<MainModule> | undefined;

function loadRDKit() {
  if (!rdkit) {
    Resources.registerFromNodejs();
    rdkit = initRDKitModule.default();
  }
  return rdkit;
}

export async function computeMolInfo(smiles: string): Promise<MolInfo> {
  const RDKit = await loadRDKit();
  const mol = RDKit.get_mol(smiles);
  if (!mol) {
    throw httpError(400, "Invalid SMILES");
  }
  try {
    const d = JSON.parse(mol.get_descriptors()) as Descriptors;
    const inchi = mol.get_inchi();
    const ocl = Molecule.fromMolfile(mol.get_molblock());
    const { formula, relativeWeight, absoluteWeight } =
      ocl.getMolecularFormula();
    const conformer =
      d.NumHeavyAtoms > MAX_3D_HEAVY_ATOMS
        ? null
        : new ConformerGenerator(CONFORMER_SEED).getOneConformerAsMolecule(ocl);

    return {
      canonicalSmiles: mol.get_smiles(),
      inchiKey: inchi ? RDKit.get_inchikey_for_inchi(inchi) : null,
      formula,
      molecularWeight: relativeWeight,
      exactMass: absoluteWeight,
      logP: d.CrippenClogP,
      tpsa: d.tpsa,
      molarRefractivity: d.CrippenMR,
      hBondDonors: d.lipinskiHBD,
      hBondAcceptors: d.lipinskiHBA,
      rotatableBonds: d.NumRotatableBonds,
      heavyAtoms: d.NumHeavyAtoms,
      aromaticRings: d.NumAromaticRings,
      fractionCsp3: d.FractionCSP3,
      lipinskiViolations: [
        relativeWeight > 500,
        d.CrippenClogP > 5,
        d.lipinskiHBD > 5,
        d.lipinskiHBA > 10,
      ].filter(Boolean).length,
      svg: mol.get_svg(),
      molblock: conformer?.toMolfile() ?? null,
    };
  } finally {
    mol.delete();
  }
}
