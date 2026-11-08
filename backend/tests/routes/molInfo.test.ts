import request from "supertest";
import { describe, expect, it } from "vitest";

import { createApp } from "../../src/app.ts";
import { MAX_3D_HEAVY_ATOMS, type MolInfo } from "../../src/models/molInfo.ts";

function get(query: Record<string, string>) {
  return request(createApp()).get("/api/mol-info").query(query);
}

describe("GET /api/mol-info", () => {
  // first conformer per process warms up OpenChemLib for a few seconds
  it(
    "returns identifiers, descriptors and structures",
    { timeout: 30_000 },
    async () => {
      const res = await get({ smiles: "OC(=O)c1ccccc1OC(C)=O" });

      expect(res.status).toBe(200);
      expect(res.body).toMatchObject({
        canonicalSmiles: "CC(=O)Oc1ccccc1C(=O)O",
        inchiKey: "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        formula: "C9H8O4",
        hBondDonors: 1,
        hBondAcceptors: 4,
        rotatableBonds: 2,
        heavyAtoms: 13,
        aromaticRings: 1,
        lipinskiViolations: 0,
      });
      const body = res.body as MolInfo;
      expect(body.molecularWeight).toBeCloseTo(180.16, 2);
      expect(body.svg).toContain("<svg");
      const z = [
        ...(body.molblock ?? "").matchAll(/^\s*\S+\s+\S+\s+(\S+)\s+[A-Z]/gm),
      ].map((m) => Number(m[1]));
      expect(z).toHaveLength(21);
      expect(z.some((c) => c !== 0)).toBe(true);
    },
  );

  it("returns 400 for unparseable SMILES", async () => {
    const res = await get({ smiles: "C1CC" });

    expect(res.status).toBe(400);
    expect(res.body).toEqual({ error: "Invalid SMILES" });
  });

  it("omits 3D above the heavy-atom cap", async () => {
    const res = await get({ smiles: "C".repeat(MAX_3D_HEAVY_ATOMS + 1) });

    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({
      heavyAtoms: MAX_3D_HEAVY_ATOMS + 1,
      lipinskiViolations: 2,
      molblock: null,
    });
  });

  it("validates smiles like /api/predict", async () => {
    const res = await get({});

    expect(res.status).toBe(400);
    expect(res.body).toEqual({ error: "smiles must be a string" });
  });
});
