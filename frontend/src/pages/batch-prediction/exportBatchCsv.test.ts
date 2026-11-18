import { describe, expect, it } from "vitest";
import { predictResponse } from "../../test/fixtures.ts";
import { batchToCsv } from "./exportBatchCsv.ts";

describe("batchToCsv", () => {
  it("writes the selected target, a CI half-width, and blanks for invalid rows", () => {
    const csv = batchToCsv(
      [
        {
          row: 1,
          smiles: "CCO",
          prediction: predictResponse(),
          error: null,
        },
        {
          row: 2,
          smiles: "=bad",
          prediction: null,
          error: "Invalid SMILES",
        },
      ],
      "BACE1",
    );
    const [header, ok, bad] = csv.trimEnd().split("\r\n");
    expect(header.split(",")).toHaveLength(20);
    expect(ok.split(",").slice(0, 6)).toEqual([
      "1",
      "CCO",
      "4.9",
      "1.1",
      "12589",
      "0.12",
    ]);
    expect(bad).toMatch(/^2,'=bad,,/);
    expect(bad.endsWith(",Invalid SMILES")).toBe(true);
  });
});
