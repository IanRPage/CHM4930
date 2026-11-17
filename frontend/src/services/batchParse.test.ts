import { describe, expect, it } from "vitest";
import { MAX_FILE_SIZE, parseSmiles, validateFile } from "./batchParse.ts";

describe("parseSmiles", () => {
  it("skips a CSV header and uses the smiles column", () => {
    const csv = "name,smiles\r\naspirin,CC(=O)O\r\n\r\nethanol,CCO\r\n";
    expect(parseSmiles(csv, "m.csv")).toEqual(["CC(=O)O", "CCO"]);
  });

  it("keeps the first CSV row when there is no header", () => {
    expect(parseSmiles("CCO\nCCC\n", "m.csv")).toEqual(["CCO", "CCC"]);
  });

  it("reads one SMILES per TXT line", () => {
    expect(parseSmiles("CCO ethanol\n\nCCC\n", "m.txt")).toEqual([
      "CCO",
      "CCC",
    ]);
  });

  it("takes the first line of each SDF record", () => {
    const sdf = "CCO\n  mol\nM  END\n$$$$\nCCC\n  mol\nM  END\n$$$$\n";
    expect(parseSmiles(sdf, "m.sdf")).toEqual(["CCO", "CCC"]);
  });
});

describe("validateFile", () => {
  it("rejects other extensions and oversize files", () => {
    expect(validateFile(new File(["x"], "m.pdf"))).toMatch(/Unsupported/);
    const big = new File(["x"], "m.csv");
    Object.defineProperty(big, "size", { value: MAX_FILE_SIZE + 1 });
    expect(validateFile(big)).toMatch(/10 MB/);
  });
});
