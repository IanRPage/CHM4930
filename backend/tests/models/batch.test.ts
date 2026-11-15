import { describe, expect, it } from "vitest";

import {
  MAX_CHUNK_SIZE,
  parseBatchPredictRequest,
} from "../../src/models/batch.ts";
import { SMILES_MAX_LENGTH } from "../../src/models/predict.ts";

describe("parseBatchPredictRequest", () => {
  it("trims valid SMILES and keys them by index", () => {
    const parsed = parseBatchPredictRequest({
      smiles: ["  CCO\n", "c1ccccc1"],
    });

    expect(parsed.valid).toEqual(
      new Map([
        [0, "CCO"],
        [1, "c1ccccc1"],
      ]),
    );
    expect(parsed.errors).toEqual([]);
    expect(parsed.inputLength).toBe(2);
  });

  it.each([
    ["no body", undefined, "Request body must be a JSON object"],
    ["an array body", [], "Request body must be a JSON object"],
    ["non-array smiles", { smiles: "CCO" }, "smiles must be an array"],
    ["empty smiles", { smiles: [] }, "smiles must not be empty"],
    [
      "an oversized chunk",
      { smiles: Array(MAX_CHUNK_SIZE + 1).fill("CCO") },
      `smiles must have at most ${MAX_CHUNK_SIZE} entries`,
    ],
  ])("rejects %s", (_case, body, message) => {
    expect(() => parseBatchPredictRequest(body)).toThrow(message);
  });

  it("reports invalid rows per index without failing the request", () => {
    const tooLong = "C".repeat(SMILES_MAX_LENGTH + 1);

    const parsed = parseBatchPredictRequest({
      smiles: ["CCO", "C C", "", tooLong, 42],
    });

    expect(parsed.valid).toEqual(new Map([[0, "CCO"]]));
    expect(parsed.errors).toEqual([
      { index: 1, smiles: "C C", reason: "smiles must not contain whitespace" },
      { index: 2, smiles: "", reason: "smiles must not be empty" },
      {
        index: 3,
        smiles: tooLong,
        reason: `smiles must be at most ${SMILES_MAX_LENGTH} characters`,
      },
      { index: 4, smiles: "42", reason: "smiles must be a string" },
    ]);
    expect(parsed.inputLength).toBe(5);
  });
});
