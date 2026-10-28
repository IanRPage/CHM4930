import { describe, expect, it } from "vitest";

import {
  parsePredictRequest,
  SMILES_MAX_LENGTH,
} from "../../src/models/predict.ts";

describe("parsePredictRequest", () => {
  it("accepts a SMILES string and trims it", () => {
    expect(parsePredictRequest({ smiles: "  CCO\n" })).toEqual({
      smiles: "CCO",
    });
  });

  it.each([
    ["no body", undefined, "Request body must be a JSON object"],
    [
      "an array body",
      [{ smiles: "CCO" }],
      "Request body must be a JSON object",
    ],
    ["missing smiles", {}, "smiles must be a string"],
    ["non-string smiles", { smiles: 42 }, "smiles must be a string"],
    ["empty smiles", { smiles: "" }, "smiles must not be empty"],
    ["whitespace-only smiles", { smiles: " \t" }, "smiles must not be empty"],
    [
      "too long smiles",
      { smiles: "C".repeat(SMILES_MAX_LENGTH + 1) },
      `smiles must be at most ${SMILES_MAX_LENGTH} characters`,
    ],
    [
      "inner whitespace",
      { smiles: "CCO ethanol" },
      "smiles must not contain whitespace",
    ],
  ])("rejects %s with an exposed 400", (_name, body, message) => {
    expect(() => parsePredictRequest(body)).toThrow(
      expect.objectContaining({ status: 400, expose: true, message }),
    );
  });
});
