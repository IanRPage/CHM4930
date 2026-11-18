import request from "supertest";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { createApp } from "../../src/app.ts";
import { httpError } from "../../src/errors.ts";
import {
  parseInferenceResponse,
  toPredictResponse,
} from "../../src/models/inference.ts";
import { rawResponse } from "../fixtures.ts";

const predictBatch = vi.hoisted(() => vi.fn());

vi.mock("../../src/services/mufuClient.ts", () => ({
  MuFuClient: class {
    predictBatch = predictBatch;
  },
}));

function post(body: unknown) {
  return request(createApp())
    .post("/api/batch")
    .send(body as object);
}

const prediction = toPredictResponse(parseInferenceResponse(rawResponse()));

describe("POST /api/batch", () => {
  beforeEach(() => {
    predictBatch.mockReset();
  });

  it("returns a prediction per row", async () => {
    predictBatch.mockResolvedValue([prediction, prediction]);

    const res = await post({ smiles: ["CCO", "c1ccccc1"] });

    expect(res.status).toBe(200);
    expect(res.body).toEqual({ results: [prediction, prediction], errors: [] });
    expect(predictBatch.mock.calls[0]![0]).toEqual(["CCO", "c1ccccc1"]);
  });

  it("predicts duplicates once and fills invalid rows with null", async () => {
    predictBatch.mockResolvedValue([prediction]);

    const res = await post({ smiles: ["CCO", "C C", " CCO "] });

    expect(res.status).toBe(200);
    expect(predictBatch.mock.calls[0]![0]).toEqual(["CCO"]);
    expect(res.body).toEqual({
      results: [prediction, null, prediction],
      errors: [
        {
          index: 1,
          smiles: "C C",
          reason: "smiles must not contain whitespace",
        },
      ],
    });
  });

  it("returns 400 for an oversized chunk", async () => {
    const res = await post({ smiles: Array(11).fill("CCO") });

    expect(res.status).toBe(400);
    expect(res.body).toEqual({ error: "smiles must have at most 10 entries" });
    expect(predictBatch).not.toHaveBeenCalled();
  });

  it("records an error when a SMILES fails inference", async () => {
    predictBatch.mockResolvedValue([null]);

    const res = await post({ smiles: ["INVALID"] });

    expect(res.status).toBe(200);
    expect(res.body).toEqual({
      results: [null],
      errors: [
        {
          index: 0,
          smiles: "INVALID",
          reason: "Invalid SMILES",
        },
      ],
    });
  });

  it("returns upstream errors", async () => {
    predictBatch.mockRejectedValue(httpError(504, "Model server timed out"));

    const res = await post({ smiles: ["CCO"] });

    expect(res.status).toBe(504);
    expect(res.body).toEqual({ error: "Model server timed out" });
  });

  it("forwards Retry-After", async () => {
    predictBatch.mockRejectedValue(
      httpError(503, "Model server is busy, try again shortly", "7"),
    );

    const res = await post({ smiles: ["CCO"] });

    expect(res.status).toBe(503);
    expect(res.headers["retry-after"]).toBe("7");
  });
});
