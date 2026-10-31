import request from "supertest";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { createApp } from "../../src/app.ts";
import { httpError } from "../../src/errors.ts";
import {
  parseInferenceResponse,
  toPredictResponse,
} from "../../src/models/inference.ts";
import { rawResponse } from "../fixtures.ts";

const predict = vi.hoisted(() => vi.fn());

vi.mock("../../src/services/mufuClient.ts", () => ({
  MuFuClient: class {
    predict = predict;
  },
}));

function post(body: unknown) {
  return request(createApp())
    .post("/api/predict")
    .send(body as object);
}

describe("POST /api/predict", () => {
  beforeEach(() => {
    predict.mockReset();
  });

  it("returns the prediction for trimmed smiles", async () => {
    const prediction = toPredictResponse(parseInferenceResponse(rawResponse()));
    predict.mockResolvedValue(prediction);

    const res = await post({ smiles: "  CCO  " });

    expect(res.status).toBe(200);
    expect(res.body).toEqual(prediction);
    expect(predict).toHaveBeenCalledWith("CCO");
  });

  it.each([
    ["missing", {}, "smiles must be a string"],
    ["empty", { smiles: "   " }, "smiles must not be empty"],
    ["non-string", { smiles: 42 }, "smiles must be a string"],
  ])("returns 400 for %s smiles", async (_case, body, error) => {
    const res = await post(body);

    expect(res.status).toBe(400);
    expect(res.body).toEqual({ error });
    expect(predict).not.toHaveBeenCalled();
  });

  it.each([
    [400, "Invalid SMILES"],
    [502, "Model server returned an invalid response"],
    [503, "Model server is unavailable"],
    [504, "Model server timed out"],
  ])("returns %i from the model client", async (status, error) => {
    predict.mockRejectedValue(httpError(status, error));

    const res = await post({ smiles: "CCO" });

    expect(res.status).toBe(status);
    expect(res.body).toEqual({ error });
  });
});
