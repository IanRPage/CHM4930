import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { predictResponse as response } from "../test/fixtures.ts";
import { PredictError } from "./errors.ts";
import { NOT_RETURNED, predict } from "./predict.ts";

const fetchMock = vi.fn<typeof fetch>();

function reply(status: number, body: unknown) {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  fetchMock.mockResolvedValue(new Response(text, { status }));
}

function hang() {
  fetchMock.mockImplementation(
    (_url, init) =>
      new Promise((_resolve, reject) => {
        init!.signal!.addEventListener("abort", () =>
          reject(init!.signal!.reason),
        );
      }),
  );
}

async function predictError(promise: Promise<unknown>) {
  const err = await promise.catch((e: unknown) => e);
  expect(err).toBeInstanceOf(PredictError);
  return err as PredictError;
}

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

describe("predict", () => {
  it("posts the SMILES to the relative /api/predict URL", async () => {
    reply(200, response());
    await predict("CCO");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/predict");
    expect(init).toMatchObject({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ smiles: "CCO" }),
    });
  });

  it("returns a valid response unchanged", async () => {
    reply(200, response());
    await expect(predict("CCO")).resolves.toEqual(response());
  });

  it("treats missing or malformed fields as unavailable", async () => {
    const body = JSON.parse(JSON.stringify(response()));
    delete body.bioactivity.EGFR;
    body.bioactivity.BACE1.pic50 = { value: 4.9 };
    body.bioactivity.BACE1.pActive = { value: "0.12" };
    delete body.toxicity.nuclearReceptor["NR-AR"];
    body.toxicity.stressResponse["SR-ARE"] = { value: null };
    body.toxicity.stressResponse.extra = { value: 1 };
    body.fdaApproved = 0.91;
    body.metadata = { modelVersion: 3 };
    reply(200, body);

    const result = await predict("CCO");
    const missing = { value: null, reason: NOT_RETURNED };
    expect(result.bioactivity.EGFR).toEqual({
      pic50: missing,
      ic50Nm: missing,
      pActive: missing,
    });
    expect(result.bioactivity.BACE1.pic50).toEqual(missing);
    expect(result.bioactivity.BACE1.pActive).toEqual(missing);
    expect(result.toxicity.nuclearReceptor["NR-AR"]).toEqual(missing);
    expect(result.toxicity.stressResponse["SR-ARE"]).toEqual(missing);
    expect(result.toxicity.stressResponse).not.toHaveProperty("extra");
    expect(result.fdaApproved).toEqual(missing);
    expect(result.metadata).toEqual({
      modelVersion: null,
      mcDropoutPasses: null,
      ciCoverage: null,
    });
  });

  it.each([
    [
      400,
      { error: "smiles must not be empty" },
      "invalid-smiles",
      "smiles must not be empty. Check your SMILES string and try again.",
    ],
    [
      400,
      { error: "Invalid SMILES." },
      "invalid-smiles",
      "Invalid SMILES. Check your SMILES string and try again.",
    ],
    [
      400,
      { error: "   " },
      "invalid-smiles",
      "Invalid SMILES. Check your SMILES string and try again.",
    ],
    [
      400,
      {},
      "invalid-smiles",
      "Invalid SMILES. Check your SMILES string and try again.",
    ],
    [
      503,
      { error: "Model server is busy, try again shortly" },
      "busy",
      "The model is busy. Please wait a moment and try again.",
    ],
    [
      503,
      { error: "Model server is unavailable" },
      "unavailable",
      "The model is currently unavailable. Please try again later.",
    ],
    [
      504,
      { error: "Model server timed out" },
      "timeout",
      "The prediction took too long. Please try again.",
    ],
    [
      502,
      { error: "Model server returned an invalid response" },
      "server",
      "We couldn't complete the prediction. Please try again later.",
    ],
    [
      500,
      "<html>oops</html>",
      "server",
      "We couldn't complete the prediction. Please try again later.",
    ],
    [
      503,
      { error: "Model is BUSY" },
      "busy",
      "The model is busy. Please wait a moment and try again.",
    ],
    [
      503,
      {},
      "unavailable",
      "The model is currently unavailable. Please try again later.",
    ],
  ])(
    "maps a %i response to a typed error",
    async (status, body, kind, message) => {
      reply(status, body);
      const err = await predictError(predict("CCO"));
      expect(err).toMatchObject({ kind, status, message });
    },
  );

  it.each([
    ["non-JSON", "not json"],
    ["non-object", [1, 2]],
  ])("rejects a %s success body", async (_name, body) => {
    reply(200, body);
    const err = await predictError(predict("CCO"));
    expect(err).toMatchObject({
      kind: "invalid-response",
      message: "We couldn't read the prediction. Please try again.",
    });
  });

  it("maps a failed request to a network error", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const err = await predictError(predict("CCO"));
    expect(err).toMatchObject({
      kind: "network",
      message: "Can't reach the server. Check your connection and try again.",
    });
  });

  it("times out on its own", async () => {
    hang();
    const err = await predictError(predict("CCO", { timeoutMs: 10 }));
    expect(err).toMatchObject({
      kind: "timeout",
      message: "The prediction took too long. Please try again.",
    });
  });

  it("rethrows caller aborts as is", async () => {
    hang();
    const controller = new AbortController();
    const promise = predict("CCO", { signal: controller.signal });
    controller.abort();
    await expect(promise).rejects.toMatchObject({ name: "AbortError" });
  });
});
