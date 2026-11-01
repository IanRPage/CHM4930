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
      "smiles must not be empty",
    ],
    [
      503,
      { error: "Model server is busy, try again shortly" },
      "busy",
      "Model server is busy, try again shortly",
    ],
    [
      503,
      { error: "Model server is unavailable" },
      "unavailable",
      "Model server is unavailable",
    ],
    [
      504,
      { error: "Model server timed out" },
      "timeout",
      "Prediction timed out",
    ],
    [
      502,
      { error: "Model server returned an invalid response" },
      "server",
      "Server error, try again later",
    ],
    [500, "<html>oops</html>", "server", "Server error, try again later"],
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
    expect(err.kind).toBe("invalid-response");
  });

  it("maps a failed request to a network error", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));
    const err = await predictError(predict("CCO"));
    expect(err.kind).toBe("network");
  });

  it("times out on its own", async () => {
    hang();
    const err = await predictError(predict("CCO", { timeoutMs: 10 }));
    expect(err.kind).toBe("timeout");
  });

  it("rethrows caller aborts as is", async () => {
    hang();
    const controller = new AbortController();
    const promise = predict("CCO", { signal: controller.signal });
    controller.abort();
    await expect(promise).rejects.toMatchObject({ name: "AbortError" });
  });
});
