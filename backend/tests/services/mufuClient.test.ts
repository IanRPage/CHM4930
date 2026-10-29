import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  parseInferenceResponse,
  toPredictResponse,
} from "../../src/models/inference.ts";
import { MuFuClient } from "../../src/services/mufuClient.ts";
import { rawResponse } from "../fixtures.ts";

const BASE_URL = "http://mufu.internal:8000";
const API_KEY = "test-key-123";
const UPSTREAM_BODY = "secret upstream detail";

const INVALID_SMILES = "Invalid SMILES";
const BUSY = "Model server is busy, try again shortly";
const UNAVAILABLE = "Model server is unavailable";
const TIMED_OUT = "Model server timed out";
const BAD_RESPONSE = "Model server returned an invalid response";

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
  });
}

function mockFetch() {
  return vi.spyOn(globalThis, "fetch");
}

function client() {
  return new MuFuClient({ url: BASE_URL, apiKey: API_KEY });
}

async function rejection(promise: Promise<unknown>) {
  try {
    await promise;
  } catch (err) {
    return err;
  }
  throw new Error("expected predict to reject");
}

function expectSafeMessage(err: unknown) {
  expect(err).toBeInstanceOf(Error);
  const { message } = err as Error;
  for (const secret of [BASE_URL, API_KEY, UPSTREAM_BODY]) {
    expect(message).not.toContain(secret);
  }
}

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("MuFuClient", () => {
  it.each([
    ["no URL", undefined, API_KEY],
    ["an empty URL", "", API_KEY],
    ["no API key", BASE_URL, undefined],
  ])("rejects construction with %s", (_name, url, apiKey) => {
    expect(() => new MuFuClient({ url, apiKey })).toThrow(
      "MUFU_API_URL and MUFU_API_KEY must be set",
    );
  });
});

describe("MuFuClient.predict", () => {
  it("posts the SMILES with the API key and maps the response", async () => {
    const raw = rawResponse();
    raw.toxicity["SR-p53"] = null;
    const fetchMock = mockFetch().mockResolvedValue(jsonResponse(raw));

    const res = await new MuFuClient({
      url: `${BASE_URL}/`,
      apiKey: API_KEY,
    }).predict("CCO");

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(`${BASE_URL}/predict`);
    expect(init).toMatchObject({
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
      body: JSON.stringify({ smiles: "CCO" }),
    });
    expect(init?.signal).toBeInstanceOf(AbortSignal);
    expect(res).toEqual(toPredictResponse(parseInferenceResponse(raw)));
    expect(res.toxicity.stressResponse["SR-p53"]).toEqual({
      value: null,
      reason: "Not returned by the model",
    });
  });

  it.each([
    [422, 400, INVALID_SMILES],
    [429, 503, BUSY],
    [503, 503, BUSY],
    [502, 503, UNAVAILABLE],
    [504, 503, UNAVAILABLE],
    [530, 503, UNAVAILABLE],
    [500, 502, BAD_RESPONSE],
    [401, 502, BAD_RESPONSE],
  ])("maps upstream %i to %i", async (upstream, status, message) => {
    mockFetch().mockResolvedValue(
      new Response(UPSTREAM_BODY, { status: upstream }),
    );

    const err = await rejection(client().predict("CCO"));

    expect(err).toMatchObject({ status, expose: true, message });
    expectSafeMessage(err);
  });

  it("maps an unreachable server to 503", async () => {
    mockFetch().mockRejectedValue(
      new TypeError("fetch failed", {
        cause: new Error(`connect ECONNREFUSED ${BASE_URL}`),
      }),
    );

    const err = await rejection(client().predict("CCO"));

    expect(err).toMatchObject({
      status: 503,
      expose: true,
      message: UNAVAILABLE,
    });
    expectSafeMessage(err);
  });

  it("maps a timeout to 504", async () => {
    mockFetch().mockRejectedValue(
      new DOMException(`timed out calling ${BASE_URL}`, "TimeoutError"),
    );

    const err = await rejection(client().predict("CCO"));

    expect(err).toMatchObject({
      status: 504,
      expose: true,
      message: TIMED_OUT,
    });
    expectSafeMessage(err);
  });

  it("maps a timeout while reading the body to 504", async () => {
    const res = jsonResponse(rawResponse());
    vi.spyOn(res, "json").mockRejectedValue(
      new DOMException("", "TimeoutError"),
    );
    mockFetch().mockResolvedValue(res);

    await expect(client().predict("CCO")).rejects.toMatchObject({
      status: 504,
    });
  });

  it("maps a malformed response to 502 and logs only the validator message", async () => {
    mockFetch().mockResolvedValue(
      jsonResponse({ ...rawResponse(), n_samples: 1 }),
    );

    const err = await rejection(client().predict("CCO"));

    expect(err).toMatchObject({
      status: 502,
      expose: true,
      message: BAD_RESPONSE,
    });
    expectSafeMessage(err);
    expect(console.error).toHaveBeenCalledWith(
      expect.stringContaining("n_samples must be at least 2"),
    );
  });

  it("maps a non-JSON body to 502", async () => {
    mockFetch().mockResolvedValue(
      new Response(`<html>${UPSTREAM_BODY}</html>`),
    );

    const err = await rejection(client().predict("CCO"));

    expect(err).toMatchObject({
      status: 502,
      expose: true,
      message: BAD_RESPONSE,
    });
    expectSafeMessage(err);
    expect(console.error).not.toHaveBeenCalledWith(
      expect.stringContaining(UPSTREAM_BODY),
    );
  });
});
