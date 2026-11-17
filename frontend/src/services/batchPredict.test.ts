import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { predictResponse } from "../test/fixtures.ts";
import type { BatchCallbacks } from "./batchPredict.ts";
import { CHUNK_SIZE, batchPredict } from "./batchPredict.ts";

const fetchMock = vi.fn<typeof fetch>();

function chunkReply(smiles: string[], invalid: string[] = []) {
  return new Response(
    JSON.stringify({
      results: smiles.map((s) =>
        invalid.includes(s) ? null : { ...predictResponse(), smiles: s },
      ),
      errors: smiles.flatMap((s, index) =>
        invalid.includes(s) ? [{ index, smiles: s, reason: "bad" }] : [],
      ),
    }),
  );
}

function echo() {
  fetchMock.mockImplementation(async (_url, init) =>
    chunkReply(JSON.parse(init!.body as string).smiles),
  );
}

function callbacks() {
  return {
    onProgress: vi.fn(),
    onComplete: vi.fn(),
    onError: vi.fn(),
  } satisfies BatchCallbacks;
}

const rows = (n: number) => Array.from({ length: n }, (_, i) => `C${i}`);

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("batchPredict", () => {
  it("sends chunks and reports progress after each", async () => {
    echo();
    const cb = callbacks();
    await batchPredict(rows(CHUNK_SIZE + 3), cb, new AbortController().signal);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(cb.onProgress.mock.calls.map(([p]) => p.processed)).toEqual([
      CHUNK_SIZE,
      CHUNK_SIZE + 3,
    ]);
    expect(cb.onComplete.mock.calls[0][0]).toHaveLength(CHUNK_SIZE + 3);
  });

  it("predicts duplicates once and copies the result to every row", async () => {
    echo();
    const cb = callbacks();
    await batchPredict(["CCO", "CCC", "CCO"], cb, new AbortController().signal);
    expect(JSON.parse(fetchMock.mock.calls[0][1]!.body as string)).toEqual({
      smiles: ["CCO", "CCC"],
    });
    const results = cb.onComplete.mock.calls[0][0];
    expect(results[2].prediction).toEqual(results[0].prediction);
  });

  it("records rows the server rejected", async () => {
    fetchMock.mockImplementation(async (_url, init) =>
      chunkReply(JSON.parse(init!.body as string).smiles, ["bad"]),
    );
    const cb = callbacks();
    await batchPredict(["CCO", "bad"], cb, new AbortController().signal);
    expect(cb.onProgress.mock.calls[0][0].invalid).toBe(1);
    expect(cb.onComplete.mock.calls[0][0][1]).toMatchObject({
      prediction: null,
      error: "bad",
    });
  });

  it("retries a busy model, honoring Retry-After", async () => {
    vi.useFakeTimers();
    fetchMock
      .mockResolvedValueOnce(
        new Response("{}", { status: 503, headers: { "Retry-After": "7" } }),
      )
      .mockImplementation(async () => chunkReply(["CCO"]));
    const cb = callbacks();
    const run = batchPredict(["CCO"], cb, new AbortController().signal);
    await vi.advanceTimersByTimeAsync(6999);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    await run;
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(cb.onComplete).toHaveBeenCalled();
  });

  it("does not retry a 400", async () => {
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ error: "bad" }), { status: 400 }),
    );
    const cb = callbacks();
    await batchPredict(["CCO"], cb, new AbortController().signal);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(cb.onError).toHaveBeenCalled();
    expect(cb.onComplete).not.toHaveBeenCalled();
  });

  it("does not complete when the last progress callback cancels", async () => {
    echo();
    const controller = new AbortController();
    const cb = callbacks();
    cb.onProgress.mockImplementation(() => controller.abort());
    await batchPredict(["CCO"], cb, controller.signal);
    expect(cb.onComplete).not.toHaveBeenCalled();
  });

  it("stops quietly when aborted", async () => {
    const controller = new AbortController();
    fetchMock.mockImplementation(async (_url, init) => {
      controller.abort();
      return chunkReply(JSON.parse(init!.body as string).smiles);
    });
    const cb = callbacks();
    await batchPredict(rows(CHUNK_SIZE * 2), cb, controller.signal);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(cb.onError).not.toHaveBeenCalled();
    expect(cb.onComplete).not.toHaveBeenCalled();
  });
});
