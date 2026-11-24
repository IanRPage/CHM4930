import { afterEach, describe, expect, it, vi } from "vitest";
import {
  datasetMoleculesResponse,
  datasetSummaryResponse,
} from "../test/datasetFixtures.ts";
import { getDatasetMolecules, getDatasetSummary } from "./dataset.ts";

afterEach(() => vi.unstubAllGlobals());

describe("dataset API", () => {
  it("reads source summaries and passes cancellation through", async () => {
    const body = datasetSummaryResponse();
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ ok: true, json: async () => body });
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();
    expect(await getDatasetSummary(controller.signal)).toEqual(body);
    expect(fetchMock).toHaveBeenCalledWith("/api/dataset/summary", {
      signal: expect.any(AbortSignal),
    });
    controller.abort();
    expect(fetchMock.mock.calls[0][1].signal.aborted).toBe(true);
  });

  it("encodes SMILES and forwards filters and page to the API", async () => {
    const body = datasetMoleculesResponse(2);
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ ok: true, json: async () => body });
    vi.stubGlobal("fetch", fetchMock);
    expect(
      await getDatasetMolecules({
        q: "[NH4+]",
        activity: "unknown",
        risk: "positive",
        page: 2,
      }),
    ).toEqual(body);
    const url = new URL(fetchMock.mock.calls[0][0], "http://localhost");
    expect(Object.fromEntries(url.searchParams)).toEqual({
      q: "[NH4+]",
      activity: "unknown",
      risk: "positive",
      page: "2",
    });
  });

  it("rejects a failed HTTP response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false }));
    await expect(getDatasetSummary()).rejects.toThrow("Couldn't load dataset");
  });

  it("rejects a summary response missing a source group", async () => {
    const body = datasetSummaryResponse();
    delete (body.summary as Partial<typeof body.summary>).tox21;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => body }),
    );
    await expect(getDatasetSummary()).rejects.toThrow(
      "summary response is invalid",
    );
  });

  it("rejects empty source groups before rendering", async () => {
    const summary = Object.fromEntries(
      ["all", "bace1", "egfr", "tox21", "clintox"].map((source) => [
        source,
        {},
      ]),
    );
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => ({ summary }) }),
    );
    await expect(getDatasetSummary()).rejects.toThrow(
      "summary response is invalid",
    );
  });

  it.each([
    ["counts", {}],
    ["elements", { C: {} }],
    ["molecularWeight", { histograms: [] }],
    ["bioactivity", {}],
    ["tox21", { "NR-AR": {} }],
    ["clintox", {}],
    ["heavyAtomCounts", {}],
  ] as const)("rejects incomplete %s fields", async (field, value) => {
    const body = datasetSummaryResponse();
    (body.summary.all as unknown as Record<string, unknown>)[field] = value;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => body }),
    );
    await expect(getDatasetSummary()).rejects.toThrow(
      "summary response is invalid",
    );
  });

  it("rejects a malformed pagination response", async () => {
    const body = datasetMoleculesResponse();
    (body as unknown as Record<string, unknown>).total = "2";
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => body }),
    );
    await expect(
      getDatasetMolecules({ q: "", activity: "all", risk: "all", page: 1 }),
    ).rejects.toThrow("molecules response is invalid");
  });
});
