import request from "supertest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createApp } from "../../src/app.ts";
import type {
  DatasetArtifact,
  DatasetMoleculesResponse,
} from "../../src/models/dataset.ts";
import { datasetFixture } from "../fixtures.ts";

const loadDataset = vi.hoisted(() => vi.fn<() => Promise<DatasetArtifact>>());
vi.mock("../../src/services/dataset.ts", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../src/services/dataset.ts")>()),
  loadDataset,
}));

const artifact = datasetFixture();
const app = createApp();

beforeEach(() => {
  loadDataset.mockReset().mockResolvedValue(artifact);
});
afterEach(() => {
  vi.restoreAllMocks();
});

function get(query: Record<string, string> = {}) {
  return request(app).get("/api/dataset/molecules").query(query);
}

describe("GET /api/dataset/summary", () => {
  it("returns only the stored summaries", async () => {
    const response = await request(app).get("/api/dataset/summary");
    expect(response.status).toBe(200);
    expect(response.body).toEqual({
      summary: artifact.summary,
    });
    expect(response.headers["cache-control"]).toBeUndefined();
  });
});

describe("GET /api/dataset/molecules", () => {
  it("pages in stable ID order with 24 items per page", async () => {
    const first = await get();
    const second = await get({ page: "2" });
    expect(first.status).toBe(200);
    expect(first.body).toEqual({
      items: artifact.molecules.slice(0, 24),
      page: 1,
      pageSize: 24,
      total: 26,
      totalPages: 2,
      hasMore: true,
    });
    expect(second.body).toEqual({
      items: artifact.molecules.slice(24),
      page: 2,
      pageSize: 24,
      total: 26,
      totalPages: 2,
      hasMore: false,
    });
  });

  it.each(["3", String(Number.MAX_SAFE_INTEGER)])(
    "returns an empty page beyond the results: %s",
    async (page) => {
      const response = await get({ page });
      expect(response.status).toBe(200);
      expect(response.body).toMatchObject({
        items: [],
        total: 26,
        totalPages: 2,
        hasMore: false,
      });
    },
  );

  it.each([
    ["mol_001", ["mol_001"]],
    ["c1", ["mol_000"]],
    ["[NH4+]", ["mol_001"]],
    [".*", []],
    ["cco", []],
    [" CCO", []],
  ])(
    "searches ID/SMILES literally and case-sensitively: %s",
    async (q, ids) => {
      const response = await get({ q });
      const body = response.body as DatasetMoleculesResponse;
      expect(response.status).toBe(200);
      expect(body.items.map((item) => item.id)).toEqual(ids);
      expect(body.total).toBe(ids.length);
    },
  );

  it("combines search, activity, and risk before pagination", async () => {
    const response = await get({
      q: "CCO",
      activity: "active",
      risk: "negative",
    });
    expect(response.status).toBe(200);
    expect(response.body).toMatchObject({
      items: artifact.molecules.filter(
        (_, index) => index >= 2 && index % 2 === 0,
      ),
      total: 12,
      totalPages: 1,
      hasMore: false,
    });
  });

  it("exposes unknown labels and their coverage", async () => {
    const response = await get({ activity: "unknown", risk: "unknown" });
    expect(response.status).toBe(200);
    expect(response.body).toMatchObject({
      items: [
        expect.objectContaining({ id: "mol_001", riskLabelsObserved: 0 }),
      ],
      total: 1,
    });
  });

  it("returns zero pages when no records match", async () => {
    const response = await get({ activity: "inactive", risk: "positive" });
    expect(response.status).toBe(200);
    expect(response.body).toMatchObject({
      items: [],
      total: 0,
      totalPages: 0,
      hasMore: false,
    });
  });

  it.each([
    "page=0",
    "page=-1",
    "page=1.5",
    "page=1e2",
    "page=abc",
    "page=",
    "page=%20",
    "page=9007199254740992",
    "page=1&page=2",
    "q=CC&q=CO",
    "activity=invalid",
    "activity=",
    "activity=active&activity=inactive",
    "risk=high",
    "risk=",
    "risk=positive&risk=negative",
    "limit=24",
    "q[x]=CC",
  ])("returns 400 for invalid parameters: %s", async (query) => {
    const response = await request(app).get(`/api/dataset/molecules?${query}`);
    expect(response.status).toBe(400);
    expect(response.body).toEqual({ error: expect.any(String) as string });
    expect(loadDataset).not.toHaveBeenCalled();
  });

  it("uses the existing generic 500 response when the artifact cannot load", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => undefined);
    loadDataset.mockRejectedValue(new Error("missing internal artifact path"));
    const response = await get();
    expect(response.status).toBe(500);
    expect(response.body).toEqual({ error: "Internal server error" });
    expect(log).toHaveBeenCalled();
  });
});
