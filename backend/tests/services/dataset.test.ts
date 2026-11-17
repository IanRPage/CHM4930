import { readFile } from "node:fs/promises";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { datasetFixture } from "../fixtures.ts";

vi.mock("node:fs/promises", () => ({ readFile: vi.fn() }));

beforeEach(() => {
  vi.resetModules();
  vi.mocked(readFile).mockReset();
});

describe("dataset artifact loading", () => {
  it("loads once for concurrent requests and sorts records once by ID", async () => {
    const artifact = datasetFixture();
    artifact.molecules.reverse();
    vi.mocked(readFile).mockResolvedValue(JSON.stringify(artifact));
    const { loadDataset } = await import("../../src/services/dataset.ts");
    const [first, second] = await Promise.all([loadDataset(), loadDataset()]);
    expect(first).toBe(second);
    expect(await loadDataset()).toBe(first);
    expect(readFile).toHaveBeenCalledTimes(1);
    expect(readFile).toHaveBeenCalledWith(expect.any(URL), "utf8");
    expect(first.molecules.map((molecule) => molecule.id)).toEqual(
      datasetFixture().molecules.map((molecule) => molecule.id),
    );
  });

  it("propagates a file error without repeatedly loading the same missing artifact", async () => {
    const error = new Error("artifact not found");
    vi.mocked(readFile).mockRejectedValue(error);
    const { loadDataset } = await import("../../src/services/dataset.ts");
    await expect(loadDataset()).rejects.toBe(error);
    await expect(loadDataset()).rejects.toBe(error);
    expect(readFile).toHaveBeenCalledTimes(1);
  });
});
