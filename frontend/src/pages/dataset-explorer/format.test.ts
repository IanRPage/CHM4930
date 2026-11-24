import { describe, expect, it } from "vitest";
import { combineHistograms } from "./format.ts";

describe("pooled pIC50 histogram", () => {
  it("aligns different target bounds and preserves all observed labels", () => {
    const pooled = combineHistograms([
      { binWidth: 0.5, edges: [2.5, 3, 3.5, 4], counts: [1, 2, 3] },
      { binWidth: 0.5, edges: [3.5, 4, 4.5], counts: [4, 5] },
    ]);
    expect(pooled).toEqual({
      binWidth: 0.5,
      edges: [2.5, 3, 3.5, 4, 4.5],
      counts: [1, 2, 7, 5],
    });
    expect(pooled.counts.reduce((sum, n) => sum + n, 0)).toBe(15);
  });
});
