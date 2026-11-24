import type {
  DatasetMolecule,
  DatasetMoleculesResponse,
  DatasetStatistics,
  DatasetSummaryGroup,
  DatasetSummaryResponse,
} from "../types/dataset.ts";
import { NR_TASKS, SR_TASKS } from "../types/predict.ts";

const stats = (count: number): DatasetStatistics => ({
  count,
  min: count ? 1 : null,
  max: count ? 8 : null,
  mean: count ? 4.5 : null,
  median: count ? 4.5 : null,
  p25: count ? 2 : null,
  p75: count ? 7 : null,
});

export function summaryGroup(empty = false): DatasetSummaryGroup {
  const n = empty ? 0 : 2;
  const distribution = {
    statistics: stats(n),
    histograms: [0.25, 0.5, 1].map((binWidth) => ({
      binWidth,
      edges: [5, 5 + binWidth, 5 + binWidth * 2],
      counts: [n / 2, n / 2],
    })),
  };
  const binary = {
    positive: n / 2,
    negative: n / 2,
    labeled: n,
    missing: n,
    positiveRate: empty ? null : 0.5,
  };
  const target = {
    ...distribution,
    labeled: n,
    missing: n,
    active: n / 2,
    inactive: n / 2,
  };
  return {
    counts: {
      total: empty ? 0 : 4,
      withBioactivityLabels: empty ? 0 : 3,
      withToxicityLabels: n,
    },
    bioactivity: {
      BACE1: structuredClone(target),
      EGFR: structuredClone(target),
    },
    tox21: Object.fromEntries(
      [...NR_TASKS, ...SR_TASKS].map((task) => [
        task,
        { ...binary, family: task.startsWith("NR") ? "NR" : "SR" },
      ]),
    ) as DatasetSummaryGroup["tox21"],
    clintox: { FDA_APPROVED: { ...binary }, CT_TOX: { ...binary } },
    elements: empty
      ? {}
      : Object.fromEntries(
          (
            [
              ["C", 4, 8],
              ["H", 4, 24],
              ["O", 2, 2],
              ["F", 1, 1],
            ] as const
          ).map(([symbol, moleculesContaining, totalAtoms]) => [
            symbol,
            {
              moleculesContaining,
              totalAtoms,
              percentageOfMolecules: (moleculesContaining / 4) * 100,
              averageAtomsPerMolecule: totalAtoms / 4,
            },
          ]),
        ),
    atomCounts: stats(empty ? 0 : 4),
    heavyAtomCounts: stats(empty ? 0 : 4),
    wildcardAtomCounts: stats(empty ? 0 : 4),
    moleculesWithWildcardAtoms: 0,
    molecularWeight: {
      statistics: stats(empty ? 0 : 4),
      histograms: [25, 50, 100].map((binWidth) => ({
        binWidth,
        edges: [0, binWidth, binWidth * 2],
        counts: empty ? [0, 0] : [2, 2],
      })),
    },
  };
}

export function datasetSummaryResponse(): DatasetSummaryResponse {
  return {
    summary: {
      all: summaryGroup(),
      bace1: summaryGroup(),
      egfr: summaryGroup(),
      tox21: summaryGroup(true),
      clintox: summaryGroup(),
    },
  };
}

export function datasetMolecule(id = "mol_first"): DatasetMolecule {
  return {
    id,
    smiles: "CCO",
    sources: ["bace1", "tox21"],
    bioactivity: {
      BACE1: { pic50: 6, active: true },
      EGFR: { pic50: null, active: null },
    },
    tox21: Object.fromEntries(
      [...NR_TASKS, ...SR_TASKS].map((task) => [
        task,
        task === "NR-AR" ? 0 : null,
      ]),
    ) as DatasetMolecule["tox21"],
    clintox: { FDA_APPROVED: null, CT_TOX: null },
    molecularWeight: 46.069,
    elementCounts: { C: 2, H: 6, O: 1 },
    atomCount: 9,
    heavyAtomCount: 3,
    wildcardAtomCount: 0,
    activity: "active",
    risk: "negative",
    riskLabelsObserved: 1,
  };
}

export function datasetMoleculesResponse(page = 1): DatasetMoleculesResponse {
  return {
    items: [datasetMolecule(page === 1 ? "mol_first" : "mol_second")],
    page,
    pageSize: 1,
    total: 2,
    totalPages: 2,
    hasMore: page === 1,
  };
}
