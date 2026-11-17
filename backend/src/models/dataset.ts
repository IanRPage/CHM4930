import { httpError } from "../errors.ts";
import type { NrTask, SrTask, Target } from "./predict.ts";

const ACTIVITIES = ["all", "active", "inactive", "unknown"] as const;
const RISKS = ["all", "positive", "negative", "unknown"] as const;
export const DATASET_PAGE_SIZE = 24;

export type DatasetSource = "bace1" | "egfr" | "tox21" | "clintox";
export type ActivityFilter = (typeof ACTIVITIES)[number];
export type RiskFilter = (typeof RISKS)[number];
type ClinToxLabel = "FDA_APPROVED" | "CT_TOX";
type BinaryLabel = 0 | 1 | null;

export interface DatasetStatistics {
  count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  median: number | null;
  p25: number | null;
  p75: number | null;
}

export interface DatasetHistogram {
  binWidth: number;
  edges: number[];
  counts: number[];
}

interface Distribution {
  statistics: DatasetStatistics;
  histograms: DatasetHistogram[];
}

interface BinarySummary {
  positive: number;
  negative: number;
  labeled: number;
  missing: number;
  positiveRate: number | null;
}

export interface DatasetSummaryGroup {
  counts: {
    total: number;
    withBioactivityLabels: number;
    withToxicityLabels: number;
  };
  bioactivity: Record<
    Target,
    Distribution & {
      labeled: number;
      missing: number;
      active: number;
      inactive: number;
    }
  >;
  tox21: Record<NrTask | SrTask, BinarySummary & { family: "NR" | "SR" }>;
  clintox: Record<ClinToxLabel, BinarySummary>;
  elements: Record<
    string,
    {
      moleculesContaining: number;
      percentageOfMolecules: number;
      totalAtoms: number;
      averageAtomsPerMolecule: number;
    }
  >;
  atomCounts: DatasetStatistics;
  heavyAtomCounts: DatasetStatistics;
  wildcardAtomCounts: DatasetStatistics;
  moleculesWithWildcardAtoms: number;
  molecularWeight: Distribution;
}

export interface DatasetMolecule {
  id: string;
  smiles: string;
  sources: DatasetSource[];
  bioactivity: Record<Target, { pic50: number | null; active: boolean | null }>;
  tox21: Record<NrTask | SrTask, BinaryLabel>;
  clintox: Record<ClinToxLabel, BinaryLabel>;
  molecularWeight: number;
  elementCounts: Record<string, number>;
  atomCount: number;
  heavyAtomCount: number;
  wildcardAtomCount: number;
  activity: Exclude<ActivityFilter, "all">;
  risk: Exclude<RiskFilter, "all">;
  riskLabelsObserved: number;
}

export interface DatasetArtifact {
  summary: Record<"all" | DatasetSource, DatasetSummaryGroup>;
  molecules: DatasetMolecule[];
}

export type DatasetSummaryResponse = Pick<DatasetArtifact, "summary">;

export interface DatasetQuery {
  q: string;
  activity: ActivityFilter;
  risk: RiskFilter;
  page: number;
}

export interface DatasetMoleculesResponse {
  items: DatasetMolecule[];
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
  hasMore: boolean;
}

function queryString(value: unknown, name: string, fallback: string): string {
  if (value === undefined) return fallback;
  if (typeof value !== "string") {
    throw httpError(400, `${name} must be a single string`);
  }
  return value;
}

function filterValue<T extends string>(
  value: string,
  name: string,
  choices: readonly T[],
): T {
  const choice = choices.find((choice) => choice === value);
  if (choice === undefined) {
    throw httpError(400, `${name} must be one of: ${choices.join(", ")}`);
  }
  return choice;
}

export function parseDatasetQuery(
  query: Record<string, unknown>,
): DatasetQuery {
  for (const name of Object.keys(query)) {
    if (!["q", "activity", "risk", "page"].includes(name)) {
      throw httpError(400, `Unknown query parameter: ${name}`);
    }
  }
  const pageText = queryString(query.page, "page", "1");
  const page = Number(pageText);
  if (!/^\d+$/.test(pageText) || !Number.isSafeInteger(page) || page < 1) {
    throw httpError(400, "page must be a positive integer");
  }
  return {
    q: queryString(query.q, "q", ""),
    activity: filterValue(
      queryString(query.activity, "activity", "all"),
      "activity",
      ACTIVITIES,
    ),
    risk: filterValue(queryString(query.risk, "risk", "all"), "risk", RISKS),
    page,
  };
}
