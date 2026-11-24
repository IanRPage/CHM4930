// Type-only imports keep the frontend aligned with the existing API contract.
export type {
  ActivityFilter,
  RiskFilter,
  DatasetHistogram,
  DatasetStatistics,
  DatasetMolecule,
  DatasetMoleculesResponse,
  DatasetQuery,
  DatasetSummaryGroup,
  DatasetSummaryResponse,
} from "../../../backend/src/models/dataset.ts";

import type { DatasetSummaryResponse } from "../../../backend/src/models/dataset.ts";
export type SourceFilter = keyof DatasetSummaryResponse["summary"];
