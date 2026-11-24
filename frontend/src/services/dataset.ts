import type {
  DatasetMoleculesResponse,
  DatasetQuery,
  DatasetSummaryResponse,
} from "../types/dataset.ts";
import { isNumber, isRecord } from "./predict.ts";

// Validate response envelopes; nested fields follow the shared backend types.
const summaryResponse = (body: unknown) =>
  isRecord(body) &&
  isRecord(body.summary) &&
  ["all", "bace1", "egfr", "tox21", "clintox"].every((source) =>
    isRecord((body.summary as Record<string, unknown>)[source]),
  );
const moleculesResponse = (body: unknown) =>
  isRecord(body) &&
  Array.isArray(body.items) &&
  body.items.every(isRecord) &&
  ["page", "pageSize", "total", "totalPages"].every((key) =>
    isNumber(body[key]),
  ) &&
  typeof body.hasMore === "boolean";

async function request<T>(
  endpoint: string,
  validate: (body: unknown) => boolean,
  signal?: AbortSignal,
  query = "",
): Promise<T> {
  const timeout = AbortSignal.timeout(15_000);
  const response = await fetch(`/api/dataset/${endpoint}${query}`, {
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
  });
  if (!response.ok) throw new Error("Couldn't load dataset. Please try again.");
  const body: unknown = await response.json();
  if (!validate(body))
    throw new Error(`The dataset ${endpoint} response is invalid.`);
  return body as T;
}

export function getDatasetSummary(signal?: AbortSignal) {
  return request<DatasetSummaryResponse>("summary", summaryResponse, signal);
}

export function getDatasetMolecules(query: DatasetQuery, signal?: AbortSignal) {
  const params = new URLSearchParams({ ...query, page: String(query.page) });
  return request<DatasetMoleculesResponse>(
    "molecules",
    moleculesResponse,
    signal,
    `?${params}`,
  );
}
