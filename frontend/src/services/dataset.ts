import type {
  DatasetMoleculesResponse,
  DatasetQuery,
  DatasetSummaryResponse,
} from "../types/dataset.ts";
import { isNumber, isRecord } from "./predict.ts";
import { TARGETS } from "../types/predict.ts";

const numbers = (value: unknown, keys: string[]) =>
  isRecord(value) && keys.every((key) => isNumber(value[key]));
const nullableNumber = (value: unknown) => value === null || isNumber(value);

function distribution(value: unknown): boolean {
  return (
    isRecord(value) &&
    Array.isArray(value.histograms) &&
    value.histograms.length > 0 &&
    value.histograms.every(
      (histogram: unknown) =>
        isRecord(histogram) &&
        isNumber(histogram.binWidth) &&
        histogram.binWidth > 0 &&
        Array.isArray(histogram.counts) &&
        histogram.counts.length > 0 &&
        histogram.counts.every(isNumber) &&
        Array.isArray(histogram.edges) &&
        histogram.edges.length === histogram.counts.length + 1 &&
        histogram.edges.every(
          (edge: unknown, index: number, edges: unknown[]) =>
            isNumber(edge) &&
            (index === 0 || edge > (edges[index - 1] as number)),
        ),
    )
  );
}

// Validate only fields read by the summary views before entering the ready state.
function summaryGroup(group: unknown): boolean {
  if (!isRecord(group)) return false;
  const { bioactivity, elements, tox21, clintox, heavyAtomCounts } = group;
  return (
    numbers(group.counts, [
      "total",
      "withBioactivityLabels",
      "withToxicityLabels",
    ]) &&
    distribution(group.molecularWeight) &&
    isRecord(elements) &&
    Object.values(elements).every((element) =>
      numbers(element, [
        "moleculesContaining",
        "percentageOfMolecules",
        "averageAtomsPerMolecule",
      ]),
    ) &&
    isRecord(bioactivity) &&
    TARGETS.every((target) => {
      const entry = bioactivity[target];
      return (
        isRecord(entry) &&
        numbers(entry, ["labeled", "active", "inactive"]) &&
        distribution(entry) &&
        isRecord(entry.statistics) &&
        nullableNumber(entry.statistics.median)
      );
    }) &&
    isRecord(tox21) &&
    Object.values(tox21).every(
      (assay) =>
        isRecord(assay) &&
        nullableNumber(assay.positiveRate) &&
        (assay.family === "NR" || assay.family === "SR"),
    ) &&
    isRecord(clintox) &&
    ["FDA_APPROVED", "CT_TOX"].every((key) => {
      const label = clintox[key];
      return (
        numbers(label, ["positive", "negative", "labeled"]) &&
        isRecord(label) &&
        nullableNumber(label.positiveRate)
      );
    }) &&
    isRecord(heavyAtomCounts) &&
    nullableNumber(heavyAtomCounts.mean)
  );
}

function summaryResponse(body: unknown): boolean {
  if (!isRecord(body) || !isRecord(body.summary)) return false;
  const summary = body.summary;
  return ["all", "bace1", "egfr", "tox21", "clintox"].every((source) =>
    summaryGroup(summary[source]),
  );
}
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
