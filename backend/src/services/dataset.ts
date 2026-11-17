import { readFile } from "node:fs/promises";

import {
  DATASET_PAGE_SIZE,
  type DatasetArtifact,
  type DatasetMoleculesResponse,
  type DatasetQuery,
  type DatasetSummaryResponse,
} from "../models/dataset.ts";

let dataset: Promise<DatasetArtifact> | undefined;

export function loadDataset(): Promise<DatasetArtifact> {
  dataset ??= readFile(
    new URL("../../data/dataset-explorer.json", import.meta.url),
    "utf8",
  ).then((raw) => {
    const artifact = JSON.parse(raw) as DatasetArtifact;
    artifact.molecules.sort((left, right) => left.id.localeCompare(right.id));
    return artifact;
  });
  return dataset;
}

export function datasetSummary(
  artifact: DatasetArtifact,
): DatasetSummaryResponse {
  return { summary: artifact.summary };
}

export function findMolecules(
  artifact: DatasetArtifact,
  query: DatasetQuery,
): DatasetMoleculesResponse {
  const { q, activity, risk, page } = query;
  const matches = artifact.molecules.filter(
    (molecule) =>
      (molecule.id.includes(q) || molecule.smiles.includes(q)) &&
      (activity === "all" || molecule.activity === activity) &&
      (risk === "all" || molecule.risk === risk),
  );
  const total = matches.length;
  const totalPages = Math.ceil(total / DATASET_PAGE_SIZE);
  const offset = (page - 1) * DATASET_PAGE_SIZE;
  return {
    items:
      page > totalPages
        ? []
        : matches.slice(offset, offset + DATASET_PAGE_SIZE),
    page,
    pageSize: DATASET_PAGE_SIZE,
    total,
    totalPages,
    hasMore: page < totalPages,
  };
}
