import type { Request, Response } from "express";

import {
  parseDatasetQuery,
  type DatasetMoleculesResponse,
  type DatasetSummaryResponse,
} from "../models/dataset.ts";
import {
  datasetSummary,
  findMolecules,
  loadDataset,
} from "../services/dataset.ts";

export async function getDatasetSummary(
  req: Request,
  res: Response<DatasetSummaryResponse>,
) {
  res.json(datasetSummary(await loadDataset()));
}

export async function getDatasetMolecules(
  req: Request,
  res: Response<DatasetMoleculesResponse>,
) {
  const query = parseDatasetQuery(req.query);
  res.json(findMolecules(await loadDataset(), query));
}
