import { Router } from "express";

import {
  getDatasetMolecules,
  getDatasetSummary,
} from "../controllers/dataset.ts";

export const datasetRouter = Router();

datasetRouter.get("/summary", getDatasetSummary);
datasetRouter.get("/molecules", getDatasetMolecules);
