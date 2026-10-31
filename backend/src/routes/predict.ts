import { Router } from "express";

import { predictMolecule } from "../controllers/predict.ts";

export const predictRouter = Router();

predictRouter.post("/", predictMolecule);
