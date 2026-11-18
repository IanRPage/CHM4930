import { Router } from "express";

import { predictBatch } from "../controllers/batch.ts";

export const batchRouter = Router();

batchRouter.post("/", predictBatch);
