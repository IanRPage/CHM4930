import { Router } from "express";

import { healthRouter } from "./health.ts";
import { predictRouter } from "./predict.ts";

export const apiRouter = Router();

apiRouter.use("/health", healthRouter);
apiRouter.use("/predict", predictRouter);
