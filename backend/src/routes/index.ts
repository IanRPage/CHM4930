import { Router } from "express";

import { datasetRouter } from "./dataset.ts";
import { batchRouter } from "./batch.ts";
import { healthRouter } from "./health.ts";
import { molInfoRouter } from "./molInfo.ts";
import { predictRouter } from "./predict.ts";

export const apiRouter = Router();

apiRouter.use("/dataset", datasetRouter);
apiRouter.use("/health", healthRouter);
apiRouter.use("/mol-info", molInfoRouter);
apiRouter.use("/predict", predictRouter);
apiRouter.use("/batch", batchRouter);
