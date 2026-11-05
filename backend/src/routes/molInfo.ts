import { Router } from "express";

import { getMolInfo } from "../controllers/molInfo.ts";

export const molInfoRouter = Router();

molInfoRouter.get("/", getMolInfo);
