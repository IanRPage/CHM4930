import { Router } from "express";

import { getHealth } from "../controllers/health.ts";

export const healthRouter = Router();

healthRouter.get("/", getHealth);
