import { Router } from "express";

import { getHealth } from "../controllers/health.js";

export const healthRouter = Router();

healthRouter.get("/", getHealth);
