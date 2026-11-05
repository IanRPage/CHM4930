import type { Request, Response } from "express";

import { computeMolInfo } from "../models/molInfo.ts";
import { parseSmiles } from "../models/predict.ts";

export async function getMolInfo(req: Request, res: Response) {
  res.json(await computeMolInfo(parseSmiles(req.query.smiles)));
}
