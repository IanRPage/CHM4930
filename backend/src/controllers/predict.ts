import type { Request, Response } from "express";

import { env } from "../config/env.ts";
import { parsePredictRequest } from "../models/predict.ts";
import { MuFuClient } from "../services/mufuClient.ts";

export async function predictMolecule(req: Request, res: Response) {
  const { smiles } = parsePredictRequest(req.body);
  const client = new MuFuClient({
    url: env.mufuApiUrl,
    apiKey: env.mufuApiKey,
  });
  res.json(await client.predict(smiles));
}
