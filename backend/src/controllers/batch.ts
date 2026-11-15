import type { Request, Response } from "express";

import { env } from "../config/env.ts";
import { parseBatchPredictRequest } from "../models/batch.ts";
import type { PredictResponse } from "../models/predict.ts";
import { MuFuClient } from "../services/mufuClient.ts";

export async function predictBatch(req: Request, res: Response) {
  const { valid, errors, inputLength } = parseBatchPredictRequest(req.body);

  const uniqueSmiles = [...new Set(valid.values())];

  const controller = new AbortController();
  req.on("close", () => controller.abort());

  const client = new MuFuClient({
    url: env.mufuApiUrl,
    apiKey: env.mufuApiKey,
  });
  const predictions = await client.predictBatch(
    uniqueSmiles,
    controller.signal,
  );

  const predictionMap = new Map(
    uniqueSmiles.map((s, i) => [s, predictions[i]]),
  );

  const results = new Array<PredictResponse | null>(inputLength).fill(null);
  for (const [index, smiles] of valid) {
    results[index] = predictionMap.get(smiles)!;
  }

  res.json({ results, errors });
}
