import type { Request, Response } from "express";

import { env } from "../config/env.ts";
import { MuFuClient } from "../services/mufuClient.ts";

export async function getHealth(_req: Request, res: Response) {
  let modelVersion: string | null;
  try {
    const client = new MuFuClient({
      url: env.mufuApiUrl,
      apiKey: env.mufuApiKey,
    });
    modelVersion = await client.modelVersion();
  } catch {
    modelVersion = null;
  }
  res.json({ status: "ok", modelVersion });
}
