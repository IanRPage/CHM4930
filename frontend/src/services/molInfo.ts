import type { MolInfo } from "../types/molInfo.ts";
import { PredictError, errorFromStatus } from "./errors.ts";
import { isNumber, isRecord, readJson } from "./predict.ts";

const NUMBER_FIELDS = [
  "molecularWeight",
  "exactMass",
  "logP",
  "tpsa",
  "molarRefractivity",
  "hBondDonors",
  "hBondAcceptors",
  "rotatableBonds",
  "heavyAtoms",
  "aromaticRings",
  "fractionCsp3",
  "lipinskiViolations",
] as const;
const STRING_FIELDS = ["canonicalSmiles", "formula", "svg"] as const;
const NULLABLE_STRING_FIELDS = ["inchiKey", "molblock"] as const;

function parseMolInfo(body: unknown): MolInfo {
  if (
    !isRecord(body) ||
    !NUMBER_FIELDS.every((k) => isNumber(body[k])) ||
    !STRING_FIELDS.every((k) => typeof body[k] === "string") ||
    !NULLABLE_STRING_FIELDS.every(
      (k) => body[k] === null || typeof body[k] === "string",
    )
  ) {
    throw new PredictError("invalid-response");
  }
  return body as unknown as MolInfo;
}

export async function getMolInfo(
  smiles: string,
  { signal }: { signal?: AbortSignal } = {},
): Promise<MolInfo> {
  let res: Response;
  try {
    res = await fetch(`/api/mol-info?${new URLSearchParams({ smiles })}`, {
      signal,
    });
  } catch (err) {
    if (signal?.aborted) throw err;
    throw new PredictError("network");
  }

  const body = await readJson(res);
  if (!res.ok) {
    const message =
      isRecord(body) && typeof body.error === "string" ? body.error : undefined;
    throw errorFromStatus(res.status, message);
  }
  return parseMolInfo(body);
}
