export type PredictErrorKind =
  | "invalid-smiles"
  | "busy"
  | "unavailable"
  | "timeout"
  | "server"
  | "network"
  | "invalid-response";

const MESSAGES: Record<PredictErrorKind, string> = {
  "invalid-smiles": "Invalid SMILES. Check your SMILES string and try again.",
  busy: "The model is busy. Please wait a moment and try again.",
  unavailable: "The model is currently unavailable. Please try again later.",
  timeout: "The prediction took too long. Please try again.",
  server: "We couldn't complete the prediction. Please try again later.",
  network: "Can't reach the server. Check your connection and try again.",
  "invalid-response": "We couldn't read the prediction. Please try again.",
};

export class PredictError extends Error {
  readonly kind: PredictErrorKind;
  readonly status: number | undefined;

  constructor(kind: PredictErrorKind, message?: string, status?: number) {
    super(message ?? MESSAGES[kind]);
    this.name = "PredictError";
    this.kind = kind;
    this.status = status;
  }
}

// Preserve validation details and add recovery guidance; other errors use friendly copy.
export function errorFromStatus(status: number, message: string | undefined) {
  if (status === 400) {
    const detail = message?.trim().replace(/[.!?]+$/, "");
    return new PredictError(
      "invalid-smiles",
      detail ? `${detail}. Check your SMILES string and try again.` : undefined,
      status,
    );
  }
  if (status === 503) {
    const kind = message && /busy/i.test(message) ? "busy" : "unavailable";
    return new PredictError(kind, undefined, status);
  }
  if (status === 504) return new PredictError("timeout", undefined, status);
  return new PredictError("server", undefined, status);
}
