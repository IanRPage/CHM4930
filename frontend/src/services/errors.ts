export type PredictErrorKind =
  | "invalid-smiles"
  | "busy"
  | "unavailable"
  | "timeout"
  | "server"
  | "network"
  | "invalid-response";

const MESSAGES: Record<PredictErrorKind, string> = {
  "invalid-smiles": "Invalid SMILES",
  busy: "Model server is busy, try again shortly",
  unavailable: "Model server is unavailable",
  timeout: "Prediction timed out",
  server: "Server error, try again later",
  network: "Can't reach the server, check your connection",
  "invalid-response": "Server returned an invalid response",
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

// Only 400 passes the backend message through: it's the validation text
export function errorFromStatus(status: number, message: string | undefined) {
  if (status === 400) {
    return new PredictError("invalid-smiles", message, status);
  }
  if (status === 503) {
    const kind = message && /busy/i.test(message) ? "busy" : "unavailable";
    return new PredictError(kind, undefined, status);
  }
  if (status === 504) return new PredictError("timeout", undefined, status);
  return new PredictError("server", undefined, status);
}
