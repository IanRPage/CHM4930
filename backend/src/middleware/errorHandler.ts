import type { NextFunction, Request, Response } from "express";

interface HttpError {
  status: number;
  expose: boolean;
  message: string;
}

function isExposedHttpError(err: unknown): err is HttpError {
  return (
    err instanceof Error &&
    "status" in err &&
    typeof err.status === "number" &&
    "expose" in err &&
    err.expose === true
  );
}

export function errorHandler(
  err: unknown,
  _req: Request,
  res: Response,
  next: NextFunction,
) {
  if (res.headersSent) {
    next(err);
    return;
  }

  if (isExposedHttpError(err)) {
    res.status(err.status).json({ error: err.message });
    return;
  }

  console.error(err);
  res.status(500).json({ error: "Internal server error" });
}
