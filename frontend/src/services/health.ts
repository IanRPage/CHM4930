export const DEFAULT_MODEL_VERSION = "mufu-v1.0.0";

export async function fetchModelVersion(
  signal?: AbortSignal,
): Promise<string | null> {
  try {
    const res = await fetch("/api/health", { signal });
    if (!res.ok) return null;
    const body: unknown = await res.json();
    return typeof body === "object" &&
      body !== null &&
      "modelVersion" in body &&
      typeof body.modelVersion === "string" &&
      body.modelVersion
      ? body.modelVersion
      : null;
  } catch {
    return null;
  }
}
