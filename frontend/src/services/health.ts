export const DEFAULT_MODEL_VERSION = "mufu-v1.0.0";

export async function fetchModelVersion(
  signal?: AbortSignal,
): Promise<string | null> {
  try {
    const res = await fetch("/api/health", { signal });
    if (!res.ok) return null;
    const body: unknown = await res.json();
    if (
      typeof body !== "object" ||
      body === null ||
      !("modelVersion" in body) ||
      typeof body.modelVersion !== "string" ||
      !body.modelVersion
    ) {
      return null;
    }
    return body.modelVersion.startsWith("mufu")
      ? body.modelVersion
      : `mufu-${body.modelVersion}`;
  } catch {
    return null;
  }
}
