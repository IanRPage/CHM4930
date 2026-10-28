export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function fromKeys<K extends string, V>(
  keys: readonly K[],
  value: (key: K) => V,
): Record<K, V> {
  return Object.fromEntries(keys.map((key) => [key, value(key)])) as Record<
    K,
    V
  >;
}
