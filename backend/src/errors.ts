export function httpError(
  status: number,
  message: string,
  retryAfter?: string,
) {
  return Object.assign(new Error(message), {
    status,
    expose: true,
    retryAfter,
  });
}
