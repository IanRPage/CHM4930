export function httpError(status: number, message: string) {
  return Object.assign(new Error(message), { status, expose: true });
}
