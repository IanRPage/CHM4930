import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchModelVersion } from "./health.ts";

const fetchMock = vi.fn<typeof fetch>();

function reply(status: number, body: unknown) {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  fetchMock.mockResolvedValue(new Response(text, { status }));
}

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

describe("fetchModelVersion", () => {
  it("fetches /api/health and returns the active model version", async () => {
    reply(200, { status: "ok", modelVersion: "mufu-v1.2.0" });

    await expect(fetchModelVersion()).resolves.toBe("mufu-v1.2.0");
    expect(fetchMock).toHaveBeenCalledWith("/api/health", {
      signal: undefined,
    });
  });

  it.each([
    ["null modelVersion", 200, { status: "ok", modelVersion: null }],
    ["empty modelVersion", 200, { status: "ok", modelVersion: "" }],
    ["non-object body", 200, "not-json"],
    ["server error", 500, { error: "down" }],
  ])("returns null on %s", async (_name, status, body) => {
    reply(status, body);

    await expect(fetchModelVersion()).resolves.toBeNull();
  });

  it("returns null when the request fails", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(fetchModelVersion()).resolves.toBeNull();
  });
});
