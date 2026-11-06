import request from "supertest";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { createApp } from "../../src/app.ts";

const modelVersion = vi.hoisted(() => vi.fn());

vi.mock("../../src/services/mufuClient.ts", () => ({
  MuFuClient: class {
    modelVersion = modelVersion;
  },
}));

describe("GET /api/health", () => {
  beforeEach(() => {
    modelVersion.mockReset();
  });

  it("returns ok with the active model version", async () => {
    modelVersion.mockResolvedValue("mufu-test");

    const res = await request(createApp()).get("/api/health");

    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: "ok", modelVersion: "mufu-test" });
  });

  it("returns ok with null modelVersion when loading or unavailable", async () => {
    modelVersion.mockRejectedValue(new Error("unconfigured"));

    const res = await request(createApp()).get("/api/health");

    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: "ok", modelVersion: null });
  });

  it("returns 404 json for unknown routes", async () => {
    const res = await request(createApp()).get("/api/nope");

    expect(res.status).toBe(404);
    expect(res.body).toEqual({ error: "Not found: GET /api/nope" });
  });
});
