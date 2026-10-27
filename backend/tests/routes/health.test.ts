import request from "supertest";
import { describe, expect, it } from "vitest";

import { createApp } from "../../src/app.ts";

describe("GET /api/health", () => {
  it("returns ok", async () => {
    const res = await request(createApp()).get("/api/health");

    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: "ok" });
  });

  it("returns 404 json for unknown routes", async () => {
    const res = await request(createApp()).get("/api/nope");

    expect(res.status).toBe(404);
    expect(res.body).toEqual({ error: "Not found: GET /api/nope" });
  });
});
