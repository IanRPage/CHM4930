import express from "express";
import request from "supertest";
import { describe, expect, it, vi } from "vitest";

import { createApp } from "../../src/app.js";
import { errorHandler } from "../../src/middleware/errorHandler.js";

describe("errorHandler", () => {
  it("hides internal error details behind a 500", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const app = express()
      .get("/boom", () => {
        throw new Error("secret stack details");
      })
      .use(errorHandler);

    const res = await request(app).get("/boom");

    expect(res.status).toBe(500);
    expect(res.body).toEqual({ error: "Internal server error" });
    expect(res.text).not.toContain("secret");
  });

  it("passes through client errors from the json parser", async () => {
    const res = await request(createApp())
      .post("/api/health")
      .set("Content-Type", "application/json")
      .send("{not json");

    expect(res.status).toBe(400);
    expect(res.body).toHaveProperty("error");
  });
});
