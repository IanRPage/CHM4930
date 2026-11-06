import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { molInfoResponse } from "../test/fixtures.ts";
import { PredictError } from "./errors.ts";
import { getMolInfo } from "./molInfo.ts";

const fetchMock = vi.fn<typeof fetch>();

function reply(status: number, body: unknown) {
  fetchMock.mockResolvedValue(new Response(JSON.stringify(body), { status }));
}

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

it("fetches mol-info with the SMILES URL-encoded", async () => {
  reply(200, molInfoResponse());

  await expect(getMolInfo("C#C+")).resolves.toEqual(molInfoResponse());
  expect(fetchMock.mock.calls[0][0]).toBe("/api/mol-info?smiles=C%23C%2B");
});

it("maps a 400 to an invalid-smiles error", async () => {
  reply(400, { error: "Invalid SMILES" });

  await expect(getMolInfo("xx")).rejects.toMatchObject({
    kind: "invalid-smiles",
    status: 400,
  });
});

it("rejects a response with a missing field", async () => {
  const body: Partial<ReturnType<typeof molInfoResponse>> = molInfoResponse();
  delete body.logP;
  reply(200, body);

  const err = await getMolInfo("CCO").catch((e: unknown) => e);
  expect(err).toBeInstanceOf(PredictError);
  expect(err).toMatchObject({ kind: "invalid-response" });
});
