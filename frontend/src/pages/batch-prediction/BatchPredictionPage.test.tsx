import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { predictResponse } from "../../test/fixtures.ts";
import BatchPredictionPage from "./BatchPredictionPage.tsx";

const fetchMock = vi.fn<typeof fetch>();

function upload(name: string, content: string, applyAccept = true) {
  const user = userEvent.setup({ applyAccept });
  return user.upload(
    screen.getByLabelText("Choose a file"),
    new File([content], name),
  );
}

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

describe("BatchPredictionPage", () => {
  it("rejects an unsupported file type", async () => {
    render(<BatchPredictionPage />);
    await upload("molecules.pdf", "CCO", false);
    expect(screen.getByRole("alert")).toHaveTextContent("Unsupported");
    expect(
      screen.getByRole("button", { name: "Run Prediction" }),
    ).toBeDisabled();
  });

  it("runs a file through to the results table", async () => {
    fetchMock.mockImplementation(async (_url, init) => {
      const { smiles } = JSON.parse(init!.body as string) as {
        smiles: string[];
      };
      return new Response(
        JSON.stringify({
          results: smiles.map((s) => ({ ...predictResponse(), smiles: s })),
          errors: [],
        }),
      );
    });
    render(<BatchPredictionPage />);
    await upload("molecules.csv", "smiles\nCCO\nCCC\n");
    await userEvent.click(
      screen.getByRole("button", { name: "Run Prediction" }),
    );

    expect(
      await screen.findByRole("heading", { name: "Batch Results" }),
    ).toBeInTheDocument();
    expect(screen.getByText("CCO")).toBeInTheDocument();
    expect(screen.getByText("CCC")).toBeInTheDocument();
  });

  it("returns to the upload panel when cancelled", async () => {
    fetchMock.mockImplementation(
      (_url, init) =>
        new Promise((_resolve, reject) =>
          init!.signal!.addEventListener("abort", () =>
            reject(init!.signal!.reason),
          ),
        ),
    );
    render(<BatchPredictionPage />);
    await upload("molecules.txt", "CCO\n");
    await userEvent.click(
      screen.getByRole("button", { name: "Run Prediction" }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Cancel" }),
    );
    expect(screen.getByText("Drag and drop a file here")).toBeInTheDocument();
  });
});
