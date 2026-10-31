import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PredictError } from "../services/errors.ts";
import { predict } from "../services/predict.ts";
import { predictResponse } from "../test/fixtures.ts";
import SinglePredictionPage from "./SinglePredictionPage.tsx";

vi.mock("../services/predict.ts", () => ({ predict: vi.fn() }));

const predictMock = vi.mocked(predict);

async function submit(smiles: string) {
  render(<SinglePredictionPage />);
  await userEvent.type(screen.getByLabelText("Enter SMILES"), smiles);
  await userEvent.click(screen.getByRole("button", { name: "Predict" }));
}

afterEach(() => {
  predictMock.mockReset();
});

describe("SinglePredictionPage", () => {
  it("shows predictions per target, with unavailable outputs as a dash", async () => {
    const result = predictResponse();
    result.toxicity.stressResponse["SR-ARE"] = { value: 0.54 };
    result.bioactivity.BACE1.pActive = { value: 0.9 };
    predictMock.mockResolvedValue(result);
    await submit("CCO");

    expect(predictMock).toHaveBeenCalledWith("CCO", expect.anything());
    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "Prediction Results",
      }),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Not returned by the model")).toHaveLength(2);
    expect(screen.getByText("1 of 12 high risk")).toBeInTheDocument();
    expect(screen.getByText("54%")).toBeInTheDocument();
    expect(
      screen.getByText("Confidence via MC-Dropout · 50 passes"),
    ).toBeInTheDocument();
    const bioactivity = screen
      .getByRole("heading", { name: "Bioactivity" })
      .closest("section");
    expect(bioactivity).toHaveClass("predict-panel-neutral");

    await userEvent.click(screen.getByRole("button", { name: "BACE1" }));
    expect(bioactivity).toHaveClass("predict-panel-good");
    expect(screen.getByText("4.90")).toBeInTheDocument();
    expect(screen.getByText("95% CI 3.80 – 6.00")).toBeInTheDocument();
    expect(screen.getByText("12.6 µM")).toBeInTheDocument();
    expect(screen.getByText("95% CI 1 µM – 158 µM")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Input Molecule" }));
    expect(screen.getByLabelText("Enter SMILES")).toHaveValue("CCO");
  });

  it("shows the error message and stays on the input tab", async () => {
    predictMock.mockRejectedValue(new PredictError("invalid-smiles"));
    await submit("xyz");
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Invalid SMILES",
    );
    expect(
      screen.getByRole("tab", { name: "Prediction Results" }),
    ).toBeDisabled();
  });
});
