import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PredictError } from "../../services/errors.ts";
import { predict } from "../../services/predict.ts";
import { predictResponse } from "../../test/fixtures.ts";
import SinglePredictionPage from "./SinglePredictionPage.tsx";

vi.mock("../../services/predict.ts", () => ({ predict: vi.fn() }));

const predictMock = vi.mocked(predict);

function pendingPrediction() {
  let resolve!: (result: Awaited<ReturnType<typeof predict>>) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<Awaited<ReturnType<typeof predict>>>(
    (res, rej) => {
      resolve = res;
      reject = rej;
    },
  );
  return { promise, resolve, reject };
}

function expectNoResults() {
  expect(
    screen.getByRole("tab", { name: "Prediction Results" }),
  ).toBeDisabled();
  expect(
    screen.getByRole("tab", { name: "Molecular Information" }),
  ).toBeDisabled();
  expect(
    screen.queryByRole("heading", { name: "Predicted Properties" }),
  ).not.toBeInTheDocument();
}

async function submit(smiles: string) {
  render(<SinglePredictionPage />);
  await userEvent.type(screen.getByLabelText("Enter SMILES"), smiles);
  await userEvent.click(screen.getByRole("button", { name: "Predict" }));
}

afterEach(() => {
  predictMock.mockReset();
});

describe("SinglePredictionPage", () => {
  it("announces loading and blocks overlapping submissions, including before rerender", async () => {
    const pending = pendingPrediction();
    predictMock.mockReturnValue(pending.promise);
    render(<SinglePredictionPage />);
    const status = screen.getByRole("status");
    expect(status).toBeEmptyDOMElement();
    const input = screen.getByRole("textbox", { name: "Enter SMILES" });
    await userEvent.type(input, "CCO");

    // Submit twice in the same render to exercise the guard behind the button.
    act(() => {
      fireEvent.submit(input.closest("form")!);
      fireEvent.submit(input.closest("form")!);
    });

    expect(status).toHaveAttribute("aria-live", "polite");
    expect(status).toHaveAttribute("aria-atomic", "true");
    expect(status).toHaveTextContent(
      "Predicting molecule properties. Please wait…",
    );
    const button = screen.getByRole("button", { name: "Predicting…" });
    expect(button).toBeDisabled();
    expect(screen.getByRole("button", { name: "Clear" })).toBeDisabled();
    expect(input).toBeEnabled();
    expectNoResults();
    await userEvent.dblClick(button);
    await userEvent.type(input, "{Enter}");
    expect(predictMock).toHaveBeenCalledTimes(1);
    expect(predictMock.mock.calls[0][1]?.signal?.aborted).toBe(false);

    await act(async () => pending.resolve(predictResponse()));
    expect(
      screen.getByRole("heading", { level: 1, name: "Prediction Results" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("validates blank input accessibly without sending a request", async () => {
    render(<SinglePredictionPage />);
    const input = screen.getByRole("textbox", { name: "Enter SMILES" });
    await userEvent.type(input, "   ");
    await userEvent.click(screen.getByRole("button", { name: "Predict" }));

    expect(predictMock).not.toHaveBeenCalled();
    const message = "Enter a SMILES string to predict a molecule.";
    expect(input).toBeInvalid();
    expect(input).toHaveFocus();
    expect(input).toHaveAccessibleDescription(message);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    // Repeated submission keeps focus on the input with a single message.
    await userEvent.click(screen.getByRole("button", { name: "Predict" }));
    expect(input).toHaveFocus();
    expect(screen.getAllByText(message)).toHaveLength(1);

    await userEvent.clear(input);
    await userEvent.type(input, " CCO ");
    expect(input).toBeValid();
    expect(input).not.toHaveAttribute("aria-describedby");
    predictMock.mockResolvedValue(predictResponse());
    await userEvent.click(screen.getByRole("button", { name: "Predict" }));
    expect(predictMock).toHaveBeenCalledWith("CCO", expect.anything());
  });

  it("focuses the results tab after keyboard submission without overriding manual navigation", async () => {
    const user = userEvent.setup();
    render(<SinglePredictionPage />);
    const inputTab = screen.getByRole("tab", { name: "Input Molecule" });
    const resultsTab = screen.getByRole("tab", { name: "Prediction Results" });
    const infoTab = screen.getByRole("tab", { name: "Molecular Information" });

    predictMock.mockResolvedValue(predictResponse());
    await user.click(screen.getByRole("textbox", { name: "Enter SMILES" }));
    await user.keyboard("CCO{Enter}");
    expect(resultsTab).toHaveFocus();
    expect(resultsTab).toHaveAttribute("aria-selected", "true");

    await user.click(infoTab);
    expect(infoTab).toHaveAttribute("aria-selected", "true");
    await user.click(inputTab);
    expect(inputTab).toHaveAttribute("aria-selected", "true");
    await user.click(resultsTab);
    expect(resultsTab).toHaveAttribute("aria-selected", "true");
  });

  it.each([
    [
      new PredictError("invalid-smiles"),
      "Invalid SMILES. Check your SMILES string and try again.",
      "true",
    ],
    [
      new PredictError("busy"),
      "The model is busy. Please wait a moment and try again.",
      "false",
    ],
    [
      new Error("Internal diagnostic details"),
      "Something went wrong. Please try again.",
      "false",
    ],
  ])(
    "shows alert error copy for %s and recovers on retry",
    async (error, message, ariaInvalid) => {
      predictMock
        .mockRejectedValueOnce(error)
        .mockResolvedValueOnce(predictResponse());
      await submit("CCO");

      expect(await screen.findByRole("alert")).toHaveTextContent(message);
      const input = screen.getByRole("textbox", { name: "Enter SMILES" });
      expect(input).toHaveAccessibleDescription(message);
      expect(input).toHaveAttribute("aria-invalid", ariaInvalid);
      expectNoResults();

      await userEvent.click(screen.getByRole("button", { name: "Predict" }));
      expect(
        await screen.findByRole("heading", {
          level: 1,
          name: "Prediction Results",
        }),
      ).toBeInTheDocument();
    },
  );

  it.each(["resolve", "reject"] as const)(
    "cancels an in-flight request on input change and ignores its late %s",
    async (settlement) => {
      const first = pendingPrediction();
      const second = pendingPrediction();
      predictMock
        .mockReturnValueOnce(first.promise)
        .mockReturnValueOnce(second.promise);
      await submit("CCO");
      const signal = predictMock.mock.calls[0][1]!.signal!;
      const input = screen.getByRole("textbox", { name: "Enter SMILES" });

      await userEvent.clear(input);
      expect(signal.aborted).toBe(true);
      expect(screen.getByRole("status")).toBeEmptyDOMElement();

      await userEvent.type(input, "CCN");
      await userEvent.click(screen.getByRole("button", { name: "Predict" }));
      await act(async () => {
        if (settlement === "resolve") first.resolve(predictResponse());
        else first.reject(new PredictError("server"));
      });
      expect(
        screen.getByRole("button", { name: "Predicting…" }),
      ).toBeDisabled();
      expectNoResults();

      await act(async () =>
        second.resolve({ ...predictResponse(), smiles: "CCN" }),
      );
      expect(screen.getByText("CCN")).toBeInTheDocument();
    },
  );

  it("aborts on unmount and ignores AbortError", async () => {
    const pending = pendingPrediction();
    predictMock.mockReturnValueOnce(pending.promise);
    const { unmount } = render(<SinglePredictionPage />);
    await userEvent.type(
      screen.getByRole("textbox", { name: "Enter SMILES" }),
      "CCO",
    );
    await userEvent.click(screen.getByRole("button", { name: "Predict" }));
    const signal = predictMock.mock.calls[0][1]!.signal!;
    unmount();
    expect(signal.aborted).toBe(true);
    await act(async () => pending.resolve(predictResponse()));

    predictMock.mockRejectedValueOnce(
      new DOMException("Aborted", "AbortError"),
    );
    await submit("CCO");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Predict" })).toBeEnabled();
  });

  it("shows predictions per target, with unavailable outputs as a dash", async () => {
    const result = predictResponse();
    result.toxicity.stressResponse["SR-ARE"] = { value: 0.54 };
    result.bioactivity.BACE1.pActive = { value: 0.9 };
    const missing = { value: null, reason: "Not returned by the server" };
    result.toxicity.nuclearReceptor["NR-AR"] = missing;
    result.toxicity.nuclearReceptor["NR-ER"] = missing;
    result.fdaApproved = { value: null, reason: "Not returned by the model" };
    predictMock.mockResolvedValue(result);
    await submit("CCO");

    expect(predictMock).toHaveBeenCalledWith("CCO", expect.anything());
    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "Prediction Results",
      }),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Not returned by the model")).toHaveLength(3);
    expect(screen.getByText("1 of 10 high risk")).toBeInTheDocument();
    expect(
      screen.getByText("NR-AR, NR-ER: Not returned by the server"),
    ).toBeInTheDocument();
    expect(screen.getByText("54%")).toBeInTheDocument();
    expect(
      screen.getByText("Confidence via MC-Dropout · 50 passes"),
    ).toBeInTheDocument();
    const bioactivity = screen
      .getByRole("heading", { name: "Bioactivity" })
      .closest("section");
    expect(bioactivity).toHaveClass("panel-neutral");

    await userEvent.click(screen.getByRole("button", { name: "BACE1" }));
    expect(bioactivity).toHaveClass("panel-good");
    expect(screen.getByText("4.90")).toBeInTheDocument();
    expect(screen.getByText("95% CI 3.80 – 6.00")).toBeInTheDocument();
    expect(screen.getByText("12.6 µM")).toBeInTheDocument();
    expect(screen.getByText("95% CI 1 µM – 158 µM")).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("tab", { name: "Molecular Information" }),
    );
    expect(
      screen.getByRole("heading", { level: 1, name: "Molecular Information" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Coming Soon")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Input Molecule" }));
    expect(screen.getByLabelText("Enter SMILES")).toHaveValue("CCO");

    await userEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByLabelText("Enter SMILES")).toHaveValue("");
    expect(
      screen.getByRole("tab", { name: "Prediction Results" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("tab", { name: "Molecular Information" }),
    ).toBeDisabled();
  });

  it("shows the error and drops the previous result when a retry fails", async () => {
    predictMock
      .mockResolvedValueOnce(predictResponse())
      .mockRejectedValueOnce(new PredictError("invalid-smiles"));
    await submit("CCO");
    await screen.findByRole("heading", {
      level: 1,
      name: "Prediction Results",
    });
    await userEvent.click(screen.getByRole("tab", { name: "Input Molecule" }));
    await userEvent.click(screen.getByRole("button", { name: "Predict" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Invalid SMILES",
    );
    expect(
      screen.getByRole("tab", { name: "Prediction Results" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("tab", { name: "Molecular Information" }),
    ).toBeDisabled();

    await userEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.getByLabelText("Enter SMILES")).toHaveValue("");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
