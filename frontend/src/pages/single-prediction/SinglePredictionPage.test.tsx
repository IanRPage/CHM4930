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

  it.each(["", "   ", "\u00a0\t "])(
    "validates blank input %j accessibly without a request",
    async (smiles) => {
      render(<SinglePredictionPage />);
      const input = screen.getByRole("textbox", { name: "Enter SMILES" });
      fireEvent.change(input, { target: { value: smiles } });
      await userEvent.click(screen.getByRole("button", { name: "Predict" }));

      expect(predictMock).not.toHaveBeenCalled();
      const message = "Enter a SMILES string to predict a molecule.";
      expect(screen.getByText(message)).toBeVisible();
      expect(input).toBeInvalid();
      expect(input).toHaveFocus();
      expect(input).toHaveAccessibleDescription(message);
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(screen.getByRole("status")).toBeEmptyDOMElement();
      expect(screen.getByRole("button", { name: "Predict" })).toBeEnabled();
      expectNoResults();

      // Repeated button and keyboard submissions keep one associated message.
      await userEvent.click(screen.getByRole("button", { name: "Predict" }));
      expect(input).toHaveFocus();
      await userEvent.keyboard("{Enter}");
      expect(input).toHaveFocus();
      expect(input).toBeInvalid();
      expect(input).toHaveAccessibleDescription(message);
      expect(screen.getAllByText(message)).toHaveLength(1);
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(screen.getByRole("status")).toBeEmptyDOMElement();
      expect(predictMock).not.toHaveBeenCalled();

      await userEvent.clear(input);
      await userEvent.type(input, " CCO ");
      expect(input).toBeValid();
      expect(input).not.toHaveAttribute("aria-describedby");
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      predictMock.mockResolvedValue(predictResponse());
      await userEvent.click(screen.getByRole("button", { name: "Predict" }));
      expect(predictMock).toHaveBeenCalledWith("CCO", expect.anything());
      expect(
        await screen.findByRole("heading", {
          level: 1,
          name: "Prediction Results",
        }),
      ).toBeInTheDocument();
    },
  );

  it("focuses the results tab after keyboard submission without overriding manual navigation", async () => {
    const user = userEvent.setup();
    const pending = pendingPrediction();
    predictMock.mockReturnValue(pending.promise);
    render(<SinglePredictionPage />);
    const inputTab = screen.getByRole("tab", { name: "Input Molecule" });
    const resultsTab = screen.getByRole("tab", { name: "Prediction Results" });
    const infoTab = screen.getByRole("tab", { name: "Molecular Information" });
    const input = screen.getByRole("textbox", { name: "Enter SMILES" });

    await user.tab();
    expect(inputTab).toHaveFocus();
    await user.tab();
    expect(input).toHaveFocus();
    await user.keyboard("CCO{Enter}");
    expect(predictMock).toHaveBeenCalledWith("CCO", expect.anything());
    expect(input).toHaveFocus();

    await act(async () => pending.resolve(predictResponse()));
    expect(resultsTab).toHaveFocus();
    expect(resultsTab).toHaveAttribute("aria-selected", "true");
    expect(
      screen.getByRole("heading", { level: 1, name: "Prediction Results" }),
    ).toBeInTheDocument();

    await user.tab();
    expect(infoTab).toHaveFocus();
    await user.keyboard("{Enter}");
    expect(infoTab).toHaveFocus();
    expect(infoTab).toHaveAttribute("aria-selected", "true");
    await user.tab({ shift: true });
    await user.tab({ shift: true });
    expect(inputTab).toHaveFocus();
    await user.keyboard("{Enter}");
    expect(inputTab).toHaveFocus();
    expect(screen.getByRole("button", { name: "Predict" })).toBeEnabled();
    expect(screen.getByRole("status")).toBeEmptyDOMElement();
    await user.tab();
    await user.keyboard("{Enter}");
    expect(resultsTab).toHaveFocus();
    expect(resultsTab).toHaveAttribute("aria-selected", "true");
    expect(predictMock).toHaveBeenCalledTimes(1);
  });

  it.each([
    [
      new PredictError("invalid-smiles"),
      "Invalid SMILES. Check your SMILES string and try again.",
    ],
    [
      new PredictError("busy"),
      "The model is busy. Please wait a moment and try again.",
    ],
    [
      new PredictError("unavailable"),
      "The model is currently unavailable. Please try again later.",
    ],
    [
      new PredictError("timeout"),
      "The prediction took too long. Please try again.",
    ],
    [
      new PredictError("network"),
      "Can't reach the server. Check your connection and try again.",
    ],
    [
      new PredictError("server"),
      "We couldn't complete the prediction. Please try again later.",
    ],
    [
      new PredictError("invalid-response"),
      "We couldn't read the prediction. Please try again.",
    ],
    [
      new Error("Internal diagnostic details"),
      "Something went wrong. Please try again.",
    ],
  ])(
    "shows recovery guidance for %s and allows correction and retry",
    async (error, message) => {
      predictMock
        .mockRejectedValueOnce(error)
        .mockResolvedValueOnce({ ...predictResponse(), smiles: "CCN" });
      await submit("CCO");

      expect(await screen.findByRole("alert")).toHaveTextContent(message);
      const input = screen.getByRole("textbox", { name: "Enter SMILES" });
      expect(input).toHaveValue("CCO");
      expect(input).toHaveAccessibleDescription(message);
      expect(input).toHaveAttribute(
        "aria-invalid",
        String(
          error instanceof PredictError && error.kind === "invalid-smiles",
        ),
      );
      expect(screen.getByRole("button", { name: "Predict" })).toBeEnabled();
      expect(screen.getByRole("button", { name: "Clear" })).toBeEnabled();
      expect(screen.getByRole("status")).toBeEmptyDOMElement();
      expectNoResults();

      await userEvent.clear(input);
      await userEvent.type(input, "CCN");
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expect(input).toBeValid();
      await userEvent.click(screen.getByRole("button", { name: "Predict" }));
      expect(await screen.findByText("CCN")).toBeInTheDocument();
      expect(predictMock).toHaveBeenLastCalledWith("CCN", expect.anything());
    },
  );

  it("allows retrying the same input after a temporary failure", async () => {
    predictMock
      .mockRejectedValueOnce(new PredictError("busy"))
      .mockResolvedValueOnce(predictResponse());
    await submit("CCO");
    await screen.findByRole("alert");
    await userEvent.click(screen.getByRole("button", { name: "Predict" }));
    expect(
      await screen.findByRole("heading", {
        level: 1,
        name: "Prediction Results",
      }),
    ).toBeInTheDocument();
    expect(predictMock).toHaveBeenCalledTimes(2);
    expect(predictMock).toHaveBeenLastCalledWith("CCO", expect.anything());
  });

  it.each(["resolve", "reject"] as const)(
    "ignores an aborted request that later %ss while the next request is pending",
    async (settlement) => {
      const previous = pendingPrediction();
      const next = pendingPrediction();
      predictMock
        .mockReturnValueOnce(previous.promise)
        .mockReturnValueOnce(next.promise);
      await submit("CCO");
      const signal = predictMock.mock.calls[0][1]!.signal!;
      const input = screen.getByRole("textbox", { name: "Enter SMILES" });
      await userEvent.clear(input);
      expect(signal.aborted).toBe(true);
      expect(screen.getByRole("status")).toBeEmptyDOMElement();
      await userEvent.type(input, "CCN");
      await userEvent.click(screen.getByRole("button", { name: "Predict" }));

      await act(async () => {
        if (settlement === "resolve") previous.resolve(predictResponse());
        else previous.reject(new PredictError("server"));
      });
      expect(
        screen.getByRole("button", { name: "Predicting…" }),
      ).toBeDisabled();
      expect(screen.getByRole("status")).toHaveTextContent(
        "Predicting molecule properties",
      );
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expectNoResults();

      await act(async () =>
        next.resolve({ ...predictResponse(), smiles: "CCN" }),
      );
      expect(screen.getByText("CCN")).toBeInTheDocument();
      expect(
        screen.queryByText(predictResponse().smiles!),
      ).not.toBeInTheDocument();
    },
  );

  it.each(["resolve", "reject"] as const)(
    "ignores a canceled request that later %ss without a replacement request",
    async (settlement) => {
      const pending = pendingPrediction();
      predictMock.mockReturnValue(pending.promise);
      await submit("CCO");
      const signal = predictMock.mock.calls[0][1]!.signal!;
      const input = screen.getByRole("textbox", { name: "Enter SMILES" });
      await userEvent.type(input, "N");
      expect(signal.aborted).toBe(true);
      expect(input).toHaveValue("CCON");
      expect(screen.getByRole("status")).toBeEmptyDOMElement();
      expect(screen.getByRole("button", { name: "Predict" })).toBeEnabled();

      await act(async () => {
        if (settlement === "resolve") pending.resolve(predictResponse());
        else pending.reject(new PredictError("server"));
      });

      expect(predictMock).toHaveBeenCalledTimes(1);
      expect(input).toHaveValue("CCON");
      expect(input).toHaveFocus();
      expect(screen.getByRole("status")).toBeEmptyDOMElement();
      expect(screen.getByRole("button", { name: "Predict" })).toBeEnabled();
      expect(screen.getByRole("button", { name: "Clear" })).toBeEnabled();
      expect(screen.queryByRole("alert")).not.toBeInTheDocument();
      expectNoResults();
    },
  );

  it.each([
    ["success", "resolve"],
    ["success", "reject"],
    ["failure", "resolve"],
    ["failure", "reject"],
  ] as const)(
    "keeps the newer %s when an aborted request later %ss",
    async (outcome, settlement) => {
      const previous = pendingPrediction();
      const next = pendingPrediction();
      predictMock
        .mockReturnValueOnce(previous.promise)
        .mockReturnValueOnce(next.promise);
      await submit("CCO");
      const signal = predictMock.mock.calls[0][1]!.signal!;
      const input = screen.getByRole("textbox", { name: "Enter SMILES" });
      await userEvent.clear(input);
      expect(signal.aborted).toBe(true);
      await userEvent.type(input, "CCN");
      await userEvent.click(screen.getByRole("button", { name: "Predict" }));
      await act(async () => {
        if (outcome === "success")
          next.resolve({ ...predictResponse(), smiles: "CCN" });
        else next.reject(new PredictError("unavailable"));
      });
      await act(async () => {
        if (settlement === "resolve") previous.resolve(predictResponse());
        else previous.reject(new PredictError("server"));
      });

      if (outcome === "success") {
        expect(screen.getByText("CCN")).toBeInTheDocument();
        expect(
          screen.getByRole("tab", { name: "Prediction Results" }),
        ).toHaveAttribute("aria-selected", "true");
        await userEvent.click(
          screen.getByRole("tab", { name: "Input Molecule" }),
        );
        expect(screen.getByRole("button", { name: "Predict" })).toBeEnabled();
        expect(screen.getByRole("button", { name: "Clear" })).toBeEnabled();
        expect(screen.getByRole("status")).toBeEmptyDOMElement();
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
        await userEvent.click(
          screen.getByRole("tab", { name: "Prediction Results" }),
        );
        expect(screen.getByText("CCN")).toBeInTheDocument();
      } else {
        expect(screen.getByRole("alert")).toHaveTextContent(
          "The model is currently unavailable",
        );
        expect(screen.getByRole("button", { name: "Predict" })).toBeEnabled();
        expect(screen.getByRole("status")).toBeEmptyDOMElement();
        expectNoResults();
      }
      expect(
        screen.queryByText(predictResponse().smiles!),
      ).not.toBeInTheDocument();
      expect(predictMock).toHaveBeenCalledTimes(2);
    },
  );

  it.each(["resolve", "reject"] as const)(
    "aborts on unmount even if the request later %ss",
    async (settlement) => {
      const pending = pendingPrediction();
      predictMock.mockReturnValue(pending.promise);
      const { unmount } = render(<SinglePredictionPage />);
      await userEvent.type(
        screen.getByRole("textbox", { name: "Enter SMILES" }),
        "CCO",
      );
      await userEvent.click(screen.getByRole("button", { name: "Predict" }));
      const signal = predictMock.mock.calls[0][1]!.signal!;
      unmount();
      expect(signal.aborted).toBe(true);
      await act(async () => {
        if (settlement === "resolve") pending.resolve(predictResponse());
        else pending.reject(new DOMException("Aborted", "AbortError"));
      });
    },
  );

  it("ignores AbortError, drops previous results, and releases the loading state", async () => {
    predictMock
      .mockResolvedValueOnce(predictResponse())
      .mockRejectedValueOnce(new DOMException("Aborted", "AbortError"));
    await submit("CCO");
    await screen.findByRole("heading", {
      level: 1,
      name: "Prediction Results",
    });
    await userEvent.click(screen.getByRole("tab", { name: "Input Molecule" }));
    await userEvent.click(screen.getByRole("button", { name: "Predict" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toBeEmptyDOMElement();
    expect(screen.getByRole("button", { name: "Predict" })).toBeEnabled();
    expectNoResults();
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
