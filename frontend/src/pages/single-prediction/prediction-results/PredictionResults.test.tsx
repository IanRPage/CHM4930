import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { predictResponse } from "../../../test/fixtures.ts";
import { downloadPredictionCsv } from "./exportCsv.ts";
import PredictionResults from "./PredictionResults.tsx";

vi.mock("./exportCsv.ts", () => ({ downloadPredictionCsv: vi.fn() }));

describe("PredictionResults", () => {
  it("offers a CSV download for the current prediction", async () => {
    const result = predictResponse();
    render(<PredictionResults result={result} />);

    await userEvent.click(screen.getByRole("button", { name: "Download CSV" }));

    expect(downloadPredictionCsv).toHaveBeenCalledWith(result);
    expect(
      screen.getByRole("heading", { name: "Bioactivity" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Toxicity risk" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Likelihood of FDA approval" }),
    ).toBeInTheDocument();
    // SCRUM-98 has not supplied descriptors through the API yet.
    expect(screen.queryByText("Molecular Properties")).not.toBeInTheDocument();
  });

  it("renders a complete prediction response", () => {
    const result = predictResponse();
    result.bioactivity.EGFR.pic50 = { value: 6.2, lo: 5.9, hi: 6.5 };
    result.bioactivity.EGFR.ic50Nm = { value: 630, lo: 316, hi: 1259 };
    render(<PredictionResults result={result} />);

    expect(screen.getByText("6.20")).toBeInTheDocument();
    expect(screen.getByText("630 nM")).toBeInTheDocument();
    expect(screen.getAllByText("10%")).toHaveLength(12);
    expect(screen.getByText("91%")).toBeInTheDocument();
    expect(
      screen.queryByText("Not returned by the model"),
    ).not.toBeInTheDocument();
  });
});
