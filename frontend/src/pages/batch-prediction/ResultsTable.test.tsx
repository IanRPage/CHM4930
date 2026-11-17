import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { predictResponse } from "../../test/fixtures.ts";
import type { BatchResult } from "../../types/batch.ts";
import ResultsTable from "./ResultsTable.tsx";

function result(row: number, smiles: string, highTox = false): BatchResult {
  const prediction = predictResponse();
  if (highTox) prediction.toxicity.nuclearReceptor["NR-AR"] = { value: 0.8 };
  return { row, smiles, prediction, error: null };
}

const results = [
  ...Array.from({ length: 9 }, (_, i) => result(i + 1, `CC${"C".repeat(i)}`)),
  result(10, "NNN", true),
  {
    row: 11,
    smiles: "bad",
    prediction: null,
    error: "Invalid",
  },
];

function renderTable(props: { search?: string } = {}) {
  return render(
    <ResultsTable
      results={results}
      target="BACE1"
      search={props.search ?? ""}
    />,
  );
}

describe("ResultsTable", () => {
  it("pages rows with the selected target's values and shows invalid rows", async () => {
    renderTable();
    expect(screen.getByText("1–10 of 11")).toBeInTheDocument();
    expect(screen.getAllByText("4.90")).toHaveLength(10);
    expect(screen.getAllByText("±1.10")).toHaveLength(10);
    await userEvent.click(screen.getByRole("button", { name: "›" }));
    expect(screen.getByText("11–11 of 11")).toBeInTheDocument();
    expect(screen.getByText("Invalid: Invalid")).toBeInTheDocument();
  });

  it("keeps small ic50 values readable", () => {
    const small = result(1, "CCO");
    small.prediction!.bioactivity.BACE1.ic50Nm = { value: 0.4, lo: 0.1, hi: 1 };
    render(<ResultsTable results={[small]} target="BACE1" search="" />);
    expect(screen.getByText("0.4")).toBeInTheDocument();
  });

  it("filters by SMILES search", () => {
    renderTable({ search: "nn" });
    expect(screen.getByText("1–1 of 1")).toBeInTheDocument();
  });

  it("explains a tox tile on keyboard focus", async () => {
    renderTable();
    const tile = screen.getByLabelText("NR-AR: 80%");
    tile.focus();
    expect(
      await screen.findByText("NR-AR: 80%. Androgen receptor activity"),
    ).toBeInTheDocument();
  });
});
