import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getDatasetMolecules,
  getDatasetSummary,
} from "../../services/dataset.ts";
import { getMolInfo } from "../../services/molInfo.ts";
import {
  datasetMolecule,
  datasetMoleculesResponse,
  datasetSummaryResponse,
} from "../../test/datasetFixtures.ts";
import { molInfoResponse } from "../../test/fixtures.ts";
import DatasetExplorerPage from "./DatasetExplorerPage.tsx";

vi.mock("../../services/dataset.ts", () => ({
  getDatasetSummary: vi.fn(),
  getDatasetMolecules: vi.fn(),
}));
vi.mock("../../services/molInfo.ts", () => ({ getMolInfo: vi.fn() }));
const summaryMock = vi.mocked(getDatasetSummary);
const moleculesMock = vi.mocked(getDatasetMolecules);

beforeEach(() => {
  vi.resetAllMocks();
  summaryMock.mockResolvedValue(datasetSummaryResponse());
  moleculesMock.mockImplementation(async (query) =>
    datasetMoleculesResponse(query.page),
  );
  vi.mocked(getMolInfo).mockResolvedValue(molInfoResponse());
});

describe("DatasetExplorerPage", () => {
  it("renders actual API counts and charts, and supports keyboard tab navigation", async () => {
    render(<DatasetExplorerPage />);
    expect(
      await screen.findByText("With Bioactivity Labels"),
    ).toBeInTheDocument();
    const overview = screen.getByRole("tabpanel", { name: "Overview" });
    expect(
      within(overview).getByText("4", { selector: "strong" }),
    ).toBeInTheDocument();
    expect(
      within(overview).getByText("3", { selector: "strong" }),
    ).toBeInTheDocument();
    expect(
      within(overview).getByText("2", { selector: "strong" }),
    ).toBeInTheDocument();
    expect(
      within(overview).getByRole("img", { name: /pIC50 distribution/ }),
    ).toBeInTheDocument();
    const overviewTab = screen.getByRole("tab", { name: "Overview" });
    overviewTab.focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(
      screen.getByRole("tab", { name: "Property Distribution" }),
    ).toHaveFocus();
    expect(
      screen.getByRole("tabpanel", { name: "Property Distribution" }),
    ).toBeInTheDocument();
    await userEvent.keyboard("{Home}");
    expect(overviewTab).toHaveFocus();
  });

  it("changes bins and source together, preserving the fixed activity threshold and handling missing labels", async () => {
    render(<DatasetExplorerPage />);
    await screen.findByText("With Bioactivity Labels");
    await userEvent.click(
      screen.getByRole("tab", { name: "Property Distribution" }),
    );
    expect(
      screen.getByRole("img", { name: /Active threshold: 6/ }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Class imbalance: 1 of 2/)).toBeInTheDocument();
    const panel = screen.getByRole("tabpanel", {
      name: "Property Distribution",
    });
    await userEvent.selectOptions(
      screen.getByRole("combobox", { name: "Bin width" }),
      "1",
    );
    expect(
      panel.querySelector('[aria-label="BACE1: 5 to less than 6: 1"]'),
    ).toBeInTheDocument();
    expect(
      panel.querySelector('[aria-label="EGFR: 5 to less than 6: 1"]'),
    ).toBeInTheDocument();
    await userEvent.selectOptions(
      screen.getByRole("combobox", { name: "Source" }),
      "tox21",
    );
    expect(
      screen.getByText("No observed bioactivity labels in this source."),
    ).toBeInTheDocument();
    expect(screen.getAllByText("No observed labels")).toHaveLength(4);
    expect(panel.textContent).not.toMatch(/NaN|Infinity/);
  });

  it("shows all element percentages, hydrogen, sorting and real summary statistics", async () => {
    render(<DatasetExplorerPage />);
    await screen.findByText("With Bioactivity Labels");
    await userEvent.click(
      screen.getByRole("tab", { name: "Element Composition" }),
    );

    expect(screen.getByText("Hydrogen")).toBeInTheDocument();
    expect(
      screen.getByText("Total distinct elements").nextElementSibling,
    ).toHaveTextContent("4");
    await userEvent.selectOptions(
      screen.getByRole("combobox", { name: "Element sort order" }),
      "element",
    );
    const bars = screen
      .getByRole("img", {
        name: "Percentage of molecules containing each element",
      })
      .querySelectorAll(".explorer-bar");
    expect(
      Array.from(bars).map((bar) => bar.nextElementSibling?.textContent),
    ).toEqual(["C", "F", "H", "O"]);
  });

  it("retries summary loading after a failure", async () => {
    summaryMock.mockRejectedValueOnce(new Error("offline"));
    render(<DatasetExplorerPage />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Couldn't load dataset statistics",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Retry statistics" }),
    );
    expect(
      await screen.findByText("With Bioactivity Labels"),
    ).toBeInTheDocument();
    expect(summaryMock).toHaveBeenCalledTimes(2);
  });

  it("appends pages without losing existing cards and removes Load more on the final page", async () => {
    render(<DatasetExplorerPage />);
    await userEvent.click(
      screen.getByRole("tab", { name: "Sample Molecules" }),
    );
    expect(
      await screen.findByRole("article", { name: "mol_first" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Negative for toxicity", { selector: "span" }),
    ).toHaveClass("pill-good");
    expect(
      await screen.findByAltText("Molecular structure of CCO"),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Load more" }));
    expect(
      await screen.findByRole("article", { name: "mol_second" }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("article")).toHaveLength(2);
    expect(screen.getByRole("status")).toHaveTextContent("Showing 2 of 2");
    expect(
      screen.queryByRole("button", { name: "Load more" }),
    ).not.toBeInTheDocument();
  });

  it("resets pagination for search and filters, preserving SMILES case and special characters", async () => {
    render(<DatasetExplorerPage />);
    await userEvent.click(
      screen.getByRole("tab", { name: "Sample Molecules" }),
    );
    await screen.findByRole("article", { name: "mol_first" });
    await userEvent.click(screen.getByRole("button", { name: "Load more" }));
    await screen.findByRole("article", { name: "mol_second" });
    fireEvent.change(screen.getByRole("searchbox"), {
      target: { value: "[NH4+]" },
    });
    await userEvent.selectOptions(
      screen.getByRole("combobox", { name: "Activity" }),
      "unknown",
    );
    await userEvent.selectOptions(
      screen.getByRole("combobox", { name: "Toxicity labels" }),
      "positive",
    );
    await waitFor(() =>
      expect(moleculesMock).toHaveBeenLastCalledWith(
        { q: "[NH4+]", activity: "unknown", risk: "positive", page: 1 },
        expect.any(AbortSignal),
      ),
    );
    await screen.findByRole("article", { name: "mol_first" });
    expect(
      screen.queryByRole("article", { name: "mol_second" }),
    ).not.toBeInTheDocument();
  });

  it("aborts old searches and ignores late results, while rendering an explicit empty state", async () => {
    let resolveOld!: (
      response: ReturnType<typeof datasetMoleculesResponse>,
    ) => void;
    moleculesMock.mockReturnValueOnce(
      new Promise((resolve) => {
        resolveOld = resolve;
      }),
    );
    render(<DatasetExplorerPage />);
    await userEvent.click(
      screen.getByRole("tab", { name: "Sample Molecules" }),
    );
    await waitFor(() => expect(moleculesMock).toHaveBeenCalledTimes(1));
    const oldSignal = moleculesMock.mock.calls[0][1]!;
    moleculesMock.mockResolvedValueOnce({
      items: [],
      page: 1,
      pageSize: 24,
      total: 0,
      totalPages: 0,
      hasMore: false,
    });
    fireEvent.change(screen.getByRole("searchbox"), {
      target: { value: "no match" },
    });
    expect(
      await screen.findByText("No molecules match your search and filters."),
    ).toBeInTheDocument();
    expect(oldSignal.aborted).toBe(true);
    await act(async () => resolveOld(datasetMoleculesResponse()));
    expect(screen.queryByRole("article")).not.toBeInTheDocument();
  });

  it("retains loaded cards on a page failure and retries that same page", async () => {
    moleculesMock
      .mockResolvedValueOnce(datasetMoleculesResponse())
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(datasetMoleculesResponse(2));
    vi.mocked(getMolInfo).mockRejectedValue(new Error("structure unavailable"));
    render(<DatasetExplorerPage />);
    await userEvent.click(
      screen.getByRole("tab", { name: "Sample Molecules" }),
    );
    await screen.findByRole("article", { name: "mol_first" });
    expect(
      await screen.findByText("Structure unavailable"),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Load more" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Couldn't load molecules",
    );
    expect(
      screen.getByRole("article", { name: "mol_first" }),
    ).toBeInTheDocument();
    await userEvent.click(
      screen.getByRole("button", { name: "Retry molecules" }),
    );
    expect(
      await screen.findByRole("article", { name: "mol_second" }),
    ).toBeInTheDocument();
    expect(moleculesMock.mock.calls.map(([query]) => query.page)).toEqual([
      1, 2, 2,
    ]);
  });

  it("keeps missing target and toxicity labels distinct from measured zero values", async () => {
    const molecule = datasetMolecule();
    molecule.risk = "unknown";
    moleculesMock.mockResolvedValue({
      ...datasetMoleculesResponse(),
      items: [molecule],
    });
    render(<DatasetExplorerPage />);
    await userEvent.click(
      screen.getByRole("tab", { name: "Sample Molecules" }),
    );
    const card = await screen.findByRole("article", { name: "mol_first" });
    expect(
      within(card).getByText("BACE1 pIC50").nextElementSibling,
    ).toHaveTextContent("6.00");
    expect(
      within(card).getByText("EGFR pIC50").nextElementSibling,
    ).toHaveTextContent("—");
    expect(within(card).getByText("Unlabeled")).toHaveClass("pill-neutral");
  });
});
