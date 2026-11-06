import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { molInfoResponse } from "../../../test/fixtures.ts";
import MolecularInformation from "./MolecularInformation.tsx";

vi.mock("../../../components/common/MoleculeViewer.tsx", () => ({
  default: () => <div data-testid="viewer" />,
}));

function renderInfo(info = molInfoResponse()) {
  render(<MolecularInformation info={info} loading={false} error={null} />);
}

it("shows identifiers, descriptors and the Lipinski checklist", async () => {
  renderInfo({ ...molInfoResponse(), logP: 6.2, lipinskiViolations: 1 });

  expect(screen.getByText("C9H8O4")).toBeInTheDocument();
  expect(screen.getByText("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")).toBeInTheDocument();
  expect(screen.getByText("180.16")).toBeInTheDocument();
  expect(screen.getByText(/1 of 4 rules violated/)).toBeInTheDocument();
  expect(screen.getByText("LogP ≤ 5").parentElement).toHaveTextContent("Fail");
  expect(screen.getByText("H-bond donors ≤ 5").parentElement).toHaveTextContent(
    "Pass",
  );
  expect(screen.getByRole("img")).toHaveAttribute(
    "src",
    expect.stringMatching(/^data:image\/svg\+xml/),
  );

  await userEvent.click(screen.getByRole("button", { name: "3D" }));
  expect(screen.getByTestId("viewer")).toBeInTheDocument();
  expect(
    screen.getByText("Computed with RDKit and OpenChemLib."),
  ).toBeVisible();
});

it("hides the 3D option without a molblock and InChIKey without a key", () => {
  renderInfo({ ...molInfoResponse(), molblock: null, inchiKey: null });

  expect(screen.queryByRole("button", { name: "3D" })).not.toBeInTheDocument();
  expect(screen.queryByText("InChIKey")).not.toBeInTheDocument();
});

it("explains descriptors, identifiers and drug-likeness through info tooltips", () => {
  renderInfo();

  for (const [name, text] of [
    ["LogP", /lipophilicity/],
    ["Formula", /atoms of each element/],
    ["Drug-likeness", /rule of five/],
  ] as const) {
    expect(
      screen.getByRole("button", { name: `About ${name}` }),
    ).toHaveAccessibleDescription(text);
  }
});

it("opens an info tooltip on tap and closes it on a tap elsewhere", async () => {
  renderInfo();
  const tip = screen.getByRole("button", { name: "About LogP" });

  await userEvent.click(tip);
  expect(tip).toHaveAttribute("data-open");

  await userEvent.click(document.body);
  expect(tip).not.toHaveAttribute("data-open");
});
