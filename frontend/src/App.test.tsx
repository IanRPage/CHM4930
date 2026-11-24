import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import App from "./App.tsx";

vi.mock("./components/common/MoleculeViewer.tsx", () => ({
  default: () => null,
}));

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("App", () => {
  it.each([
    ["/", "MuFu"],
    ["/predict", "Single Molecule Prediction"],
    ["/batch", "Batch Prediction"],
    ["/explorer", "Dataset Explorer"],
    ["/about", "About MuFu"],
  ])("renders the page for %s", (path, heading) => {
    renderAt(path);
    expect(
      screen.getByRole("heading", { level: 1, name: heading }),
    ).toBeInTheDocument();
  });

  it("renders a not-found page with a link home for unknown paths", async () => {
    renderAt("/does-not-exist");
    expect(
      screen.getByRole("heading", { level: 1, name: "Page not found" }),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("link", { name: "Back to home" }));
    expect(
      screen.getByRole("heading", { level: 1, name: "MuFu" }),
    ).toBeInTheDocument();
  });

  it("navigates with the sidebar, including the dataset explorer", async () => {
    renderAt("/");
    await userEvent.click(
      screen.getByRole("link", { name: "Batch Prediction" }),
    );
    expect(
      screen.getByRole("heading", { level: 1, name: "Batch Prediction" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Drag and drop a file here")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Batch Prediction" })).toHaveClass(
      "active",
    );

    await userEvent.click(
      screen.getByRole("link", { name: "Dataset Explorer" }),
    );
    expect(
      screen.getByRole("heading", { level: 1, name: "Dataset Explorer" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByRole("link", { name: "Dataset Explorer" })).toHaveClass(
      "active",
    );

    await userEvent.click(screen.getByRole("link", { name: "About" }));
    expect(
      screen.getByRole("heading", { level: 1, name: "About MuFu" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "About" })).toHaveClass("active");
  });

  it.each([
    ["Get Started", "Single Molecule Prediction"],
    ["Explore", "Dataset Explorer"],
  ])("routes the home entry card %s", async (action, heading) => {
    renderAt("/");
    expect(screen.queryByText("Coming soon")).not.toBeInTheDocument();
    expect(within(screen.getByRole("main")).getAllByRole("link")).toHaveLength(
      3,
    );
    await userEvent.click(screen.getByRole("link", { name: action }));
    expect(
      screen.getByRole("heading", {
        level: 1,
        name: heading,
      }),
    ).toBeInTheDocument();
  });
});
