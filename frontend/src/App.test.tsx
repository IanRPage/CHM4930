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

  it("navigates with the sidebar, including Coming Soon routes", async () => {
    renderAt("/");
    await userEvent.click(
      screen.getByRole("link", { name: "Batch Prediction" }),
    );
    expect(
      screen.getByRole("heading", { level: 1, name: "Batch Prediction" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Coming Soon")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Batch Prediction" })).toHaveClass(
      "active",
    );

    await userEvent.click(
      screen.getByRole("link", { name: "Dataset Explorer" }),
    );
    expect(
      screen.getByRole("heading", { level: 1, name: "Dataset Explorer" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Coming Soon")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Dataset Explorer" })).toHaveClass(
      "active",
    );

    await userEvent.click(screen.getByRole("link", { name: "About" }));
    expect(
      screen.getByRole("heading", { level: 1, name: "About MuFu" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "About" })).toHaveClass("active");
  });

  it("routes the home entry cards and marks unfinished ones as coming soon", async () => {
    renderAt("/");
    expect(screen.getAllByText("Coming soon")).toHaveLength(2);
    expect(within(screen.getByRole("main")).getAllByRole("link")).toHaveLength(
      1,
    );
    await userEvent.click(screen.getByRole("link", { name: "Get Started" }));
    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Single Molecule Prediction",
      }),
    ).toBeInTheDocument();
  });
});
