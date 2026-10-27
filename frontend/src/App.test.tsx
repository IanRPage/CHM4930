import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import App from "./App.tsx";

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
    ["/predict", "Single Prediction"],
    ["/batch", "Batch Prediction"],
    ["/explorer", "Dataset Explorer"],
    ["/about", "About"],
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

  it("navigates with the sidebar", async () => {
    renderAt("/");
    await userEvent.click(screen.getByRole("link", { name: "About" }));
    expect(
      screen.getByRole("heading", { level: 1, name: "About" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "About" })).toHaveClass("active");
  });
});
