import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fetchModelVersion } from "../../services/health.ts";
import Sidebar from "./Sidebar.tsx";

vi.mock("../../services/health.ts", async (importOriginal) => ({
  ...(await importOriginal()),
  fetchModelVersion: vi.fn(),
}));

const fetchModelVersionMock = vi.mocked(fetchModelVersion);

function renderSidebar() {
  return render(
    <MemoryRouter>
      <Sidebar />
    </MemoryRouter>,
  );
}

afterEach(() => {
  fetchModelVersionMock.mockReset();
});

describe("Sidebar", () => {
  it("displays the active model version from the backend and external links", async () => {
    fetchModelVersionMock.mockResolvedValue("mufu-v2.1.0");
    renderSidebar();

    expect(await screen.findByText("Model: mufu-v2.1.0")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "GitHub" })).toHaveAttribute(
      "href",
      "https://github.com/IanRPage/CHM4930",
    );
    expect(screen.getByRole("link", { name: "Hugging Face" })).toHaveAttribute(
      "href",
      "https://huggingface.co/CHM4930/mufu",
    );
  });

  it("falls back to mufu-v1.0.0 and toggles between expanded and collapsed states", async () => {
    fetchModelVersionMock.mockResolvedValue(null);
    renderSidebar();

    expect(screen.getByText("Model: mufu-v1.0.0")).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("button", { name: "Collapse sidebar" }),
    );
    expect(screen.getByRole("complementary")).toHaveClass("sidebar-collapsed");
    expect(screen.queryByText("MuFu")).not.toBeInTheDocument();
    expect(screen.queryByText(/Model:/)).not.toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Batch Prediction" }),
    ).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("button", { name: "Expand sidebar" }),
    );
    expect(screen.getByRole("complementary")).not.toHaveClass(
      "sidebar-collapsed",
    );
    expect(screen.getByText("MuFu")).toBeInTheDocument();
    expect(screen.getByText("Model: mufu-v1.0.0")).toBeInTheDocument();
  });
});
