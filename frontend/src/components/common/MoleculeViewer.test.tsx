import { render } from "@testing-library/react";
import { createViewerGrid } from "3dmol";
import { afterEach, expect, it, vi } from "vitest";
import MoleculeViewer from "./MoleculeViewer.tsx";

vi.mock("3dmol", () => ({ createViewerGrid: vi.fn() }));

afterEach(() => {
  vi.resetAllMocks();
});

it("renders a molecule from SDF text", () => {
  const viewer = {
    setBackgroundColor: vi.fn(),
    addModel: vi.fn(),
    setStyle: vi.fn(),
    resize: vi.fn(),
    zoomTo: vi.fn(),
    render: vi.fn(),
    clear: vi.fn(),
  };
  vi.mocked(createViewerGrid).mockReturnValue([
    [viewer],
  ] as unknown as ReturnType<typeof createViewerGrid>);

  render(<MoleculeViewer sdf="molecule SDF" />);

  expect(viewer.render).toHaveBeenCalledOnce();
  expect(viewer.addModel).toHaveBeenCalledWith("molecule SDF", "sdf");
});
