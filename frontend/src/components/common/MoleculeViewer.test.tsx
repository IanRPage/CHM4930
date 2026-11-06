import { fireEvent, render } from "@testing-library/react";
import { createViewerGrid } from "3dmol";
import { afterEach, expect, it, vi } from "vitest";
import MoleculeViewer from "./MoleculeViewer.tsx";

vi.mock("3dmol", () => ({ createViewerGrid: vi.fn() }));

afterEach(() => {
  vi.resetAllMocks();
});

function mockViewer() {
  const viewer = {
    setBackgroundColor: vi.fn(),
    addModel: vi.fn(),
    setStyle: vi.fn(),
    resize: vi.fn(),
    zoomTo: vi.fn(),
    getPerceivedDistance: vi.fn().mockReturnValue(100),
    setZoomLimits: vi.fn(),
    render: vi.fn(),
    clear: vi.fn(),
  };
  vi.mocked(createViewerGrid).mockReturnValue([
    [viewer],
  ] as unknown as ReturnType<typeof createViewerGrid>);
  return viewer;
}

it("renders a molecule from SDF text, limiting zoom relative to its fitted size", () => {
  const viewer = mockViewer();

  render(<MoleculeViewer sdf="molecule SDF" />);

  expect(viewer.render).toHaveBeenCalledOnce();
  expect(viewer.addModel).toHaveBeenCalledWith("molecule SDF", "sdf");
  expect(viewer.setZoomLimits).toHaveBeenCalledWith(25, 200);
});

it("treats right-button drag as a pan only when pan is set", () => {
  mockViewer();
  const which = (pan: boolean) => {
    const { container } = render(<MoleculeViewer sdf="x" pan={pan} />);
    const canvas = document.createElement("canvas");
    container.firstElementChild!.append(canvas);
    let seen = 0;
    canvas.addEventListener("mousedown", (ev) => (seen = ev.which));
    fireEvent.mouseDown(canvas, { button: 2, which: 3 });
    return seen;
  };

  expect(which(true)).toBe(2);
  expect(which(false)).toBe(3);
});
