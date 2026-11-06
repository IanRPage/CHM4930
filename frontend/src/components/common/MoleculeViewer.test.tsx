import { fireEvent, render } from "@testing-library/react";
import { createViewerGrid } from "3dmol";
import { afterEach, expect, it, vi } from "vitest";
import MoleculeViewer from "./MoleculeViewer.tsx";

vi.mock("3dmol", () => ({ createViewerGrid: vi.fn() }));

afterEach(() => {
  vi.resetAllMocks();
});

function mockViewer() {
  let view = [0, 0, 0, 0, 0, 0, 0, 1];
  const viewer = {
    setBackgroundColor: vi.fn(),
    addModel: vi.fn(),
    setStyle: vi.fn(),
    resize: vi.fn(),
    zoomTo: vi.fn(),
    getModel: vi.fn(() => ({
      selectedAtoms: () => [
        { x: 0, y: 0, z: 0 },
        { x: 3, y: 1, z: 0 },
        { x: 1, y: 2, z: 0.1 },
      ],
    })),
    getView: vi.fn(() => [...view]),
    setView: vi.fn((v: number[]) => (view = v)),
    // offsets are 1:1 and the molecule center's screen offset is the view's x
    screenOffsetToModel: vi.fn((x: number, y: number) => ({ x, y, z: 0 })),
    modelToScreen: vi.fn(() => [
      { x: view[0], y: view[1] },
      { x: 0, y: 0 },
    ]),
    getPerceivedDistance: vi.fn().mockReturnValue(100),
    setZoomLimits: vi.fn(),
    zoom: vi.fn(),
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
  expect(viewer.setView).toHaveBeenCalledOnce();
  expect(viewer.setView.mock.calls[0][0]).toHaveLength(8);
});

it("takes its own zoom range and starting zoom", () => {
  const viewer = mockViewer();

  render(<MoleculeViewer sdf="x" zoom={{ start: 0.7, min: 0.5, max: 0.95 }} />);

  expect(viewer.setZoomLimits).toHaveBeenCalledWith(100 / 0.95, 200);
  expect(viewer.zoom).toHaveBeenCalledWith(0.7);
});

it("pans on right-button drag, following the mouse outside the tile but stopping at its edge", () => {
  const viewer = mockViewer();
  const { container } = render(<MoleculeViewer sdf="x" pan />);
  const tile = container.firstElementChild!;
  Object.defineProperty(tile, "clientWidth", { value: 200 });
  Object.defineProperty(tile, "clientHeight", { value: 100 });
  const canvas = document.createElement("canvas");
  tile.append(canvas);
  const onCanvas = vi.fn();
  canvas.addEventListener("mousedown", onCanvas);

  fireEvent.mouseDown(canvas, { button: 2, clientX: 100, clientY: 50 });
  expect(onCanvas).not.toHaveBeenCalled();

  fireEvent.mouseMove(window, { clientX: 1100, clientY: 50 });
  expect(viewer.getView()[0]).toBe(100);
  fireEvent.mouseMove(window, { clientX: 150, clientY: 50 });
  expect(viewer.getView()[0]).toBe(50);

  fireEvent.mouseUp(window);
  fireEvent.mouseMove(window, { clientX: 500, clientY: 50 });
  expect(viewer.getView()[0]).toBe(50);
});

it("leaves 3Dmol's own mouse handling alone without pan", () => {
  mockViewer();
  const { container } = render(<MoleculeViewer sdf="x" />);
  const canvas = document.createElement("canvas");
  container.firstElementChild!.append(canvas);
  const onCanvas = vi.fn();
  canvas.addEventListener("mousedown", onCanvas);

  fireEvent.mouseDown(canvas, { button: 2 });

  expect(onCanvas).toHaveBeenCalledOnce();
});
