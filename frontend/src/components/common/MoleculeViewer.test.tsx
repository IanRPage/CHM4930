import { render, waitFor } from "@testing-library/react";
import { createViewer } from "3dmol";
import { afterEach, expect, it, vi } from "vitest";
import MoleculeViewer from "./MoleculeViewer.tsx";

vi.mock("3dmol", () => ({ createViewer: vi.fn() }));

afterEach(() => {
  vi.unstubAllGlobals();
  vi.resetAllMocks();
});

it("successfully loads and renders a molecule from an SDF URL", async () => {
  const viewer = {
    addModel: vi.fn(),
    setStyle: vi.fn(),
    resize: vi.fn(),
    zoomTo: vi.fn(),
    render: vi.fn(),
    clear: vi.fn(),
  };
  vi.mocked(createViewer).mockReturnValue(
    viewer as unknown as ReturnType<typeof createViewer>,
  );
  const fetchMock = vi.fn().mockResolvedValue(new Response("molecule SDF"));
  vi.stubGlobal("fetch", fetchMock);

  render(<MoleculeViewer sdfUrl="/molecules/ethanol.sdf" />);

  await waitFor(() => expect(viewer.render).toHaveBeenCalledOnce());
  expect(fetchMock).toHaveBeenCalledWith("/molecules/ethanol.sdf");
  expect(viewer.addModel).toHaveBeenCalledWith("molecule SDF", "sdf");
});
