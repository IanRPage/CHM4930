import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import PanZoomImage from "./PanZoomImage.tsx";

beforeEach(() => {
  HTMLElement.prototype.setPointerCapture = vi.fn();
});

function renderImage() {
  render(<PanZoomImage src="data:image/svg+xml," alt="molecule" />);
  return screen.getByRole("img");
}

it("zooms with the wheel within the zoom limits", () => {
  const img = renderImage();
  const frame = img.parentElement!;

  fireEvent.wheel(frame, { deltaY: -10_000 });
  expect(img.style.transform).toContain("scale(4)");
  fireEvent.wheel(frame, { deltaY: 10_000 });
  expect(img.style.transform).toContain("scale(0.5)");
});

it("pans by dragging", () => {
  const img = renderImage();
  const frame = img.parentElement!;

  fireEvent.pointerDown(frame, { clientX: 10, clientY: 10 });
  fireEvent.pointerMove(frame, { clientX: 40, clientY: 25 });
  fireEvent.pointerUp(frame);
  fireEvent.pointerMove(frame, { clientX: 90, clientY: 90 });

  expect(img.style.transform).toContain("translate(30px, 15px)");
});
