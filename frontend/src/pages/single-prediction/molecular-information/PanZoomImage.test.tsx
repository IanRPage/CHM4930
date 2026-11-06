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

it("zooms a line-mode wheel notch (Firefox) like its pixel equivalent", () => {
  const img = renderImage();

  fireEvent.wheel(img.parentElement!, { deltaY: -3, deltaMode: 1 });

  expect(img.style.transform).toMatch(/scale\(1\.10/);
});

it("pans by dragging and stops at the frame edge without jumping", () => {
  const img = renderImage();
  const frame = img.parentElement!;
  Object.defineProperty(frame, "clientWidth", { value: 200 });
  Object.defineProperty(frame, "clientHeight", { value: 100 });

  fireEvent.pointerDown(frame, { clientX: 10, clientY: 10 });
  fireEvent.pointerMove(frame, { clientX: 40, clientY: 25 });
  expect(img.style.transform).toContain("translate(30px, 15px)");

  fireEvent.pointerMove(frame, { clientX: 1010, clientY: 25 });
  expect(img.style.transform).toContain("translate(100px, 15px)");
  fireEvent.pointerMove(frame, { clientX: 70, clientY: 25 });
  expect(img.style.transform).toContain("translate(60px, 15px)");

  fireEvent.pointerUp(frame);
  fireEvent.pointerMove(frame, { clientX: 90, clientY: 90 });
  expect(img.style.transform).toContain("translate(60px, 15px)");
});
