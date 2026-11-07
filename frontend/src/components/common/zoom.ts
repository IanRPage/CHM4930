// Zoom range relative to the fitted view, shared by the 2D and 3D molecule views
export const ZOOM_LIMITS = { min: 0.5, max: 4 };

// Zoom multiplier for one wheel event: positive scroll zooms out
export function wheelZoomFactor(ev: WheelEvent) {
  // Firefox reports mouse-wheel notches in lines, not pixels
  const dy =
    ev.deltaMode === WheelEvent.DOM_DELTA_LINE ? ev.deltaY * 16 : ev.deltaY;
  return Math.exp(-dy * 0.002);
}
