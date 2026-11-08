import { useEffect, useRef, useState } from "react";
import {
  wheelZoomFactor,
  ZOOM_LIMITS,
} from "../../../components/common/zoom.ts";

function clamp(value: number, limit: number) {
  return Math.min(limit, Math.max(-limit, value));
}

// Wheel zooms, drag pans. The image center can reach the frame's edge but not
// leave it, so the molecule can always be dragged back.
export default function PanZoomImage({
  src,
  alt,
}: {
  src: string;
  alt: string;
}) {
  const frame = useRef<HTMLDivElement>(null);
  const drag = useRef<{ px: number; py: number; x: number; y: number } | null>(
    null,
  );
  const [view, setView] = useState({ x: 0, y: 0, scale: 1 });

  useEffect(() => {
    const el = frame.current;
    if (!el) return;
    // React's onWheel is passive, so it can't stop the page from scrolling
    function onWheel(ev: WheelEvent) {
      ev.preventDefault();
      setView((v) => ({
        ...v,
        scale: Math.min(
          ZOOM_LIMITS.max,
          Math.max(ZOOM_LIMITS.min, v.scale * wheelZoomFactor(ev)),
        ),
      }));
    }
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  return (
    <div
      ref={frame}
      className="predict-info-pan"
      onPointerDown={(ev) => {
        ev.currentTarget.setPointerCapture(ev.pointerId);
        drag.current = { px: ev.clientX, py: ev.clientY, x: view.x, y: view.y };
      }}
      onPointerMove={(ev) => {
        const start = drag.current;
        if (!start) return;
        // Measured from where the drag began, so pushing past the edge and
        // coming back doesn't jump
        const { clientWidth, clientHeight } = ev.currentTarget;
        setView((v) => ({
          ...v,
          x: clamp(start.x + ev.clientX - start.px, clientWidth / 2),
          y: clamp(start.y + ev.clientY - start.py, clientHeight / 2),
        }));
      }}
      onPointerUp={() => (drag.current = null)}
      onPointerCancel={() => (drag.current = null)}
    >
      <img
        src={src}
        alt={alt}
        draggable={false}
        style={{
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
        }}
      />
    </div>
  );
}
