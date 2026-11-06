import { useEffect, useRef, useState } from "react";
import { ZOOM_LIMITS } from "../../../components/common/zoom.ts";

// Wheel zooms, drag pans
export default function PanZoomImage({
  src,
  alt,
}: {
  src: string;
  alt: string;
}) {
  const frame = useRef<HTMLDivElement>(null);
  const last = useRef<{ x: number; y: number } | null>(null);
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
          Math.max(ZOOM_LIMITS.min, v.scale * Math.exp(-ev.deltaY * 0.002)),
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
        last.current = { x: ev.clientX, y: ev.clientY };
      }}
      onPointerMove={(ev) => {
        if (!last.current) return;
        const dx = ev.clientX - last.current.x;
        const dy = ev.clientY - last.current.y;
        last.current = { x: ev.clientX, y: ev.clientY };
        setView((v) => ({ ...v, x: v.x + dx, y: v.y + dy }));
      }}
      onPointerUp={() => (last.current = null)}
      onPointerCancel={() => (last.current = null)}
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
