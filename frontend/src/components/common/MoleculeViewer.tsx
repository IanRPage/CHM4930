import { useEffect, useRef } from "react";
import * as $3Dmol from "3dmol";
import "./MoleculeViewer.css";
import { defaultOrientation } from "./orientation.ts";
import { ZOOM_LIMITS } from "./zoom.ts";

interface MoleculeViewerProps {
  sdf: string;
  pan?: boolean;
  // Zoom limits as multiples of the fitted view, and the zoom it starts at
  zoom?: { start?: number; min: number; max: number };
}

function clamp(value: number, limit: number) {
  return Math.min(limit, Math.max(-limit, value));
}

export default function MoleculeViewer({
  sdf,
  pan,
  zoom = ZOOM_LIMITS,
}: MoleculeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let viewer: ReturnType<typeof $3Dmol.createViewer>;
    try {
      viewer = $3Dmol.createViewerGrid(
        container,
        // control_all lets 3Dmol act on drag moves forwarded from outside the canvas
        { rows: 1, cols: 1, control_all: true },
        {
          backgroundColor: "white",
          backgroundAlpha: 0,
          antialias: true,
        },
      )[0][0];

      viewer.setBackgroundColor(0xffffff, 0.0);

      viewer.addModel(sdf, "sdf");

      // Ball-and-stick representation
      viewer.setStyle(
        {},
        {
          stick: { radius: 0.15 },
          sphere: { scale: 0.3 },
        },
      );

      viewer.resize();
      viewer.zoomTo();
      const atoms = viewer.getModel().selectedAtoms({});
      const [x, y, z, distance] = viewer.getView();
      viewer.setView([
        x,
        y,
        z,
        distance,
        ...defaultOrientation(
          atoms.map((a) => ({ x: a.x ?? 0, y: a.y ?? 0, z: a.z ?? 0 })),
        ),
      ]);
      // fitted distance scales with molecule, limits do too
      const fit = viewer.getPerceivedDistance();
      viewer.setZoomLimits(fit / zoom.max, fit / zoom.min);
      if (zoom.start) viewer.zoom(zoom.start);
      viewer.render();
    } catch (error) {
      console.error("Molecule viewer error:", error);
      return;
    }

    // After zoomTo the model sits at -center, which is the middle of the view
    const [cx, cy, cz] = viewer.getView();
    const center = { x: -cx, y: -cy, z: -cz };

    // Keep the molecule's center inside the tile, so it can't be panned away
    const clampPan = () => {
      const view = viewer.getView();
      const [at, middle] = viewer.modelToScreen([
        center,
        { x: -view[0], y: -view[1], z: -view[2] },
      ]);
      const dx = at.x - middle.x;
      const dy = at.y - middle.y;
      const fixX = clamp(dx, container.clientWidth / 2) - dx;
      const fixY = clamp(dy, container.clientHeight / 2) - dy;
      if (!fixX && !fixY) return;
      const t = viewer.screenOffsetToModel(fixX, fixY);
      viewer.setView([
        view[0] + t.x,
        view[1] + t.y,
        view[2] + t.z,
        ...view.slice(3),
      ]);
    };

    // 3Dmol only sees mouse moves over its canvas, so a drag that leaves the tile
    // stalls and then jumps. Track the whole drag on window instead.
    let stopDrag: (() => void) | null = null;
    function trackDrag(move: (e: MouseEvent) => void) {
      stopDrag?.();
      stopDrag = () => {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", stopDrag!);
        stopDrag = null;
      };
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", stopDrag);
    }

    function onMouseDown(ev: MouseEvent) {
      const rotate = ev.button === 0 && !ev.ctrlKey;
      if (rotate) {
        // 3Dmol handles rotation itself; give it the moves it can't see
        const canvas = container!.querySelector("canvas");
        trackDrag((e) => {
          if (e.target !== canvas)
            canvas?.dispatchEvent(new MouseEvent("mousemove", e));
        });
        return;
      }
      if (!pan) return;
      ev.stopPropagation();
      const x0 = ev.clientX;
      const y0 = ev.clientY;
      const view0 = viewer.getView();
      trackDrag((e) => {
        const t = viewer.screenOffsetToModel(e.clientX - x0, e.clientY - y0);
        viewer.setView([
          view0[0] + t.x,
          view0[1] + t.y,
          view0[2] + t.z,
          ...view0.slice(3),
        ]);
        clampPan();
      });
    }

    container.addEventListener("mousedown", onMouseDown, true);
    if (pan) container.addEventListener("wheel", clampPan);
    return () => {
      container.removeEventListener("mousedown", onMouseDown, true);
      container.removeEventListener("wheel", clampPan);
      stopDrag?.();
      viewer.clear();
      // clear() leaves its canvas behind, which would stack up on re-render
      container.replaceChildren();
    };
  }, [sdf, pan, zoom]);

  return <div ref={containerRef} />;
}
