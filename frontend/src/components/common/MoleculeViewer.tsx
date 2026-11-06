import { useEffect, useRef } from "react";
import * as $3Dmol from "3dmol";
import "./MoleculeViewer.css";
import { ZOOM_LIMITS } from "./zoom.ts";

interface MoleculeViewerProps {
  sdf: string;
  pan?: boolean;
}

// 3Dmol pans on middle-button drag only; make right-button drag do the same
function rightDragPans(ev: MouseEvent) {
  if (ev.button === 2) Object.defineProperty(ev, "which", { value: 2 });
}

export default function MoleculeViewer({ sdf, pan }: MoleculeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let viewer: ReturnType<typeof $3Dmol.createViewer>;
    try {
      viewer = $3Dmol.createViewerGrid(
        container,
        { rows: 1, cols: 1 },
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
      // The fitted distance scales with the molecule, so the limits do too
      const fit = viewer.getPerceivedDistance();
      viewer.setZoomLimits(fit / ZOOM_LIMITS.max, fit / ZOOM_LIMITS.min);
      viewer.render();
    } catch (error) {
      console.error("Molecule viewer error:", error);
      return;
    }

    if (pan) container.addEventListener("mousedown", rightDragPans, true);
    return () => {
      container.removeEventListener("mousedown", rightDragPans, true);
      viewer.clear();
    };
  }, [sdf, pan]);

  return <div ref={containerRef} />;
}
