import { useEffect, useRef } from "react";
import * as $3Dmol from "3dmol";
import "./MoleculeViewer.css";

interface MoleculeViewerProps {
  sdfUrl: string;
}

export default function MoleculeViewer({ sdfUrl }: MoleculeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    let cancelled = false;

    const viewer = $3Dmol.createViewer(containerRef.current, {
      backgroundColor: "white",
      backgroundAlpha: 0,
      antialias: true,
    });

    async function loadMolecule() {
      try {
        const response = await fetch(sdfUrl);
        if (!response.ok) {
          throw new Error("Failed to load molecule");
        }

        const sdf = await response.text();
        if (cancelled) return;

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
        viewer.render();
      } catch (error) {
        if (!cancelled) {
          console.error("Molecule loading error:", error);
        }
      }
    }

    loadMolecule();

    return () => {
      cancelled = true;
      viewer.clear();
    };
  }, [sdfUrl]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
      }}
    />
  );
}
