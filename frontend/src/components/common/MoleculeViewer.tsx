import { useEffect, useRef } from "react";
import * as $3Dmol from "3dmol";
import "./MoleculeViewer.css";

interface MoleculeViewerProps {
  sdf: string;
}

export default function MoleculeViewer({ sdf }: MoleculeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    let viewer: ReturnType<typeof $3Dmol.createViewer>;
    try {
      viewer = $3Dmol.createViewerGrid(
        containerRef.current,
        { rows: 1, cols: 1 },
        {
          backgroundColor: "white",
          backgroundAlpha: 0,
          antialias: true,
        },
      )[0][0];

      viewer.setBackgroundColor(0xffffff, 0.0);

      viewer.addModel(sdf, "sdf");

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
      console.error("Molecule viewer error:", error);
      return;
    }

    return () => {
      viewer.clear();
    };
  }, [sdf]);

  return <div ref={containerRef} />;
}
