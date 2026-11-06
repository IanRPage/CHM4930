import { useState } from "react";
import Card from "../../../components/common/Card.tsx";
import MoleculeViewer from "../../../components/common/MoleculeViewer.tsx";
import Pill from "../../../components/common/Pill.tsx";
import type { PredictError } from "../../../services/errors.ts";
import type { MolInfo } from "../../../types/molInfo.ts";
import "./MolecularInformation.css";
import PanZoomImage from "./PanZoomImage.tsx";

type Row = readonly [label: string, value: string];
type View = "2D" | "3D";

const DESCRIPTORS: readonly (readonly [string, (m: MolInfo) => string])[] = [
  ["Molecular weight (g/mol)", (m) => m.molecularWeight.toFixed(2)],
  ["Exact mass (Da)", (m) => m.exactMass.toFixed(4)],
  ["LogP", (m) => m.logP.toFixed(2)],
  ["TPSA (Å²)", (m) => m.tpsa.toFixed(2)],
  ["Molar refractivity", (m) => m.molarRefractivity.toFixed(2)],
  ["H-bond donors", (m) => String(m.hBondDonors)],
  ["H-bond acceptors", (m) => String(m.hBondAcceptors)],
  ["Rotatable bonds", (m) => String(m.rotatableBonds)],
  ["Heavy atoms", (m) => String(m.heavyAtoms)],
  ["Aromatic rings", (m) => String(m.aromaticRings)],
  ["Fraction sp³", (m) => m.fractionCsp3.toFixed(2)],
];

// same thresholds as backend's lipinskiViolations
const LIPINSKI: readonly (readonly [string, (m: MolInfo) => boolean])[] = [
  ["Molecular weight ≤ 500", (m) => m.molecularWeight <= 500],
  ["LogP ≤ 5", (m) => m.logP <= 5],
  ["H-bond donors ≤ 5", (m) => m.hBondDonors <= 5],
  ["H-bond acceptors ≤ 10", (m) => m.hBondAcceptors <= 10],
];

function List({
  rows,
  className,
}: {
  rows: readonly Row[];
  className: string;
}) {
  return (
    <dl className={className}>
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

const HINTS: Record<View, string> = {
  "2D": "Drag to pan · scroll to zoom",
  "3D": "Drag to rotate · right-drag to pan · scroll to zoom",
};

function Loaded({ info }: { info: MolInfo }) {
  const [view, setView] = useState<View>("2D");
  const views: View[] = info.molblock === null ? ["2D"] : ["2D", "3D"];
  const identifiers: Row[] = [
    ["Canonical SMILES", info.canonicalSmiles],
    ["Formula", info.formula],
  ];
  if (info.inchiKey !== null) identifiers.push(["InChIKey", info.inchiKey]);

  return (
    <div className="predict-info">
      <Card className="predict-info-structure">
        <div className="predict-info-view">
          {view === "3D" && info.molblock !== null ? (
            <MoleculeViewer sdf={info.molblock} pan />
          ) : (
            <PanZoomImage
              alt={`2D structure of ${info.canonicalSmiles}`}
              src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(
                info.svg,
              )}`}
            />
          )}
        </div>
        <div className="predict-info-structure-head">
          <h2>Structure</h2>
          <div className="predict-info-toggle">
            {views.map((v) => (
              <button
                key={v}
                type="button"
                aria-pressed={view === v}
                onClick={() => setView(v)}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
        <p className="predict-info-hint">{HINTS[view]}</p>
      </Card>
      <Card>
        <h2>Identifiers</h2>
        <List className="predict-info-list" rows={identifiers} />
      </Card>
      <Card>
        <h2>Descriptors</h2>
        <List
          className="predict-info-tiles"
          rows={DESCRIPTORS.map(([label, fn]) => [label, fn(info)])}
        />
      </Card>
      <Card>
        <h2>Drug-likeness</h2>
        <p className="predict-info-subtitle">
          Lipinski&apos;s rule of five · {info.lipinskiViolations} of 4 rules
          violated
        </p>
        <ul className="predict-info-rules">
          {LIPINSKI.map(([label, passes]) => (
            <li key={label}>
              {label}
              <Pill tone={passes(info) ? "good" : "bad"}>
                {passes(info) ? "Pass" : "Fail"}
              </Pill>
            </li>
          ))}
        </ul>
      </Card>
      <p className="predict-info-footer">
        Computed with RDKit and OpenChemLib.
      </p>
    </div>
  );
}

export default function MolecularInformation({
  info,
  loading,
  error,
}: {
  info: MolInfo | null;
  loading: boolean;
  error: PredictError | null;
}) {
  if (error)
    return (
      <p role="alert" className="predict-info-message">
        {error.message}
      </p>
    );
  if (loading || !info)
    return (
      <p role="status" className="predict-info-message">
        Loading molecular information…
      </p>
    );
  return <Loaded info={info} />;
}
