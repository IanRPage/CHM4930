import { useEffect, useId, useRef, useState } from "react";
import infoIcon from "../../../assets/icon-info-muted.svg";
import Card from "../../../components/common/Card.tsx";
import MoleculeViewer from "../../../components/common/MoleculeViewer.tsx";
import Pill from "../../../components/common/Pill.tsx";
import type { PredictError } from "../../../services/errors.ts";
import type { MolInfo } from "../../../types/molInfo.ts";
import ElementLegend from "./ElementLegend.tsx";
import "./MolecularInformation.css";
import PanZoomImage from "./PanZoomImage.tsx";

type Row = readonly [label: string, value: string, hint?: string];
type View = "2D" | "3D";

const DESCRIPTORS: readonly (readonly [
  label: string,
  value: (m: MolInfo) => string,
  hint: string,
])[] = [
  [
    "Molecular weight (g/mol)",
    (m) => m.molecularWeight.toFixed(2),
    "Sum of atomic masses using average isotope abundances.",
  ],
  [
    "Exact mass (Da)",
    (m) => m.exactMass.toFixed(4),
    "Mass using the most abundant isotope of each element (what mass spectrometers measure).",
  ],
  [
    "LogP",
    (m) => m.logP.toFixed(2),
    "Crippen estimate of lipophilicity (octanol-water partition). Higher means more fat-soluble.",
  ],
  [
    "TPSA (Å²)",
    (m) => m.tpsa.toFixed(2),
    "Topological polar surface area: the surface from polar atoms (H, N, O). Lower generally means better membrane permeability.",
  ],
  [
    "Molar refractivity",
    (m) => m.molarRefractivity.toFixed(2),
    "Crippen estimate of the molecule's volume and polarizability.",
  ],
  [
    "H-bond donors",
    (m) => String(m.hBondDonors),
    "N-H and O-H groups that can donate a hydrogen bond (Lipinski count).",
  ],
  [
    "H-bond acceptors",
    (m) => String(m.hBondAcceptors),
    "Nitrogen and oxygen atoms that can accept a hydrogen bond (Lipinski count).",
  ],
  [
    "Rotatable bonds",
    (m) => String(m.rotatableBonds),
    "Single bonds outside rings that can rotate. More means a more flexible molecule.",
  ],
  ["Heavy atoms", (m) => String(m.heavyAtoms), "Number of non-hydrogen atoms."],
  [
    "Aromatic rings",
    (m) => String(m.aromaticRings),
    "Number of aromatic rings.",
  ],
  [
    "Fraction sp³",
    (m) => m.fractionCsp3.toFixed(2),
    "Share of carbons that are sp³ hybridized. Higher means a more three-dimensional, less flat molecule.",
  ],
];

// same thresholds as backend's lipinskiViolations
const LIPINSKI: readonly (readonly [string, (m: MolInfo) => boolean])[] = [
  ["Molecular weight ≤ 500", (m) => m.molecularWeight <= 500],
  ["LogP ≤ 5", (m) => m.logP <= 5],
  ["H-bond donors ≤ 5", (m) => m.hBondDonors <= 5],
  ["H-bond acceptors ≤ 10", (m) => m.hBondAcceptors <= 10],
];

function Tip({ label, hint }: { label: string; hint: string }) {
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    function close(ev: PointerEvent) {
      if (!button.current?.contains(ev.target as Node)) setOpen(false);
    }
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);

  return (
    <button
      ref={button}
      type="button"
      className="predict-info-tip"
      aria-label={`About ${label}`}
      aria-describedby={id}
      data-open={open || undefined}
      onClick={() => setOpen((o) => !o)}
    >
      <img src={infoIcon} alt="" />
      <span role="tooltip" id={id}>
        {hint}
      </span>
    </button>
  );
}

function List({
  rows,
  className,
}: {
  rows: readonly Row[];
  className: string;
}) {
  return (
    <dl className={className}>
      {rows.map(([label, value, hint]) => (
        <div key={label}>
          <dt>
            {label}
            {hint && <Tip label={label} hint={hint} />}
          </dt>
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
    [
      "Canonical SMILES",
      info.canonicalSmiles,
      "Text encoding of the structure, canonicalized by RDKit. Other tools may write a different string for the same molecule.",
    ],
    [
      "Formula",
      info.formula,
      "Number of atoms of each element in the molecule (charge not shown).",
    ],
  ];
  if (info.inchiKey !== null)
    identifiers.push([
      "InChIKey",
      info.inchiKey,
      "27-character hash of the molecule's InChI. Handy for looking it up in databases like PubChem.",
    ]);

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
        {view === "3D" && <ElementLegend formula={info.formula} />}
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
          rows={DESCRIPTORS.map(([label, fn, hint]) => [label, fn(info), hint])}
        />
      </Card>
      <Card>
        <h2 className="predict-info-title">
          Drug-likeness
          <Tip
            label="Drug-likeness"
            hint="Lipinski's rule of five predicts whether a molecule is likely to be absorbed when taken by mouth. Each threshold exceeded is a violation. More than one makes poor absorption more likely."
          />
        </h2>
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
