import Card from "../../components/common/Card.tsx";
import PageHeader from "../../components/common/PageHeader.tsx";
import "./AboutPage.css";

const TEAM = [
  { initials: "VF", name: "Varvara Folimonova", tint: "var(--blue-100)" },
  { initials: "ED", name: "Eman Daraiseh", tint: "var(--violet-100)" },
  { initials: "AR", name: "Artem Rospotniuk", tint: "var(--green-100)" },
  { initials: "QM", name: "Qiyu Ma", tint: "var(--amber-100)" },
  { initials: "IR", name: "Ian Rodriguez-Page", tint: "var(--red-100)" },
];

const PALETTE = {
  blue: {
    fill: "var(--blue-50)",
    stroke: "var(--blue-500)",
    text: "var(--blue-700)",
  },
  cyan: {
    fill: "var(--cyan-50)",
    stroke: "var(--cyan-500)",
    text: "var(--cyan-700)",
  },
  slate: {
    fill: "var(--slate-100)",
    stroke: "var(--slate-300)",
    text: "var(--slate-900)",
  },
  green: {
    fill: "var(--green-50)",
    stroke: "var(--green-500)",
    text: "var(--green-700)",
  },
  red: {
    fill: "var(--red-50)",
    stroke: "var(--red-500)",
    text: "var(--red-700)",
  },
};

const NODES = [
  {
    x: 30,
    y: 20,
    w: 180,
    h: 60,
    color: PALETTE.blue,
    title: "Graph Encoder",
    sub: "GINE on atoms & bonds",
  },
  {
    x: 290,
    y: 20,
    w: 180,
    h: 60,
    color: PALETTE.blue,
    title: "Fingerprint Encoder",
    sub: "ECFP4 + MLP",
  },
  {
    x: 550,
    y: 20,
    w: 180,
    h: 60,
    color: PALETTE.cyan,
    title: "Sequence Encoder",
    sub: "MoLFormer on SMILES",
  },
  {
    x: 230,
    y: 136,
    w: 300,
    h: 56,
    color: PALETTE.slate,
    title: "Late Fusion",
    sub: "Concatenation + modality dropout",
  },
  {
    x: 70,
    y: 250,
    w: 200,
    h: 62,
    color: PALETTE.green,
    title: "Bioactivity Prediction",
    sub: "pIC50 + active / inactive",
  },
  {
    x: 490,
    y: 250,
    w: 200,
    h: 62,
    color: PALETTE.red,
    title: "Toxicity Prediction",
    sub: "Tox21 + ClinTox",
  },
];

const ARROWS = [
  [120, 80, 330, 134],
  [380, 80, 380, 134],
  [640, 80, 430, 134],
  [330, 192, 172, 248],
  [430, 192, 588, 248],
];

const DIAGRAM_LABEL =
  "Graph, fingerprint and sequence encoders feed a late-fusion layer, which feeds the bioactivity and toxicity prediction heads.";

export default function AboutPage() {
  return (
    <div className="about">
      <PageHeader
        title="About MuFu"
        subtitle="Multimodal Fusion for Molecular Property Prediction"
      />

      <div className="about-grid">
        <div className="about-side">
          <Card className="about-card">
            <h2>Our Goal</h2>
            <p className="about-goal">
              Develop a multimodal deep learning model to predict molecular
              bioactivity and toxicity, supporting early-stage drug discovery
              and chemical research.
            </p>
          </Card>

          <Card className="about-card">
            <h2>Team</h2>
            <ul className="about-team">
              {TEAM.map(({ initials, name, tint }) => (
                <li key={name}>
                  <span
                    className="about-avatar"
                    style={{ background: tint }}
                    aria-hidden="true"
                  >
                    {initials}
                  </span>
                  <div>
                    <strong>{name}</strong>
                    <span>Engineer</span>
                  </div>
                </li>
              ))}
            </ul>
          </Card>
        </div>

        <Card className="about-card about-model">
          <h2>Model Overview</h2>
          <figure>
            <svg viewBox="0 0 760 330" role="img" aria-label={DIAGRAM_LABEL}>
              <defs>
                <marker
                  id="about-arrow"
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="7"
                  markerHeight="7"
                  orient="auto-start-reverse"
                >
                  <path
                    d="M0 0 L10 5 L0 10 z"
                    style={{ fill: "var(--slate-400)" }}
                  />
                </marker>
              </defs>
              {ARROWS.map(([x1, y1, x2, y2]) => (
                <line
                  key={`${x1}-${y1}-${x2}`}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  style={{ stroke: "var(--slate-400)" }}
                  strokeWidth="1.5"
                  markerEnd="url(#about-arrow)"
                />
              ))}
              {NODES.map(({ x, y, w, h, color, title, sub }) => (
                <g key={title}>
                  <rect
                    x={x}
                    y={y}
                    width={w}
                    height={h}
                    rx="8"
                    style={{ fill: color.fill, stroke: color.stroke }}
                    strokeWidth="1.5"
                  />
                  <text
                    x={x + w / 2}
                    y={y + h / 2 - 7}
                    style={{ fill: color.text }}
                    className="about-node-title"
                  >
                    {title}
                  </text>
                  <text
                    x={x + w / 2}
                    y={y + h / 2 + 11}
                    style={{ fill: color.text }}
                    className="about-node-sub"
                  >
                    {sub}
                  </text>
                </g>
              ))}
            </svg>
          </figure>
        </Card>
      </div>
    </div>
  );
}
