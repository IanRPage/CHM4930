import "./AboutPage.css";

const TEAM = [
  { initials: "VF", name: "Varvara Folimonova", tint: "#dbeafe" },
  { initials: "ED", name: "Eman Daraiseh", tint: "#ede9fe" },
  { initials: "AR", name: "Artem Rospotniuk", tint: "#dcfce7" },
  { initials: "QM", name: "Qiyu Ma", tint: "#fef3c7" },
  { initials: "IR", name: "Ian Rodriguez-Page", tint: "#fee2e2" },
];

const PALETTE = {
  blue: { fill: "#eff6ff", stroke: "#3b82f6", text: "#1d4ed8" },
  cyan: { fill: "#ecfeff", stroke: "#06b6d4", text: "#0e7490" },
  slate: { fill: "#f1f5f9", stroke: "#cbd5e1", text: "#0f172a" },
  green: { fill: "#f0fdf4", stroke: "#22c55e", text: "#15803d" },
  red: { fill: "#fef2f2", stroke: "#ef4444", text: "#b91c1c" },
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
      <h1>About MuFu</h1>
      <p className="about-subtitle">
        Multimodal Fusion for Molecular Property Prediction
      </p>

      <div className="about-grid">
        <div className="about-side">
          <section className="card about-card">
            <h2>Our Goal</h2>
            <p className="about-goal">
              Develop a multimodal deep learning model to predict molecular
              bioactivity and toxicity, supporting early-stage drug discovery
              and chemical research.
            </p>
          </section>

          <section className="card about-card">
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
          </section>
        </div>

        <section className="card about-card about-model">
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
                  <path d="M0 0 L10 5 L0 10 z" fill="#94a3b8" />
                </marker>
              </defs>
              {ARROWS.map(([x1, y1, x2, y2]) => (
                <line
                  key={`${x1}-${y1}-${x2}`}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke="#94a3b8"
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
                    fill={color.fill}
                    stroke={color.stroke}
                    strokeWidth="1.5"
                  />
                  <text
                    x={x + w / 2}
                    y={y + h / 2 - 7}
                    fill={color.text}
                    className="about-node-title"
                  >
                    {title}
                  </text>
                  <text
                    x={x + w / 2}
                    y={y + h / 2 + 11}
                    fill={color.text}
                    className="about-node-sub"
                  >
                    {sub}
                  </text>
                </g>
              ))}
            </svg>
          </figure>
        </section>
      </div>
    </div>
  );
}
