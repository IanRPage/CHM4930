import { Link } from "react-router-dom";
import heroMolecule from "../assets/hero-molecule.svg";
import arrowRight from "../assets/icon-arrow-right.svg";
import batchIcon from "../assets/icon-batch.svg";
import boltIcon from "../assets/icon-bolt.svg";
import exploreIcon from "../assets/icon-explore.svg";
import leafIcon from "../assets/icon-leaf.svg";
import singleIcon from "../assets/icon-single-prediction.svg";
import targetIcon from "../assets/icon-target.svg";
import logo from "../assets/mufu-logo-light.svg";
import "./HomePage.css";

type EntryCard = {
  title: string;
  description: string;
  icon: string;
  tint: string;
  accent: string;
  action: string;
  to?: string;
};

const ENTRY_CARDS: EntryCard[] = [
  {
    title: "Single Molecule Prediction",
    description:
      "Input a molecule (SMILES or draw structure) to predict bioactivity and toxicity.",
    icon: singleIcon,
    tint: "#e8f0ff",
    accent: "#2563eb",
    action: "Get Started",
    to: "/predict",
  },
  {
    title: "Batch Prediction",
    description:
      "Upload a file with multiple molecules for large-scale screening.",
    icon: batchIcon,
    tint: "#f1ebff",
    accent: "#7c3aed",
    action: "Upload File",
  },
  {
    title: "Explore Dataset",
    description:
      "View dataset statistics and molecular properties used in the model.",
    icon: exploreIcon,
    tint: "#e3f7ee",
    accent: "#10b981",
    action: "Explore",
  },
];

const FEATURES = [
  {
    title: "Faster Screening",
    caption: "Quick prediction for many molecules",
    icon: boltIcon,
    tint: "#e8f0ff",
  },
  {
    title: "Multi-property Prediction",
    caption: "Bioactivity and toxicity",
    icon: targetIcon,
    tint: "#f1ebff",
  },
  {
    title: "Support Drug Discovery",
    caption: "Help researchers find better candidates",
    icon: leafIcon,
    tint: "#e3f7ee",
  },
];

export default function HomePage() {
  return (
    <div className="home">
      <section className="home-hero">
        <div>
          <div className="home-title">
            <img src={logo} alt="" width={84} height={84} />
            <h1>MuFu</h1>
          </div>
          <p className="home-subtitle">
            Multimodal Fusion for Molecular Property Prediction
          </p>
          <p className="home-lede">
            Predict bioactivity and toxicity from molecular structures using a
            multimodal deep learning model.
          </p>
        </div>
        <img
          className="home-graphic"
          src={heroMolecule}
          alt=""
          width={300}
          height={230}
        />
      </section>

      <section className="home-cards">
        {ENTRY_CARDS.map(
          ({ title, description, icon, tint, accent, action, to }) => (
            <article key={title} className="card home-card">
              <span className="home-chip" style={{ background: tint }}>
                <img src={icon} alt="" width={30} height={30} />
              </span>
              <div className="home-card-body">
                <h2>{title}</h2>
                <p>{description}</p>
                {to ? (
                  <Link
                    className="home-btn"
                    to={to}
                    style={{ background: accent }}
                  >
                    {action}
                    <img src={arrowRight} alt="" width={16} height={16} />
                  </Link>
                ) : (
                  <span
                    className="home-btn home-btn-soon"
                    aria-disabled="true"
                    style={{ color: accent, background: tint }}
                  >
                    Coming soon
                  </span>
                )}
              </div>
            </article>
          ),
        )}
      </section>

      <section className="home-features">
        {FEATURES.map(({ title, caption, icon, tint }) => (
          <div key={title} className="home-feature">
            <span
              className="home-chip home-chip-sm"
              style={{ background: tint }}
            >
              <img src={icon} alt="" width={22} height={22} />
            </span>
            <div>
              <h3>{title}</h3>
              <p>{caption}</p>
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
