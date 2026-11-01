import { useEffect, useRef, useState } from "react";
import Card from "../../components/common/Card.tsx";
import PageHeader from "../../components/common/PageHeader.tsx";
import Pill from "../../components/common/Pill.tsx";
import { PredictError } from "../../services/errors.ts";
import { predict } from "../../services/predict.ts";
import type { PredictResponse } from "../../types/predict.ts";
import InputMolecule from "./input-molecule/InputMolecule.tsx";
import PredictionResults from "./prediction-results/PredictionResults.tsx";
import "./SinglePredictionPage.css";

type Tab = "input" | "results" | "info";

const HEADERS: Record<Tab, { title: string; subtitle: string }> = {
  input: {
    title: "Single Molecule Prediction",
    subtitle: "Input a molecule to predict bioactivity and toxicity",
  },
  results: {
    title: "Prediction Results",
    subtitle: "Model predictions for the input molecule",
  },
  info: {
    title: "Molecular Information",
    subtitle: "Structure, identifiers and computed descriptors",
  },
};

export default function SinglePredictionPage() {
  const [smiles, setSmiles] = useState("");
  const [tab, setTab] = useState<Tab>("input");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => controller.current?.abort(), []);

  async function onSubmit() {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await predict(smiles, { signal: current.signal }));
      setTab("results");
    } catch (err) {
      if (current.signal.aborted) return;
      setError(
        err instanceof PredictError ? err.message : "Something went wrong",
      );
    }
    setLoading(false);
  }

  function onClear() {
    setSmiles("");
    setError(null);
    setResult(null);
  }

  const header = HEADERS[tab];

  return (
    <div className="predict">
      <PageHeader title={header.title} subtitle={header.subtitle} />

      <div className="predict-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "input"}
          onClick={() => setTab("input")}
        >
          Input Molecule
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "results"}
          disabled={!result}
          onClick={() => setTab("results")}
        >
          Prediction Results
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "info"}
          disabled={!result}
          onClick={() => setTab("info")}
        >
          Molecular Information
        </button>
      </div>

      {tab === "input" && (
        <InputMolecule
          smiles={smiles}
          onSmilesChange={setSmiles}
          loading={loading}
          error={error}
          onSubmit={onSubmit}
          onClear={onClear}
        />
      )}

      {tab === "results" && result && <PredictionResults result={result} />}

      {tab === "info" && result && (
        <Card className="predict-placeholder">
          <Pill tone="neutral">Coming Soon</Pill>
          <p>
            Molecular structure, identifiers, and computed RDKit descriptors are
            under development.
          </p>
        </Card>
      )}
    </div>
  );
}
