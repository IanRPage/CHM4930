import { useEffect, useRef, useState } from "react";
import { PredictError } from "../../services/errors.ts";
import { predict } from "../../services/predict.ts";
import type { PredictResponse } from "../../types/predict.ts";
import InputMolecule from "./input-molecule/InputMolecule.tsx";
import PredictionResults from "./prediction-results/PredictionResults.tsx";
import "./SinglePredictionPage.css";

type Tab = "input" | "results";

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

  return (
    <div className="predict">
      <header className="predict-header">
        <h1>
          {tab === "input"
            ? "Single Molecule Prediction"
            : "Prediction Results"}
        </h1>
        <p>
          {tab === "input"
            ? "Input a molecule to predict bioactivity and toxicity"
            : "Model predictions for the input molecule"}
        </p>
      </header>

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
      </div>

      {tab === "input" && (
        <InputMolecule
          smiles={smiles}
          onSmilesChange={setSmiles}
          loading={loading}
          error={error}
          onSubmit={onSubmit}
        />
      )}

      {tab === "results" && result && <PredictionResults result={result} />}
    </div>
  );
}
