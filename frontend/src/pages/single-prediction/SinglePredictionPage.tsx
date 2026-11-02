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
  const [error, setError] = useState<PredictError | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const controller = useRef<AbortController | null>(null);
  const resultsTab = useRef<HTMLButtonElement | null>(null);

  useEffect(() => () => controller.current?.abort(), []);

  useEffect(() => {
    // A new result triggers the automatic tab switch; manual navigation does not.
    if (result) resultsTab.current?.focus();
  }, [result]);

  async function onSubmit() {
    // The ref also guards submissions before React renders the disabled button.
    if (controller.current) return;
    setError(null);
    setResult(null);
    const input = smiles.trim();
    if (!input) {
      setError(
        new PredictError(
          "invalid-smiles",
          "Enter a SMILES string to predict a molecule.",
        ),
      );
      return;
    }

    const current = new AbortController();
    controller.current = current;
    setLoading(true);
    try {
      const prediction = await predict(input, { signal: current.signal });
      if (current.signal.aborted || controller.current !== current) return;
      setResult(prediction);
      setTab("results");
    } catch (err) {
      if (current.signal.aborted || controller.current !== current) return;
      if (
        (err instanceof DOMException || err instanceof Error) &&
        err.name === "AbortError"
      )
        return;
      setError(
        err instanceof PredictError
          ? err
          : new PredictError(
              "server",
              "Something went wrong. Please try again.",
            ),
      );
    } finally {
      if (controller.current === current && !current.signal.aborted) {
        controller.current = null;
        setLoading(false);
      }
    }
  }

  function onSmilesChange(value: string) {
    controller.current?.abort();
    controller.current = null;
    setLoading(false);
    setSmiles(value);
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
          ref={resultsTab}
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
          onSmilesChange={onSmilesChange}
          loading={loading}
          error={error}
          onSubmit={onSubmit}
          onClear={() => onSmilesChange("")}
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
