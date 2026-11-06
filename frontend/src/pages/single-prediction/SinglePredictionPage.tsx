import { useEffect, useRef, useState } from "react";
import PageHeader from "../../components/common/PageHeader.tsx";
import { PredictError } from "../../services/errors.ts";
import { getMolInfo } from "../../services/molInfo.ts";
import { predict } from "../../services/predict.ts";
import type { MolInfo } from "../../types/molInfo.ts";
import type { PredictResponse } from "../../types/predict.ts";
import InputMolecule from "./input-molecule/InputMolecule.tsx";
import MolecularInformation from "./molecular-information/MolecularInformation.tsx";
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

function toPredictError(err: unknown) {
  return err instanceof PredictError
    ? err
    : new PredictError("server", "Something went wrong. Please try again.");
}

function isAbort(err: unknown) {
  return (
    (err instanceof DOMException || err instanceof Error) &&
    err.name === "AbortError"
  );
}

export default function SinglePredictionPage() {
  const [smiles, setSmiles] = useState("");
  const [tab, setTab] = useState<Tab>("input");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<PredictError | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [info, setInfo] = useState<MolInfo | null>(null);
  const [infoLoading, setInfoLoading] = useState(false);
  const [infoError, setInfoError] = useState<PredictError | null>(null);
  const controller = useRef<AbortController | null>(null);
  const infoController = useRef<AbortController | null>(null);
  const resultsTab = useRef<HTMLButtonElement | null>(null);

  useEffect(
    () => () => {
      controller.current?.abort();
      infoController.current?.abort();
    },
    [],
  );

  useEffect(() => {
    if (result) resultsTab.current?.focus();
  }, [result]);

  async function loadInfo(input: string) {
    infoController.current?.abort();
    const current = new AbortController();
    infoController.current = current;
    setInfo(null);
    setInfoError(null);
    setInfoLoading(true);
    try {
      const molInfo = await getMolInfo(input, { signal: current.signal });
      if (infoController.current !== current) return;
      setInfo(molInfo);
    } catch (err) {
      if (infoController.current !== current || isAbort(err)) return;
      setInfoError(toPredictError(err));
    } finally {
      if (infoController.current === current) setInfoLoading(false);
    }
  }

  async function onSubmit() {
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
    void loadInfo(input);
    try {
      const prediction = await predict(input, { signal: current.signal });
      if (current.signal.aborted || controller.current !== current) return;
      setResult(prediction);
      setTab("results");
    } catch (err) {
      if (current.signal.aborted || controller.current !== current) return;
      if (isAbort(err)) return;
      setError(toPredictError(err));
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
    infoController.current?.abort();
    infoController.current = null;
    setLoading(false);
    setSmiles(value);
    setError(null);
    setResult(null);
    setInfo(null);
    setInfoLoading(false);
    setInfoError(null);
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
          disabled={!(info || infoLoading || infoError)}
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

      {tab === "results" && result && (
        <PredictionResults result={result} molInfo={info} />
      )}

      {tab === "info" && (
        <MolecularInformation
          info={info}
          loading={infoLoading}
          error={infoError}
        />
      )}
    </div>
  );
}
