import { useEffect, useRef } from "react";
import type { FormEvent } from "react";
import infoIcon from "../../../assets/icon-info.svg";
import Card from "../../../components/common/Card.tsx";
import type { PredictError } from "../../../services/errors.ts";
import "./InputMolecule.css";

export default function InputMolecule({
  smiles,
  onSmilesChange,
  loading,
  error,
  onSubmit,
  onClear,
}: {
  smiles: string;
  onSmilesChange: (smiles: string) => void;
  loading: boolean;
  error: PredictError | null;
  onSubmit: () => void;
  onClear: () => void;
}) {
  const input = useRef<HTMLInputElement | null>(null);
  const blankError = error?.kind === "invalid-smiles" && !smiles.trim();

  useEffect(() => {
    if (blankError) input.current?.focus();
  }, [blankError, error]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <Card as="form" className="predict-input" onSubmit={handleSubmit}>
      <label htmlFor="smiles">
        Enter SMILES
        <img
          src={infoIcon}
          alt=""
          title="Simplified Molecular Input Line Entry System"
        />
      </label>
      <div className="predict-input-row">
        <input
          ref={input}
          id="smiles"
          value={smiles}
          onChange={(event) => onSmilesChange(event.target.value)}
          placeholder="e.g. CCOc1ccc(cc1)N"
          spellCheck={false}
          autoComplete="off"
          aria-invalid={error?.kind === "invalid-smiles"}
          aria-describedby={error ? "smiles-error" : undefined}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Predicting…" : "Predict"}
        </button>
        <button
          type="button"
          className="predict-clear"
          disabled={loading}
          onClick={onClear}
        >
          Clear
        </button>
      </div>
      <p
        className="predict-status"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {loading ? "Predicting molecule properties. Please wait…" : ""}
      </p>
      {error && (
        // Blank validation is announced through the focused input's description.
        <p
          id="smiles-error"
          className="predict-error"
          role={blankError ? undefined : "alert"}
        >
          {error.message}
        </p>
      )}
    </Card>
  );
}
