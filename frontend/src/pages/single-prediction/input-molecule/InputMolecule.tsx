import type { FormEvent } from "react";
import infoIcon from "../../../assets/icon-info.svg";
import "./InputMolecule.css";

export default function InputMolecule({
  smiles,
  onSmilesChange,
  loading,
  error,
  onSubmit,
}: {
  smiles: string;
  onSmilesChange: (smiles: string) => void;
  loading: boolean;
  error: string | null;
  onSubmit: () => void;
}) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form className="card predict-input" onSubmit={handleSubmit}>
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
          id="smiles"
          value={smiles}
          onChange={(event) => onSmilesChange(event.target.value)}
          placeholder="e.g. CCOc1ccc(cc1)N"
          spellCheck={false}
          autoComplete="off"
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? "Predicting…" : "Predict"}
        </button>
      </div>
      {error && (
        <p className="predict-error" role="alert">
          {error}
        </p>
      )}
    </form>
  );
}
