import { useState } from "react";
import type { DragEvent } from "react";
import uploadIcon from "../../assets/icon-upload.svg";
import Card from "../../components/common/Card.tsx";
import {
  MAX_FILE_SIZE,
  MAX_ROWS,
  validateFile,
} from "../../services/batchParse.ts";
import { classNames } from "../../components/common/classNames.ts";
import { formatSize } from "./format.ts";
import "./UploadPanel.css";

const TEMPLATE =
  "smiles\nCC(=O)Oc1ccccc1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C\nCCO\n";

function downloadTemplate() {
  const url = URL.createObjectURL(new Blob([TEMPLATE], { type: "text/csv" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "mufu-batch-template.csv";
  link.click();
  URL.revokeObjectURL(url);
}

export default function UploadPanel({
  onRun,
  error,
}: {
  onRun: (file: File) => void;
  error: string | null;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [invalid, setInvalid] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  function choose(chosen: File | undefined) {
    if (!chosen) return;
    const problem = validateFile(chosen);
    setInvalid(problem);
    setFile(problem ? null : chosen);
  }

  function onDrop(ev: DragEvent) {
    ev.preventDefault();
    setDragging(false);
    choose(ev.dataTransfer.files[0]);
  }

  const message = invalid ?? error;

  return (
    <div className="batch-upload">
      <Card className="batch-upload-main">
        <label
          className={classNames("batch-drop", dragging && "batch-drop-active")}
          onDragOver={(ev) => {
            ev.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input
            type="file"
            accept=".csv,.sdf,.txt"
            aria-label="Choose a file"
            onChange={(ev) => choose(ev.target.files?.[0])}
          />
          <img src={uploadIcon} alt="" />
          {file ? (
            <>
              <strong>
                {file.name} ({formatSize(file.size)})
              </strong>
              <span className="batch-drop-check">✓ Ready for prediction</span>
            </>
          ) : (
            <>
              <strong>Drag and drop a file here</strong>
              <span>or click to browse</span>
            </>
          )}
          <small>Supported formats: CSV, SDF, TXT (SMILES)</small>
        </label>
        {message && (
          <p role="alert" className="batch-upload-error">
            {message}
          </p>
        )}
        <button
          type="button"
          className="batch-upload-run"
          disabled={!file}
          onClick={() => file && onRun(file)}
        >
          Run Prediction
        </button>
      </Card>
      <Card as="aside" className="batch-upload-info">
        <h2>File Requirements</h2>
        <ul>
          <li>One SMILES per row (the &quot;smiles&quot; column for CSV)</li>
          <li>
            Up to {MAX_ROWS.toLocaleString()} molecules and{" "}
            {MAX_FILE_SIZE / 1024 / 1024} MB per file
          </li>
          <li>Duplicate SMILES are predicted once</li>
          <li>Invalid SMILES are skipped and listed in the results</li>
        </ul>
        <button type="button" onClick={downloadTemplate}>
          Download template
        </button>
      </Card>
    </div>
  );
}
