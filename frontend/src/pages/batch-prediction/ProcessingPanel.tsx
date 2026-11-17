import Card from "../../components/common/Card.tsx";
import Pill from "../../components/common/Pill.tsx";
import { classNames } from "../../components/common/classNames.ts";
import type { BatchProgress } from "../../services/batchPredict.ts";
import { formatDuration, formatSize } from "./format.ts";
import "./ProcessingPanel.css";

const STEPS = [
  "Parse file",
  "Validate SMILES",
  "Predict properties",
  "Build results",
];

export default function ProcessingPanel({
  file,
  rows,
  progress,
  elapsedMs,
  onCancel,
}: {
  file: File;
  rows: number;
  progress: BatchProgress;
  elapsedMs: number;
  onCancel: () => void;
}) {
  const { processed, total, invalid } = progress;
  const percent = Math.round((processed / total) * 100);
  const active = processed === 0 ? 1 : 2;
  const eta =
    processed === 0
      ? null
      : formatDuration((elapsedMs / processed) * (total - processed));

  return (
    <Card className="batch-processing">
      <div className="batch-file">
        <div>
          <strong>{file.name}</strong>
          <span>
            {formatSize(file.size)} · {rows} molecules
          </span>
        </div>
        <Pill tone="good">Validated</Pill>
        <button type="button" aria-label="Remove file" onClick={onCancel}>
          ×
        </button>
      </div>

      <ol className="batch-steps">
        {STEPS.map((step, i) => (
          <li
            key={step}
            className={classNames(
              i < active && "batch-step-done",
              i === active && "batch-step-active",
            )}
          >
            <span>{i < active ? "✓" : i + 1}</span>
            {step}
          </li>
        ))}
      </ol>

      <div
        className="batch-progress"
        role="progressbar"
        aria-label="Prediction progress"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <span style={{ width: `${percent}%` }} />
      </div>
      <p className="batch-progress-text">
        <span>
          {processed} / {total} molecules processed ({percent}%)
        </span>
        {eta && <span>About {eta} remaining</span>}
      </p>

      <dl className="batch-validation">
        <div className="batch-valid">
          <dt>Valid</dt>
          <dd>{processed - invalid}</dd>
        </div>
        <div className="batch-invalid">
          <dt>Invalid (skipped)</dt>
          <dd>{invalid}</dd>
        </div>
        <div className="batch-duplicate">
          <dt>Duplicates</dt>
          <dd>{rows - total}</dd>
        </div>
      </dl>

      <button type="button" className="batch-cancel" onClick={onCancel}>
        Cancel
      </button>
    </Card>
  );
}
