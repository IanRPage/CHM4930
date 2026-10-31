import { useEffect, useRef, useState } from "react";
import type { CSSProperties, FormEvent } from "react";
import approvalIcon from "../assets/icon-approval.svg";
import bioactivityIcon from "../assets/icon-bioactivity.svg";
import infoMutedIcon from "../assets/icon-info-muted.svg";
import infoIcon from "../assets/icon-info.svg";
import toxicityIcon from "../assets/icon-toxicity.svg";
import { PredictError } from "../services/errors.ts";
import { predict } from "../services/predict.ts";
import type {
  Output,
  PredictResponse,
  Target,
  Value,
} from "../types/predict.ts";
import { NR_TASKS, SR_TASKS } from "../types/predict.ts";
import "./SinglePredictionPage.css";

type Tab = "input" | "results";

const TARGET_ORDER: Target[] = ["EGFR", "BACE1"];

// pIC50 scale shown on the bioactivity bar
const PIC50_MIN = 2;
const PIC50_MAX = 12;
const MODERATE_RISK = 0.3;
const HIGH_RISK = 0.5;
const FAVORABLE = 0.5;

function tone(output: Output) {
  return output.value !== null && output.value >= FAVORABLE
    ? "predict-panel-good"
    : "predict-panel-neutral";
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function scalePos(pic50: number, min = 0, max = 1) {
  const t = (pic50 - PIC50_MIN) / (PIC50_MAX - PIC50_MIN);
  return `${clamp(t, min, max) * 100}%`;
}

const CONCENTRATION_UNITS = [
  ["nM", 1],
  ["µM", 1e3],
  ["mM", 1e6],
  ["M", 1e9],
] as const;

function concentration(nm: number) {
  const [unit, scale] =
    CONCENTRATION_UNITS.findLast(([, s]) => nm >= s) ?? CONCENTRATION_UNITS[0];
  return `${Number((nm / scale).toPrecision(3))} ${unit}`;
}

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function riskLevel(p: number) {
  if (p >= HIGH_RISK) return "high";
  if (p >= MODERATE_RISK) return "moderate";
  return "low";
}

function Stat<T extends Value>({
  label,
  output,
  format,
  detail,
}: {
  label: string;
  output: Output<T>;
  format: (output: T) => string;
  detail: (output: T) => string;
}) {
  return (
    <div className="predict-stat">
      <span className="predict-stat-label">{label}</span>
      {output.value === null ? (
        <>
          <span className="predict-stat-value predict-na">—</span>
          <span className="predict-stat-detail">{output.reason}</span>
        </>
      ) : (
        <>
          <span className="predict-stat-value">{format(output as T)}</span>
          <span className="predict-stat-detail">{detail(output as T)}</span>
        </>
      )}
    </div>
  );
}

function Bioactivity({ result }: { result: PredictResponse }) {
  const [target, setTarget] = useState<Target>("EGFR");
  const { pic50, ic50Nm, pActive } = result.bioactivity[target];
  const { mcDropoutPasses, ciCoverage } = result.metadata;
  const ci = ciCoverage === null ? "CI" : `${percent(ciCoverage)} CI`;

  return (
    <section className={`predict-panel ${tone(pActive)}`}>
      <div className="predict-panel-head">
        <img src={bioactivityIcon} alt="" />
        <h3>Bioactivity</h3>
        <div className="predict-segmented" role="group" aria-label="Target">
          {TARGET_ORDER.map((t) => (
            <button
              key={t}
              type="button"
              aria-pressed={t === target}
              onClick={() => setTarget(t)}
            >
              {t}
            </button>
          ))}
        </div>
        {mcDropoutPasses !== null && (
          <span className="predict-pill predict-pill-good">
            Confidence via MC-Dropout · {mcDropoutPasses} passes
          </span>
        )}
      </div>

      <div className="predict-stats">
        <Stat
          label="pIC50"
          output={pic50}
          format={(o) => o.value.toFixed(2)}
          detail={(o) => `${ci} ${o.lo.toFixed(2)} – ${o.hi.toFixed(2)}`}
        />
        <Stat
          label="IC50"
          output={ic50Nm}
          format={(o) => concentration(o.value)}
          detail={(o) =>
            `${ci} ${concentration(o.lo)} – ${concentration(o.hi)}`
          }
        />
        <Stat
          label="P(active)"
          output={pActive}
          format={(o) => o.value.toFixed(2)}
          detail={() => "pIC50 ≥ 6.0"}
        />
      </div>

      {pic50.value !== null && (
        <div className="predict-scale" aria-hidden="true">
          <div className="predict-scale-bar">
            <div
              className="predict-scale-fill"
              style={{ width: scalePos(pic50.value) }}
            />
            <div
              className="predict-scale-interval"
              style={
                {
                  "--lo": scalePos(pic50.lo),
                  "--hi": scalePos(pic50.hi),
                } as CSSProperties
              }
            />
            <div
              className="predict-scale-point"
              style={{ left: scalePos(pic50.value) }}
            />
          </div>
          <div className="predict-scale-labels">
            <span>Lower activity</span>
            <span
              className="predict-scale-ci"
              style={{ left: scalePos((pic50.lo + pic50.hi) / 2, 0.25, 0.75) }}
            >
              {ci.replace(" CI", "")} interval
            </span>
            <span>Higher activity</span>
          </div>
        </div>
      )}
    </section>
  );
}

function RiskRow({ task, output }: { task: string; output: Output }) {
  if (output.value === null) {
    return (
      <li className="predict-risk" title={output.reason}>
        <span>{task}</span>
        <span className="predict-risk-track" />
        <span className="predict-risk-value predict-na">—</span>
      </li>
    );
  }
  return (
    <li className={`predict-risk predict-risk-${riskLevel(output.value)}`}>
      <span>{task}</span>
      <span className="predict-risk-track">
        <span
          className="predict-risk-fill"
          style={{ width: percent(output.value) }}
        />
      </span>
      <span className="predict-risk-value">{percent(output.value)}</span>
    </li>
  );
}

function Toxicity({ result }: { result: PredictResponse }) {
  const { nuclearReceptor, stressResponse } = result.toxicity;
  const outputs = [
    ...Object.values(nuclearReceptor),
    ...Object.values(stressResponse),
  ];
  const high = outputs.filter(
    (o) => o.value !== null && o.value >= HIGH_RISK,
  ).length;

  return (
    <section className="predict-panel">
      <div className="predict-panel-head">
        <img src={toxicityIcon} alt="" />
        <h3>Toxicity risk</h3>
        <span
          className={`predict-pill ${
            high ? "predict-pill-bad" : "predict-pill-good"
          }`}
        >
          {high} of {outputs.length} high risk
        </span>
      </div>
      <ul className="predict-legend">
        <li className="predict-risk-low">Low &lt; 30%</li>
        <li className="predict-risk-moderate">Moderate 30–50%</li>
        <li className="predict-risk-high">High ≥ 50%</li>
      </ul>
      <div className="predict-risk-columns">
        <div>
          <h4>Nuclear receptor (Tox21)</h4>
          <ul>
            {NR_TASKS.map((t) => (
              <RiskRow key={t} task={t} output={nuclearReceptor[t]} />
            ))}
          </ul>
        </div>
        <div>
          <h4>Stress response (Tox21)</h4>
          <ul>
            {SR_TASKS.map((t) => (
              <RiskRow key={t} task={t} output={stressResponse[t]} />
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

function Approval({ output }: { output: Output }) {
  return (
    <section className={`predict-panel ${tone(output)} predict-approval`}>
      <img src={approvalIcon} alt="" />
      <div>
        <h3>Likelihood of FDA approval</h3>
        <p>Predicted from ClinTox · higher is better</p>
      </div>
      {output.value === null ? (
        <span
          className="predict-approval-value predict-na"
          title={output.reason}
        >
          —
        </span>
      ) : (
        <>
          <span className="predict-approval-track">
            <span
              className="predict-approval-fill"
              style={{ width: percent(output.value) }}
            />
          </span>
          <span className="predict-approval-value">
            {percent(output.value)}
          </span>
        </>
      )}
    </section>
  );
}

function Results({ result }: { result: PredictResponse }) {
  return (
    <div className="predict-results">
      {result.smiles && (
        <p className="predict-smiles">
          <strong>SMILES:</strong> <code>{result.smiles}</code>
        </p>
      )}
      <section className="card predict-properties">
        <h2>Predicted Properties</h2>
        <Bioactivity result={result} />
        <Toxicity result={result} />
        <Approval output={result.fdaApproved} />
        <p className="predict-note">
          <img src={infoMutedIcon} alt="" />
          Predictions are model estimates and should be validated
          experimentally.
        </p>
      </section>
    </div>
  );
}

export default function SinglePredictionPage() {
  const [smiles, setSmiles] = useState("");
  const [tab, setTab] = useState<Tab>("input");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => controller.current?.abort(), []);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
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
        <form className="card predict-input" onSubmit={onSubmit}>
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
              onChange={(event) => setSmiles(event.target.value)}
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
      )}

      {tab === "results" && result && <Results result={result} />}
    </div>
  );
}
