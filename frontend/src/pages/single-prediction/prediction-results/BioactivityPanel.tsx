import { useState } from "react";
import type { CSSProperties } from "react";
import bioactivityIcon from "../../../assets/icon-bioactivity.svg";
import Panel from "../../../components/common/Panel.tsx";
import Pill from "../../../components/common/Pill.tsx";
import type {
  Output,
  PredictResponse,
  Target,
  Value,
} from "../../../types/predict.ts";
import "./BioactivityPanel.css";
import { percent, tone } from "./format.ts";

const TARGET_ORDER: Target[] = ["EGFR", "BACE1"];

// pIC50 scale shown on the bioactivity bar
const PIC50_MIN = 2;
const PIC50_MAX = 12;

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

export default function BioactivityPanel({
  result,
}: {
  result: PredictResponse;
}) {
  const [target, setTarget] = useState<Target>("EGFR");
  const { pic50, ic50Nm, pActive } = result.bioactivity[target];
  const { mcDropoutPasses, ciCoverage } = result.metadata;
  const ci = ciCoverage === null ? "CI" : `${percent(ciCoverage)} CI`;

  return (
    <Panel
      title="Bioactivity"
      icon={bioactivityIcon}
      tone={tone(pActive)}
      controls={
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
      }
      aside={
        mcDropoutPasses !== null && (
          <Pill tone="good">
            Confidence via MC-Dropout · {mcDropoutPasses} passes
          </Pill>
        )
      }
    >
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
    </Panel>
  );
}
