import toxicityIcon from "../../../assets/icon-toxicity.svg";
import Meter from "../../../components/common/Meter.tsx";
import Panel from "../../../components/common/Panel.tsx";
import Pill from "../../../components/common/Pill.tsx";
import type { Output, PredictResponse } from "../../../types/predict.ts";
import { NR_TASKS, SR_TASKS } from "../../../types/predict.ts";
import "./ToxicityPanel.css";
import { percent } from "./format.ts";

const MODERATE_RISK = 0.3;
const HIGH_RISK = 0.5;

function riskLevel(p: number) {
  if (p >= HIGH_RISK) return "high";
  if (p >= MODERATE_RISK) return "moderate";
  return "low";
}

function RiskRow({ task, output }: { task: string; output: Output }) {
  if (output.value === null) {
    return (
      <li className="predict-risk" title={output.reason}>
        <span>{task}</span>
        <Meter value={0} className="predict-risk-meter" />
        <span className="predict-risk-value predict-na">—</span>
      </li>
    );
  }
  return (
    <li className={`predict-risk predict-risk-${riskLevel(output.value)}`}>
      <span>{task}</span>
      <Meter value={output.value} className="predict-risk-meter" />
      <span className="predict-risk-value">{percent(output.value)}</span>
    </li>
  );
}

export default function ToxicityPanel({ result }: { result: PredictResponse }) {
  const { nuclearReceptor, stressResponse } = result.toxicity;
  const outputs = [
    ...Object.values(nuclearReceptor),
    ...Object.values(stressResponse),
  ];
  const high = outputs.filter(
    (o) => o.value !== null && o.value >= HIGH_RISK,
  ).length;

  return (
    <Panel
      title="Toxicity risk"
      icon={toxicityIcon}
      aside={
        <Pill tone={high ? "bad" : "good"}>
          {high} of {outputs.length} high risk
        </Pill>
      }
    >
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
    </Panel>
  );
}
