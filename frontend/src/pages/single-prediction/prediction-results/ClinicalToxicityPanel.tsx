import toxicityIcon from "../../../assets/icon-toxicity.svg";
import Meter from "../../../components/common/Meter.tsx";
import Panel from "../../../components/common/Panel.tsx";
import type { Output } from "../../../types/predict.ts";
import "./ClinicalToxicityPanel.css";
import { percent } from "./format.ts";

export default function ClinicalToxicityPanel({ output }: { output: Output }) {
  return (
    <Panel
      title="Clinical-trial toxicity risk"
      subtitle="Predicted from ClinTox · lower is better"
      icon={toxicityIcon}
      tone="neutral"
      className="predict-clinical-tox"
      aside={
        output.value === null ? (
          <>
            <span className="predict-clinical-tox-reason">{output.reason}</span>
            <span className="predict-clinical-tox-value predict-na">—</span>
          </>
        ) : (
          <>
            <Meter
              value={output.value}
              className="predict-clinical-tox-meter"
            />
            <span className="predict-clinical-tox-value">
              {percent(output.value)}
            </span>
          </>
        )
      }
    >
      <p className="predict-clinical-tox-caveat">
        Few positive examples behind this (80 of 1,428 compounds), so treat it
        as low-confidence.
      </p>
    </Panel>
  );
}
