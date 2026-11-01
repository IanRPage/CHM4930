import approvalIcon from "../../../assets/icon-approval.svg";
import type { Output } from "../../../types/predict.ts";
import "./ApprovalPanel.css";
import { percent, tone } from "./format.ts";

export default function ApprovalPanel({ output }: { output: Output }) {
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
