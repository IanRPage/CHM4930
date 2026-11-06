import approvalIcon from "../../../assets/icon-approval.svg";
import Meter from "../../../components/common/Meter.tsx";
import Panel from "../../../components/common/Panel.tsx";
import type { Output } from "../../../types/predict.ts";
import "./ApprovalPanel.css";
import { percent, tone } from "./format.ts";

export default function ApprovalPanel({ output }: { output: Output }) {
  return (
    <Panel
      title="Likelihood of FDA approval"
      subtitle="Predicted from ClinTox · higher is better"
      icon={approvalIcon}
      tone={tone(output)}
      className="predict-approval"
      aside={
        output.value === null ? (
          <>
            <span className="predict-approval-reason">{output.reason}</span>
            <span className="predict-approval-value predict-na">—</span>
          </>
        ) : (
          <>
            <Meter value={output.value} className="predict-approval-meter" />
            <span className="predict-approval-value">
              {percent(output.value)}
            </span>
          </>
        )
      }
    />
  );
}
