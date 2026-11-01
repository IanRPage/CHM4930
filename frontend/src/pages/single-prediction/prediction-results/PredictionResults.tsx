import infoMutedIcon from "../../../assets/icon-info-muted.svg";
import Card from "../../../components/common/Card.tsx";
import type { PredictResponse } from "../../../types/predict.ts";
import ApprovalPanel from "./ApprovalPanel.tsx";
import BioactivityPanel from "./BioactivityPanel.tsx";
import "./PredictionResults.css";
import ToxicityPanel from "./ToxicityPanel.tsx";

export default function PredictionResults({
  result,
}: {
  result: PredictResponse;
}) {
  return (
    <div className="predict-results">
      {result.smiles && (
        <p className="predict-smiles">
          <strong>SMILES:</strong> <code>{result.smiles}</code>
        </p>
      )}
      <Card className="predict-properties">
        <h2>Predicted Properties</h2>
        <BioactivityPanel result={result} />
        <ToxicityPanel result={result} />
        <ApprovalPanel output={result.fdaApproved} />
        <p className="predict-note">
          <img src={infoMutedIcon} alt="" />
          Predictions are model estimates and should be validated
          experimentally.
        </p>
      </Card>
    </div>
  );
}
