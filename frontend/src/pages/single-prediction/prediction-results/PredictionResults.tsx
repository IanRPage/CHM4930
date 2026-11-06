import infoIcon from "../../../assets/icon-info.svg";
import infoMutedIcon from "../../../assets/icon-info-muted.svg";
import Card from "../../../components/common/Card.tsx";
import type { MolInfo } from "../../../types/molInfo.ts";
import type { PredictResponse } from "../../../types/predict.ts";
import ApprovalPanel from "./ApprovalPanel.tsx";
import BioactivityPanel from "./BioactivityPanel.tsx";
import { downloadPredictionCsv } from "./exportCsv.ts";
import "./PredictionResults.css";
import ToxicityPanel from "./ToxicityPanel.tsx";

const DOMAIN_WARNINGS = {
  borderline: "treat predictions with caution.",
  out_of_domain: "predictions are extrapolations and may be unreliable.",
} as const;

export default function PredictionResults({
  result,
  molInfo,
}: {
  result: PredictResponse;
  molInfo?: MolInfo | null;
}) {
  const { applicabilityDomain: domain, nearestTrainTanimoto: tanimoto } =
    result;
  const warning =
    domain !== "in_domain"
      ? `Tanimoto similarity = ${tanimoto.toFixed(2)}: ${
          DOMAIN_WARNINGS[domain]
        }`
      : null;

  return (
    <div className="predict-results">
      {result.smiles && (
        <p className="predict-smiles">
          <strong>SMILES:</strong> <code>{result.smiles}</code>
        </p>
      )}
      <Card className="predict-properties">
        <div className="predict-properties-head">
          <h2>Predicted Properties</h2>
          <button
            type="button"
            className="predict-download"
            onClick={() => downloadPredictionCsv(result)}
          >
            Download CSV
          </button>
        </div>
        {warning && (
          <p className="predict-domain-warning" role="note">
            <img src={infoIcon} alt="" />
            {warning}
          </p>
        )}
        <BioactivityPanel result={result} />
        <ToxicityPanel result={result} />
        <ApprovalPanel output={result.fdaApproved} />
        <p className="predict-note">
          <img src={infoMutedIcon} alt="" />
          Predictions are model estimates and should be validated
          experimentally.
        </p>
      </Card>
      {molInfo && (
        <Card className="predict-molprops">
          <h2>Molecular Properties</h2>
          <dl>
            <div>
              <dt>Molecular weight</dt>
              <dd>{molInfo.molecularWeight.toFixed(2)} g/mol</dd>
            </div>
            <div>
              <dt>LogP</dt>
              <dd>{molInfo.logP.toFixed(2)}</dd>
            </div>
            <div>
              <dt>H-bond donors</dt>
              <dd>{molInfo.hBondDonors}</dd>
            </div>
            <div>
              <dt>H-bond acceptors</dt>
              <dd>{molInfo.hBondAcceptors}</dd>
            </div>
            <div>
              <dt>Rotatable bonds</dt>
              <dd>{molInfo.rotatableBonds}</dd>
            </div>
          </dl>
        </Card>
      )}
    </div>
  );
}
