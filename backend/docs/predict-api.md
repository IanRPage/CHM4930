# Prediction API

`POST /api/predict` returns MuFu's predictions for one molecule. The backend
calls the inference server (SCRUM-66), validates its response, and reshapes it
for the UI. Types and validators are in [`src/models/`](../src/models/).

## Request

```json
{ "smiles": "CC(=O)Oc1ccccc1C(=O)O" }
```

`smiles` is trimmed. It must then be non-empty, at most 1000 characters, and
contain no whitespace. Chemical validity is checked by the inference server,
since parsing needs RDKit.

## Response

```jsonc
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O", // as returned by the inference server
  "bioactivity": {
    "BACE1": {
      "pic50": { "value": 4.9, "lo": 3.8, "hi": 6.0 },
      "ic50Nm": { "value": 12589, "lo": 1000, "hi": 158489 },
      "pActive": { "value": 0.12 }
    },
    "EGFR": {
      "pic50": { "value": null, "reason": "Not returned by the model" },
      "ic50Nm": { "value": null, "reason": "Not returned by the model" },
      "pActive": { "value": 0.08 }
    }
  },
  "toxicity": {
    "nuclearReceptor": { "NR-AR": { "value": 0.03 } /* 7 tasks */ },
    "stressResponse": { "SR-ARE": { "value": 0.21 } /* 5 tasks */ }
  },
  "fdaApproved": { "value": 0.91 },
  "metadata": {
    "modelVersion": "mufu-…",
    "mcDropoutPasses": 50,
    "ciCoverage": 0.95
  }
}
```

| Field                      | Meaning                                                                                                                 |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `bioactivity.<target>`     | `BACE1`, `EGFR`                                                                                                         |
| `pic50`                    | Point estimate from the eval-mode model. `lo`/`hi` is the conformal MC-Dropout interval at `ciCoverage` (SCRUM-62)      |
| `ic50Nm`                   | $10^{9 - \text{pIC50}}$ nM. Bounds come from the pIC50 bounds, swapped: higher pIC50 means lower IC50                   |
| `pActive`                  | P(pIC50 ≥ 6.0)                                                                                                          |
| `toxicity.nuclearReceptor` | Tox21 `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`                             |
| `toxicity.stressResponse`  | Tox21 `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`                                                                |
| `fdaApproved`              | ClinTox `FDA_APPROVED` probability. `CT_TOX` isn't exposed: on the training data it's almost exactly `1 - FDA_APPROVED` |
| `metadata`                 | Model version, MC-Dropout passes behind the interval, interval coverage                                                 |

Every output is either `{ "value": number, … }` or
`{ "value": null, "reason": string }`. Check `value !== null` before reading
`lo`/`hi`. An unavailable output is never sent as `0` or left out.

Molecular descriptors aren't part of this response (SCRUM-98).

## Errors

The body is always `{ "error": string }`.

| Status | When                                                                       |
| ------ | -------------------------------------------------------------------------- |
| 400    | Missing or malformed body, `smiles` fails the checks above, invalid SMILES |
| 5xx    | Inference server unavailable, timed out, or returned an invalid response   |

The mapping from inference server errors to these statuses is in SCRUM-82.

## Inference server contract

`POST /predict` with `{ "smiles": string }`. The response uses `ai/` naming:

```jsonc
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "model_version": "mufu-…",
  "n_samples": 50, // MC-Dropout passes, from the _mc_dropout.json sidecar
  "coverage": 0.95,
  "bioactivity": {
    "BACE1": {
      "pic50": 4.9,
      "pic50_lo": 3.8,
      "pic50_hi": 6.0,
      "p_active": 0.12
    },
    "EGFR": {
      /* same keys */
    }
  },
  "toxicity": {
    "NR-AR": 0.03, // … all 12 Tox21 tasks …
    "FDA_APPROVED": 0.91,
    "CT_TOX": 0.08
  }
}
```

- Every key above is required. Extra keys are ignored.
- Any output can be `null`. `pic50_lo` and `pic50_hi` are `null` exactly when
  `pic50` is, and otherwise `pic50_lo <= pic50 <= pic50_hi`.
- Probabilities are in [0, 1], `n_samples` is an integer ≥ 2, and `coverage`
  is in (0, 1).
- An invalid SMILES gets a 422 with `{ "error": string }`.

A response that breaks any of these rules is a server fault. The client gets a
generic 5xx, and the details are only logged.
