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

`MuFuClient` (`src/services/mufuClient.ts`) maps inference server failures to
these statuses. The messages are fixed strings: they never include the server
URL, API key, or upstream response body.

| Inference server                                                    | Client                                          |
| ------------------------------------------------------------------- | ----------------------------------------------- |
| 422                                                                 | 400 `Invalid SMILES`                            |
| 429, 503 (rate limit, concurrency cap, model loading)               | 503 `Model server is busy, try again shortly`   |
| 502, 504, 530 (Cloudflare: server or tunnel down), connection error | 503 `Model server is unavailable`               |
| No answer within the client timeout (`MUFU_TIMEOUT_MS`)             | 504 `Model server timed out`                    |
| Any other status (e.g. 401, 500), non-JSON, schema violation        | 502 `Model server returned an invalid response` |

Requests that fail the checks above never reach the inference server and get a
400 with the validation message.

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
  `pic50` is, and otherwise `pic50_lo <= pic50 <= pic50_hi`, with
  `pic50_lo` high enough that $10^{9 - \text{pic50\_lo}}$ is finite.
- Probabilities are in [0, 1], `n_samples` is an integer ≥ 2, and `coverage`
  is in (0, 1).
- An invalid SMILES gets a 422 with `{ "error": string }`.
- Requests carry the shared key in an `X-API-Key` header. A missing or wrong
  key gets a 401.
- A 429 or 503 means busy: over the rate limit, at the concurrency cap, or the
  model is still loading.

A response that breaks any of these rules is a server fault. The client gets a
generic 5xx, and the details are only logged.
