# API

All routes are under `/api`. Types and validators are in
[`src/models/`](../src/models/).

Errors are always `{ "error": string }`. Anything unexpected is a 500
`Internal server error`, and the details are only logged.

**SMILES input** (`/predict`, `/mol-info`): trimmed, then non-empty, at most
1000 characters, no whitespace. A request that fails this gets a 400 with the
validation message, e.g. `smiles must not be empty`.

## `GET /api/health`

| Field          | Type             | Description                                                      |
| -------------- | ---------------- | ---------------------------------------------------------------- |
| `status`       | `"ok"`           | The backend is up                                                |
| `modelVersion` | `string \| null` | The inference server's model version, `null` if it's unreachable |

## `POST /api/predict`

MuFu's predictions for one molecule, from the inference server.

```json
{ "smiles": "CC(=O)Oc1ccccc1C(=O)O" }
```

Each `Output` is `{ "value": number }` or `{ "value": null, "reason": string }`.
An `Estimate` adds `lo`/`hi` when `value` isn't `null`. Unavailable outputs are
never sent as `0` or left out.

| Field                               | Type                                             | Description                                                                                                      |
| ----------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `smiles`                            | `string`                                         | Input SMILES                                                                                                     |
| `nearestTrainTanimoto`              | `number`                                         | Nearest-neighbour 2048-bit ECFP4 Tanimoto similarity in [0, 1] to the training set                               |
| `applicabilityDomain`               | `"in_domain" \| "borderline" \| "out_of_domain"` | ≥ 0.6, [0.4, 0.6), < 0.4. Cut where benchmark error degrades in `ai/data/baseline/results_by_similarity.csv`     |
| `bioactivity.<BACE1\|EGFR>.pic50`   | `Estimate`                                       | Point estimate from the eval-mode model. `lo`/`hi` is the conformal MC-Dropout interval at `metadata.ciCoverage` |
| `bioactivity.<BACE1\|EGFR>.ic50Nm`  | `Estimate`                                       | $10^{9 - \text{pIC50}}$ nM. `lo`/`hi` come from the pIC50 bounds, swapped                                        |
| `bioactivity.<BACE1\|EGFR>.pActive` | `Output`                                         | P(pIC50 ≥ 6.0)                                                                                                   |
| `toxicity.nuclearReceptor.<task>`   | `Output`                                         | Tox21 `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`                      |
| `toxicity.stressResponse.<task>`    | `Output`                                         | Tox21 `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`                                                         |
| `clinicalToxicity`                  | `Output`                                         | ClinTox `CT_TOX` probability of failing clinical trials for toxicity (lower is better). See note below           |
| `metadata.modelVersion`             | `string`                                         | Model version                                                                                                    |
| `metadata.mcDropoutPasses`          | `number`                                         | MC-Dropout passes behind the interval                                                                            |
| `metadata.ciCoverage`               | `number`                                         | Interval coverage, e.g. `0.95`                                                                                   |

`clinicalToxicity` is exposed instead of `FDA_APPROVED`: `FDA_APPROVED` is 95.6%
positive, so the model outputs ~0.90–0.95 for almost any compound. `CT_TOX` is
the actual toxicity label, but it has few positives (80 of 1,428 overall, 9 of
144 in the default test split), so treat it as low-confidence.

`MuFuClient` (`src/services/mufuClient.ts`) maps inference server failures to
fixed messages that never include the server URL, API key, or upstream body.

| Status | `error`                                     | When                                                                         |
| ------ | ------------------------------------------- | ---------------------------------------------------------------------------- |
| 400    | `Invalid SMILES`                            | Inference server returned 422                                                |
| 502    | `Model server returned an invalid response` | Any other upstream status (e.g. 401, 500), non-JSON, schema violation        |
| 503    | `Model server is busy, try again shortly`   | Upstream 429 or 503 (rate limit, concurrency cap, model loading)             |
| 503    | `Model server is unavailable`               | Upstream 502, 504, 530 (Cloudflare: server or tunnel down), connection error |
| 504    | `Model server timed out`                    | No answer within `MUFU_TIMEOUT_MS`                                           |

## `GET /api/mol-info?smiles=`

Identifiers, descriptors and structures for one molecule, computed in the
backend with RDKit.js and OpenChemLib. It doesn't call the inference
server.

```
GET /api/mol-info?smiles=CC(%3DO)Oc1ccccc1C(%3DO)O
```

URL-encode `smiles` (`encodeURIComponent`, `URLSearchParams`). Otherwise `+`
(charges) arrives as a space and fails validation, and `#` (triple bonds) starts
the URL fragment.

| Field                | Type             | Description                                                                                                                   |
| -------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `canonicalSmiles`    | `string`         | RDKit canonical SMILES                                                                                                        |
| `inchiKey`           | `string \| null` | `null` if InChI generation fails                                                                                              |
| `formula`            | `string`         | Molecular formula, e.g. `C9H8O4`                                                                                              |
| `molecularWeight`    | `number`         | Average molecular weight, g/mol                                                                                               |
| `exactMass`          | `number`         | Monoisotopic mass, Da                                                                                                         |
| `logP`               | `number`         | Crippen cLogP                                                                                                                 |
| `tpsa`               | `number`         | Topological polar surface area, Å²                                                                                            |
| `molarRefractivity`  | `number`         | Crippen molar refractivity                                                                                                    |
| `hBondDonors`        | `number`         | Lipinski donors: N–H and O–H count                                                                                            |
| `hBondAcceptors`     | `number`         | Lipinski acceptors: N and O count                                                                                             |
| `rotatableBonds`     | `number`         |                                                                                                                               |
| `heavyAtoms`         | `number`         |                                                                                                                               |
| `aromaticRings`      | `number`         |                                                                                                                               |
| `fractionCsp3`       | `number`         | Fraction of sp³ carbons, [0, 1]                                                                                               |
| `lipinskiViolations` | `number`         | 0–4: `molecularWeight` > 500, `logP` > 5, `hBondDonors` > 5, `hBondAcceptors` > 10                                            |
| `svg`                | `string`         | 2D depiction (has a `viewBox`, so it scales with CSS)                                                                         |
| `molblock`           | `string \| null` | 3D V2000 MolBlock with explicit H, seeded so it's reproducible. `null` above 100 heavy atoms or if conformer generation fails |

Every field is always present. Numbers aren't rounded.

| Status | `error`          | When                 |
| ------ | ---------------- | -------------------- |
| 400    | `Invalid SMILES` | RDKit can't parse it |

The first request on a new instance loads RDKit.js and warms up the conformer
generator, which takes a few seconds. Later requests take milliseconds for
drug-sized molecules and about a second near the 100-heavy-atom cap.

## Inference server contract

Internal: what the backend expects from the `ai/` inference server, and what
`src/models/inference.ts` validates.

`POST /predict` with `{ "smiles": string }`. The response uses `ai/` naming:

```jsonc
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "model_version": "mufu-…",
  "n_samples": 50, // MC-Dropout passes
  "coverage": 0.95,
  "nearest_train_tanimoto": 0.72,
  "applicability_domain": "in_domain",
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
- Probabilities and `nearest_train_tanimoto` are in [0, 1],
  `applicability_domain` is `in_domain` (≥ 0.6), `borderline` ([0.4, 0.6)), or
  `out_of_domain` (< 0.4) matching `nearest_train_tanimoto`, `n_samples` is an
  integer ≥ 2, and `coverage` is in (0, 1).
- An invalid SMILES gets a 422 with `{ "error": string }`.
- Requests carry the shared key in an `X-API-Key` header. A missing or wrong
  key gets a 401.
- A 429 or 503 means busy: over the rate limit, at the concurrency cap, or the
  model is still loading.

A response that breaks any of these rules is a server fault. The client gets a
generic 5xx, and the details are only logged.
