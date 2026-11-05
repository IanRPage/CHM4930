# Molecule information inference API (SCRUM-98)

`POST /molecule` computes identifiers, descriptors, and optional 3D coordinates
using RDKit. It does not call `Predictor.infer`, use neural-network outputs, or
change `POST /predict`. This is the Python service contract for the separate
backend integration PR; it does not expose a new Node route by itself.

## Request and limits

```http
POST /molecule
Content-Type: application/json
X-API-Key: <service key>

{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
```

`smiles` must be a string, nonempty, at most `Settings.max_smiles_length`
(default 1000 characters), and contain no whitespace. The service does not trim
it. RDKit must parse and sanitize it as a nonempty molecule. Names appended to
SMILES and CXSMILES annotations are not accepted. Dummy/unknown atoms (`*`, `*C`)
and explicit isotopes without a positive mass in RDKit's isotope table (such as
`[999C]`) return `422 invalid_smiles`. Known isotopes such as `[13CH4]`, charged
molecules, and valid salts are accepted. Extra JSON fields are ignored,
consistent with `/predict`.

The existing middleware checks `X-API-Key` before the body, enforces
`Settings.max_body_bytes` (default 4096 bytes, including chunked requests), and
shares the per-key rate limit with other protected routes (default 30 requests
per minute). Both `MUFU_API_KEY` and named keys from `MUFU_API_KEYS_FILE` apply.

Molecule parsing, validation, descriptors (including native InChI), and 3D work
run in fresh subprocesses. A single supervisor thread handles pipes and process
lifecycle; it does not execute RDKit calculations. One molecule request is
admitted at a time per server process, with no queue. Descriptor and 3D workers
run sequentially in separate interpreters. The prediction slot and model state
are separate, so a native chemistry stall cannot hold the server interpreter's
GIL or occupy `/predict`'s concurrency slot.

`Settings.request_timeout_s` / `MUFU_REQUEST_TIMEOUT_S` (default 30 seconds) sets
the overall computation budget, starting when the request is admitted:

- The descriptor worker must finish within that budget. If it times out, the
  service terminates and reaps it before returning `503 {"error":"timeout"}`.
- After descriptors succeed, the optional 3D worker gets at most **2 seconds**
  (`STRUCTURE_TIMEOUT_S`), or the remaining overall budget, whichever is shorter.
  Both deadlines include interpreter startup, imports, calculation, and output
  transfer. A 3D timeout or failure returns **200** with the completed descriptors
  and `molblock: null`. If no budget remains after descriptors, 3D is skipped.

Timeouts are enforced by terminating processes, not by cancelling a Future or
relying on RDKit's cooperative native timeout. The supervisor checks deadlines
and cancellation at intervals of at most 50 ms while waiting for worker output.
Process creation, OS scheduling, termination/reaping, and response serialization
can add latency beyond the configured budget; this is not a real-time HTTP
response guarantee. Capacity remains held until worker cleanup completes.

Request-task cancellation and application shutdown signal the supervisor, which
terminates and reaps the active child. Completion and exceptions are observed
even when cancellation repeats. A client disconnect only cancels work if the
ASGI server cancels the request task; otherwise the computation deadline applies.

## Platform lifecycle and resource limits

Windows workers join a non-inheritable Job Object with kill-on-close protection
before receiving input or importing RDKit. Cleanup retains handles to job members,
terminates the job, waits for those processes to exit (including virtualenv
launcher descendants), and closes the handles. The `Popen` child is also reaped.
Closing the job on abrupt server exit terminates its workers. If job setup is
denied, computation fails rather than running chemistry without protection.

On Linux, the child sets `PR_SET_PDEATHSIG` to `SIGKILL` before importing RDKit and
checks its parent PID to cover the setup race. The spawning supervisor thread
stays alive through child cleanup. Normal timeout/cancellation uses `kill` and
`communicate` to reap the child. Workers do not spawn subprocesses. Deployments
must permit `prctl`; restrictive sandbox/seccomp policies can reject startup.
The parent-exit regression should run on the target Linux host, including its
container/PID namespace configuration. Windows tests do not validate that path.
Other operating systems are currently unsupported by the worker bootstrap.

There is one supervisor thread and one active worker interpreter per server
process; Windows virtualenv launchers may add intermediate processes in the same
job. There is no reusable worker pool or waiting request backlog. ETKDG and common
native math thread settings are limited to one. These are concurrency/work limits,
not a hard resident-memory quota or a security sandbox. Multiple server processes
multiply the resource budget, and chemistry still shares host CPU/memory with
prediction.

Fresh interpreters avoid inheriting neural-model threads/state and use the same
Python executable as the server. They import RDKit but do not import Torch or load
MuFu checkpoints. On this Windows validation environment, ordinary ethanol
requests took approximately 0.8 seconds for both phases combined; this is an
illustrative local measurement, not a latency target. Cold imports, antivirus,
and host load can consume the 3D budget and produce a valid null structure. No
additional packages are required.

The route does not wait for model readiness. The existing service lifecycle is
unchanged: startup still loads the model, `/health` still reports model readiness,
and `python -m mufu_server` still exits if model loading fails. This change does
not introduce a standalone deployment mode without checkpoints. The
`molecule_info` Python function itself only needs RDKit and the standard library.
It is synchronous and does not enforce process deadlines on direct calls; use
`MoleculeService` for supervised calculations.

## Success response

HTTP 200 returns one flat JSON object. Every field below is always present.
`smiles` is a nonempty string; all other fields are nullable. Numbers are finite
and are not rounded for display. An unavailable calculation is `null`, never a
placeholder zero, empty string, NaN, or infinity. A failed descriptor does not
discard other successful calculations.

| Field | JSON type (besides null) | Definition / RDKit calculation |
| --- | --- | --- |
| `smiles` | string (never null) | Canonical isomeric `Chem.MolToSmiles` |
| `formula` | string | `rdMolDescriptors.CalcMolFormula` |
| `inchi_key` | string | Standard `Chem.MolToInchiKey`; null if InChI support fails or is absent |
| `name` | string | Currently always null; no name lookup source is configured |
| `molecular_weight` | number | Average molecular weight, g/mol; `Descriptors.MolWt` |
| `exact_mass` | number | Monoisotopic exact mass, Da; `rdMolDescriptors.CalcExactMolWt` |
| `logp` | number | Wildman-Crippen LogP; `Crippen.MolLogP` |
| `tpsa` | number | Topological polar surface area, square angstroms; `rdMolDescriptors.CalcTPSA` defaults |
| `molar_refractivity` | number | Crippen molar refractivity, cubic centimeters/mol; `Crippen.MolMR` |
| `h_bond_donors` | integer | `Lipinski.NumHDonors` |
| `h_bond_acceptors` | integer | `Lipinski.NumHAcceptors` |
| `rotatable_bonds` | integer | `Lipinski.NumRotatableBonds` (RDKit default strict definition) |
| `heavy_atom_count` | integer | `Lipinski.HeavyAtomCount` |
| `aromatic_ring_count` | integer | `Lipinski.NumAromaticRings` |
| `fraction_sp3` | number | Fraction of carbons that are sp3, [0, 1]; `Lipinski.FractionCSP3` |
| `lipinski_violations` | integer | Count from 0 to 4: MW > 500, LogP > 5, donors > 5, acceptors > 10 |
| `qed` | number | Quantitative estimate of drug-likeness, [0, 1]; `QED.qed` default weights |
| `molblock` | string | Optional 3D MDL MolBlock with explicit hydrogens; see below |

Descriptors use the sanitized input structure. There is no salt stripping,
neutralization, tautomer selection, or protonation adjustment. Isotopes, charge,
specified stereochemistry, and disconnected fragments are retained. RDKit's
formula defaults do not label individual isotopes; exact mass accounts for them.
If any of MW, LogP, donors, or acceptors is unavailable, `lipinski_violations` is
also null. Lipinski thresholds are strict (`>`); values at the threshold do not
count as violations.

## Optional 3D structure

`molblock` is an RDKit-generated 3D conformer for visualization, not an
experimental structure or a claimed lowest-energy conformation. It contains
explicit hydrogens and can be read with `Chem.MolFromMolBlock` or a viewer's MOL
parser. It is the structure block of an SDF record; append a newline and `$$$$`
record separator if an SDF reader requires a complete record.

One `ETKDGv3` embedding call uses a fixed random seed of 0, one thread, at most
20 embedding iterations, and RDKit's native 2-second timeout. There are no retry
loops or additional force-field minimizations. Only connected structures with
at most 256 atoms **after adding hydrogens** are embedded. Larger structures and
disconnected salts/mixtures still receive descriptors, with `molblock: null`;
relative positions of disconnected fragments are not inferred.

Failed embedding, exceptions during coordinate generation or serialization,
nonfinite coordinates, or a build without the native timeout all produce
`molblock: null` and HTTP 200 with available descriptors. There is no 2D fallback
in this field. RDKit's native timeout is cooperative; the independent parent
deadline described above terminates and reaps an over-budget 3D process even
when native work does not cooperate. Descriptor results are already held in the
parent and survive that termination. Atom and iteration caps also limit the
work. Seeded coordinates are reproducible within a given RDKit build; identical
coordinates across RDKit versions/platforms are not promised.

## Errors

All error bodies use the existing `{ "error": string }` envelope.

| HTTP status | `error` | Meaning |
| --- | --- | --- |
| 401 | `unauthorized` | Missing or incorrect API key |
| 413 | `body_too_large` | Request body exceeds the byte limit |
| 422 | `invalid_request` | Invalid JSON, missing SMILES, wrong type, or invalid body shape |
| 422 | `invalid_smiles` | Empty, whitespace-containing, overlong, or chemically invalid SMILES |
| 429 | `rate_limited` | Shared per-key rate limit; includes `Retry-After` in seconds |
| 503 | `busy` | Another molecule computation is still running |
| 503 | `timeout` | Descriptor worker exceeded the overall computation budget; cleanup completes before the response |
| 500 | `internal_error` | Unexpected computation failure outside optional-field handling |

## Backend handoff and review

The proposed inference route is separate from `/predict`, with snake_case names
matching existing `ai/` conventions. Ian should confirm these names and null
semantics, the visualization limits, and the public Node route before backend
integration. A separate `GET /api/molecule?smiles=...` is the Jira proposal; the
backend can proxy it to this authenticated `POST /molecule`.

The backend PR must add a `MuFuClient` operation, validate the response types and
ranges above, map inference 422 to public 400, and handle upstream failures using
the existing error conventions. It must also update `backend/docs/predict-api.md`
and `backend/README.md` and add the Jira-required Supertest coverage. Those files,
the public route, and SCRUM-95 frontend integration are outside this AI change.

## Validation

From `ai/` in the project environment:

```sh
python -m pytest tests/mufu_server
python -m pytest tests/mufu tests/featurize
```

`test_molecule.py` compares unrounded values directly with RDKit for known,
charged, isotopic, aromatic, and high-MW molecules, checks a 3D MolBlock round
trip and stereochemistry, and exercises unavailable fields and embedding limits.
The API tests cover authentication, malformed bodies, request limits, rate
limiting, timeouts, concurrency, error recovery, and independence from prediction.
`test_molecule_service.py` also exercises real native stalls in child processes,
descriptor/3D deadlines, repeated cancellation, shutdown, worker crashes and bad
output, process reaping, capacity recovery, and abrupt parent exit. Run it on
both Windows and Linux to validate the platform-specific lifecycle protections.

References: [SCRUM-98](https://chm4930.atlassian.net/browse/SCRUM-98) and the
[RDKit embedding API](https://rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html).
Process lifecycle references:
[Windows Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects),
[Windows job termination](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-terminatejobobject),
and [Linux parent-death signals](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html).
