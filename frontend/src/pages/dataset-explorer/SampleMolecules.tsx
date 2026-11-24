import { useEffect, useRef, useState } from "react";
import Card from "../../components/common/Card.tsx";
import Pill from "../../components/common/Pill.tsx";
import { getDatasetMolecules } from "../../services/dataset.ts";
import { getMolInfo } from "../../services/molInfo.ts";
import type {
  DatasetMolecule,
  DatasetMoleculesResponse,
  DatasetQuery,
} from "../../types/dataset.ts";
import { TARGETS } from "../../types/predict.ts";
import { count, decimal } from "./format.ts";

function MoleculeStructure({ smiles }: { smiles: string }) {
  const frame = useRef<HTMLDivElement>(null);
  const [image, setImage] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    const load = () => {
      void getMolInfo(smiles, { signal: controller.signal })
        .then((info) => {
          if (!controller.signal.aborted)
            setImage(`data:image/svg+xml,${encodeURIComponent(info.svg)}`);
        })
        .catch(() => {
          if (!controller.signal.aborted) setFailed(true);
        });
    };
    if (!frame.current || typeof IntersectionObserver === "undefined") {
      load();
      return () => controller.abort();
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          load();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(frame.current);
    return () => {
      observer.disconnect();
      controller.abort();
    };
  }, [smiles]);
  return (
    <div ref={frame} className="explorer-molecule-structure">
      {image ? (
        <img src={image} alt={`Molecular structure of ${smiles}`} />
      ) : (
        <span>{failed ? "Structure unavailable" : "Loading structure…"}</span>
      )}
    </div>
  );
}

function MoleculeCard({ molecule }: { molecule: DatasetMolecule }) {
  return (
    <Card
      as="article"
      className="explorer-molecule-card"
      aria-label={molecule.id}
    >
      <MoleculeStructure smiles={molecule.smiles} />
      <div className="explorer-molecule-body">
        <div className="explorer-molecule-heading">
          <h2 title={molecule.id}>{molecule.id}</h2>
          <Pill
            tone={
              molecule.risk === "positive"
                ? "bad"
                : molecule.risk === "negative"
                  ? "good"
                  : "neutral"
            }
          >
            {molecule.risk === "positive"
              ? "Positive for toxicity"
              : molecule.risk === "negative"
                ? "Negative for toxicity"
                : "Unlabeled"}
          </Pill>
        </div>
        <p className="explorer-smiles" title={molecule.smiles}>
          {molecule.smiles}
        </p>
        <dl className="explorer-molecule-values">
          {TARGETS.map((target) => (
            <div key={target}>
              <dt>{target} pIC50</dt>
              <dd>{decimal(molecule.bioactivity[target].pic50, 2)}</dd>
            </div>
          ))}
        </dl>
        <p className="explorer-molecule-source">
          {molecule.sources
            .map((s) =>
              s === "tox21"
                ? "Tox21"
                : s === "clintox"
                  ? "ClinTox"
                  : s.toUpperCase(),
            )
            .join(" · ")}
        </p>
      </div>
    </Card>
  );
}

type ResultsState = {
  items: DatasetMolecule[];
  response: DatasetMoleculesResponse | null;
  status: "loading" | "ready" | "error";
};
const INITIAL: ResultsState = { items: [], response: null, status: "loading" };

function MoleculeResults({ filters }: { filters: Omit<DatasetQuery, "page"> }) {
  const [state, setState] = useState<ResultsState>(INITIAL);
  const [request, setRequest] = useState({ page: 1, attempt: 0 });
  const inFlight = useRef(true);
  useEffect(() => {
    const controller = new AbortController();
    inFlight.current = true;
    // Debounce filter edits; abort old requests so stale results cannot overwrite a new query.
    const timer = window.setTimeout(() => {
      void getDatasetMolecules(
        { ...filters, page: request.page },
        controller.signal,
      )
        .then((response) => {
          if (controller.signal.aborted) return;
          inFlight.current = false;
          setState((previous) => ({
            status: "ready",
            response,
            items:
              request.page === 1
                ? response.items
                : [...previous.items, ...response.items],
          }));
        })
        .catch(() => {
          if (controller.signal.aborted) return;
          inFlight.current = false;
          setState((previous) => ({ ...previous, status: "error" }));
        });
    }, 250);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [filters, request]);
  function load(page: number) {
    if (inFlight.current) return;
    inFlight.current = true;
    setState((previous) => ({ ...previous, status: "loading" }));
    setRequest((previous) => ({ page, attempt: previous.attempt + 1 }));
  }
  return (
    <>
      <div className="explorer-results-status" role="status" aria-live="polite">
        {state.response && (
          <>
            Showing {count(state.items.length)} of {count(state.response.total)}{" "}
            matching molecules
          </>
        )}
        {state.status === "loading" && (
          <span>
            {state.items.length
              ? "Loading more molecules…"
              : "Loading molecules…"}
          </span>
        )}
      </div>
      <div
        className="explorer-molecule-grid"
        aria-busy={state.status === "loading"}
      >
        {state.items.map((molecule) => (
          <MoleculeCard key={molecule.id} molecule={molecule} />
        ))}
      </div>
      {state.status === "ready" && state.items.length === 0 && (
        <p className="explorer-feedback">
          No molecules match your search and filters.
        </p>
      )}
      {state.status === "error" && (
        <div className="explorer-feedback" role="alert">
          <p>Couldn't load molecules. Please try again.</p>
          <button type="button" onClick={() => load(request.page)}>
            Retry molecules
          </button>
        </div>
      )}
      {state.response?.hasMore && state.status !== "error" && (
        <div className="explorer-load-more">
          <button
            type="button"
            disabled={state.status === "loading"}
            onClick={() => load(state.response!.page + 1)}
          >
            {state.status === "loading" ? "Loading…" : "Load more"}
          </button>
        </div>
      )}
    </>
  );
}

export default function SampleMolecules() {
  const [filters, setFilters] = useState<Omit<DatasetQuery, "page">>({
    q: "",
    activity: "all",
    risk: "all",
  });
  function update(name: keyof typeof filters, value: string) {
    setFilters((previous) => ({ ...previous, [name]: value }));
  }
  return (
    <>
      <div className="explorer-controls explorer-search-controls">
        <input
          type="search"
          aria-label="Search by ID or SMILES"
          placeholder="Search by ID or SMILES…"
          value={filters.q}
          onChange={(e) => update("q", e.target.value)}
        />
        <label>
          Activity{" "}
          <select
            value={filters.activity}
            onChange={(e) => update("activity", e.target.value)}
          >
            <option value="all">All</option>
            <option value="active">Active (≥ 6.0)</option>
            <option value="inactive">Inactive (&lt; 6.0)</option>
            <option value="unknown">Unlabeled</option>
          </select>
        </label>
        <label>
          Toxicity labels{" "}
          <select
            value={filters.risk}
            onChange={(e) => update("risk", e.target.value)}
          >
            <option value="all">All</option>
            <option value="positive">Positive for toxicity</option>
            <option value="negative">Negative for toxicity</option>
            <option value="unknown">Unlabeled</option>
          </select>
        </label>
      </div>
      <MoleculeResults key={JSON.stringify(filters)} filters={filters} />
    </>
  );
}
