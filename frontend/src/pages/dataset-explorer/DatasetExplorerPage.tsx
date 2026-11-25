import { useEffect, useRef, useState } from "react";
import PageHeader from "../../components/common/PageHeader.tsx";
import { getDatasetSummary } from "../../services/dataset.ts";
import type { DatasetSummaryResponse } from "../../types/dataset.ts";
import {
  ElementComposition,
  Overview,
  PropertyDistribution,
} from "./SummaryViews.tsx";
import SampleMolecules from "./SampleMolecules.tsx";
import "./DatasetExplorerPage.css";

const TABS = [
  "Overview",
  "Property Distribution",
  "Element Composition",
  "Sample Molecules",
] as const;
type SummaryState =
  | { status: "loading" }
  | { status: "error" }
  | { status: "ready"; data: DatasetSummaryResponse };

export default function DatasetExplorerPage() {
  const [tab, setTab] = useState(0);
  const [state, setState] = useState<SummaryState>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);
  useEffect(() => {
    const controller = new AbortController();
    void getDatasetSummary(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setState({ status: "ready", data });
      })
      .catch(() => {
        if (!controller.signal.aborted) setState({ status: "error" });
      });
    return () => controller.abort();
  }, [attempt]);
  let content;
  if (tab === 3) content = <SampleMolecules />;
  else if (state.status === "loading")
    content = (
      <p role="status" className="explorer-feedback">
        Loading dataset statistics…
      </p>
    );
  else if (state.status === "error")
    content = (
      <div className="explorer-feedback" role="alert">
        <p>Couldn't load dataset statistics. Please try again.</p>
        <button
          type="button"
          onClick={() => {
            setState({ status: "loading" });
            setAttempt((value) => value + 1);
          }}
        >
          Retry statistics
        </button>
      </div>
    );
  else
    content = (
      <>
        {
          [
            <Overview group={state.data.summary.all} />,
            <PropertyDistribution summary={state.data.summary} />,
            <ElementComposition group={state.data.summary.all} />,
          ][tab]
        }
      </>
    );
  return (
    <div className="explorer">
      <PageHeader
        title="Dataset Explorer"
        subtitle="View dataset statistics and molecular properties"
      />
      <div
        className="explorer-tabs"
        role="tablist"
        aria-label="Dataset explorer views"
      >
        {TABS.map((label, index) => (
          <button
            type="button"
            role="tab"
            id={`explorer-tab-${index}`}
            key={label}
            ref={(el) => {
              tabs.current[index] = el;
            }}
            aria-selected={tab === index}
            aria-controls={`explorer-panel-${index}`}
            tabIndex={tab === index ? 0 : -1}
            onClick={() => setTab(index)}
            onKeyDown={(event) => {
              let next: number;
              if (event.key === "ArrowRight") next = (index + 1) % TABS.length;
              else if (event.key === "ArrowLeft")
                next = (index + TABS.length - 1) % TABS.length;
              else if (event.key === "Home") next = 0;
              else if (event.key === "End") next = TABS.length - 1;
              else return;
              event.preventDefault();
              setTab(next);
              tabs.current[next]?.focus();
            }}
          >
            {label}
          </button>
        ))}
      </div>
      <div
        role="tabpanel"
        id={`explorer-panel-${tab}`}
        aria-labelledby={`explorer-tab-${tab}`}
        tabIndex={0}
      >
        {content}
      </div>
    </div>
  );
}
