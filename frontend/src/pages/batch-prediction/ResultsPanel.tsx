import { useMemo, useState } from "react";
import Card from "../../components/common/Card.tsx";
import type { BatchResult } from "../../types/batch.ts";
import { TARGETS } from "../../types/predict.ts";
import type { Target } from "../../types/predict.ts";
import { downloadBatchCsv } from "./exportBatchCsv.ts";
import "./ResultsPanel.css";
import ResultsTable from "./ResultsTable.tsx";
import {
  ACTIVE_PIC50,
  ciHalfWidth,
  isActive,
  isHighTox,
  median,
} from "./results.ts";

export default function ResultsPanel({ results }: { results: BatchResult[] }) {
  const [target, setTarget] = useState<Target>("EGFR");
  const [search, setSearch] = useState("");

  const stats = useMemo(() => {
    const widths = results.flatMap((r) => ciHalfWidth(r, target) ?? []);
    const middle = median(widths);
    return {
      predicted: results.filter((r) => r.prediction).length,
      active: results.filter((r) => isActive(r, target)).length,
      high: results.filter(isHighTox).length,
      ci: middle === null ? "—" : `±${middle.toFixed(2)}`,
    };
  }, [results, target]);

  const cards: readonly (readonly [string, string | number, string])[] = [
    ["Molecules predicted", stats.predicted, `of ${results.length} rows`],
    ["Predicted active", stats.active, `pIC50 ≥ ${ACTIVE_PIC50} for ${target}`],
    ["High tox risk", stats.high, "any endpoint ≥ 50%"],
    ["Median 95% CI", stats.ci, `pIC50 half-width for ${target}`],
  ];

  return (
    <div className="batch-results">
      <div className="batch-cards">
        {cards.map(([label, value, note]) => (
          <Card key={label} className="batch-card">
            <span>{label}</span>
            <strong>{value}</strong>
            <small>{note}</small>
          </Card>
        ))}
      </div>

      <Card className="batch-results-body">
        <div className="batch-controls">
          <input
            type="search"
            placeholder="Search SMILES"
            aria-label="Search SMILES"
            value={search}
            onChange={(ev) => setSearch(ev.target.value)}
          />
          <select
            aria-label="Target"
            value={target}
            onChange={(ev) => setTarget(ev.target.value as Target)}
          >
            {TARGETS.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
          <button
            type="button"
            className="batch-download"
            onClick={() => downloadBatchCsv(results, target)}
          >
            Download CSV
          </button>
        </div>
        <ResultsTable
          key={`${search}|${target}`}
          results={results}
          target={target}
          search={search}
        />
      </Card>
    </div>
  );
}
