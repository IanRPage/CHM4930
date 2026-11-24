import { memo, useMemo, useState } from "react";
import type { FocusEvent, MouseEvent } from "react";
import Tip from "../../components/common/Tip.tsx";
import type { BatchResult } from "../../types/batch.ts";
import type { Output, Target } from "../../types/predict.ts";
import {
  percent,
  riskLevel,
} from "../single-prediction/prediction-results/format.ts";
import "./ResultsTable.css";
import { TOX_GROUPS, ciHalfWidth, toxOutputs } from "./results.ts";
import { TOX_TOOLTIPS } from "./toxTooltips.ts";

const PAGE_SIZES = [5, 10, 25] as const;
const BIO_COLUMNS = [
  [
    "pIC50",
    "Predicted negative log10 of the IC50 in molar. Higher means more potent.",
  ],
  ["95% CI", "Half-width of the 95% confidence interval on the pIC50."],
  [
    "IC50 (nM)",
    "Concentration for half-maximal inhibition. Lower means more potent.",
  ],
  [
    "P(active)",
    "Predicted probability that the molecule is active against the target.",
  ],
] as const;

const TOX_COLUMNS = TOX_GROUPS.reduce((n, g) => n + g.tasks.length, 0);

interface TipState {
  text: string;
  x: number;
  y: number;
}

const tileLabel = (task: string, o: Output) =>
  `${task}: ${o.value === null ? "unavailable" : percent(o.value)}`;

const ic50 = (v: number) =>
  v.toLocaleString("en-US", { maximumFractionDigits: v < 100 ? 2 : 0 });

function Num({
  output,
  format = (v) => v.toFixed(2),
}: {
  output: Output;
  format?: (v: number) => string;
}) {
  return output.value === null ? (
    <span title={output.reason}>—</span>
  ) : (
    <>{format(output.value)}</>
  );
}

const Row = memo(function Row({
  result,
  target,
  onTip,
}: {
  result: BatchResult;
  target: Target;
  onTip: (tip: TipState | null) => void;
}) {
  const { prediction } = result;
  const smiles = (
    <td className="batch-smiles" title={result.smiles}>
      {result.smiles}
    </td>
  );
  if (!prediction) {
    return (
      <tr className="batch-row-invalid">
        <td>{result.row}</td>
        {smiles}
        <td colSpan={4 + TOX_COLUMNS}>Invalid: {result.error}</td>
      </tr>
    );
  }
  const bio = prediction.bioactivity[target];
  const half = ciHalfWidth(result, target);
  const tiles = toxOutputs(result);

  function show(ev: MouseEvent | FocusEvent, task: string, output: Output) {
    const box = ev.currentTarget.getBoundingClientRect();
    onTip({
      text: `${tileLabel(task, output)}. ${TOX_TOOLTIPS[task]}`,
      x: box.left + box.width / 2,
      y: box.top,
    });
  }

  return (
    <tr>
      <td>{result.row}</td>
      {smiles}
      <td>
        <Num output={bio.pic50} />
      </td>
      <td>{half === null ? "—" : `±${half.toFixed(2)}`}</td>
      <td>
        <Num output={bio.ic50Nm} format={ic50} />
      </td>
      <td>
        <Num output={bio.pActive} />
      </td>
      {tiles.map(([task, output]) => (
        <td key={task} className="batch-tile-cell">
          <span
            role="img"
            tabIndex={0}
            aria-label={tileLabel(task, output)}
            className={`batch-tile batch-risk-${
              output.value === null ? "none" : riskLevel(output.value)
            }`}
            onMouseEnter={(ev) => show(ev, task, output)}
            onFocus={(ev) => show(ev, task, output)}
            onMouseLeave={() => onTip(null)}
            onBlur={() => onTip(null)}
          />
        </td>
      ))}
    </tr>
  );
});

function pageNumbers(current: number, total: number): (number | "…")[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i);
  const pages: (number | "…")[] = [0];
  const lo = Math.max(1, current - 1);
  const hi = Math.min(total - 2, current + 1);
  if (lo > 1) pages.push("…");
  for (let i = lo; i <= hi; i++) pages.push(i);
  if (hi < total - 2) pages.push("…");
  pages.push(total - 1);
  return pages;
}

export default function ResultsTable({
  results,
  target,
  search,
}: {
  results: BatchResult[];
  target: Target;
  search: string;
}) {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(10);
  const [tip, setTip] = useState<TipState | null>(null);

  const rows = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return results.filter((r) => r.smiles.toLowerCase().includes(needle));
  }, [results, search]);

  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  const start = page * pageSize;
  const visible = rows.slice(start, start + pageSize);

  return (
    <div className="batch-table-wrap">
      <div className="batch-table-scroll">
        <table className="batch-table">
          <thead>
            <tr>
              <th>#</th>
              <th>SMILES</th>
              {BIO_COLUMNS.map(([label, hint]) => (
                <th key={label}>
                  <span className="batch-th">
                    {label}
                    <Tip label={label} hint={hint} />
                  </span>
                </th>
              ))}
              {TOX_GROUPS.map((g) => (
                <th
                  key={g.label}
                  colSpan={g.tasks.length}
                  className="batch-group"
                >
                  {g.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <Row key={r.row} result={r} target={target} onTip={setTip} />
            ))}
            {visible.length === 0 && (
              <tr>
                <td colSpan={6 + TOX_COLUMNS} className="batch-empty">
                  No molecules match.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {tip && (
        <div
          role="tooltip"
          className="batch-tox-tip"
          style={{ left: tip.x, top: tip.y }}
        >
          {tip.text}
        </div>
      )}
      <div className="batch-table-foot">
        <ul className="batch-legend">
          <li className="batch-risk-low">Low &lt; 30%</li>
          <li className="batch-risk-moderate">Moderate 30–50%</li>
          <li className="batch-risk-high">High ≥ 50%</li>
        </ul>
        <div className="batch-pager">
          <label className="batch-page-size">
            Rows per page
            <select
              value={pageSize}
              onChange={(ev) => {
                setPageSize(Number(ev.target.value));
                setPage(0);
              }}
            >
              {PAGE_SIZES.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <span className="batch-page-info">
            {rows.length === 0
              ? "0 of 0"
              : `${start + 1}–${start + visible.length} of ${rows.length}`}
          </span>
          <button
            type="button"
            className="batch-page-btn"
            disabled={page === 0}
            onClick={() => setPage(page - 1)}
          >
            ‹
          </button>
          {pageNumbers(page, pages).map((p, i) =>
            p === "…" ? (
              <span key={`e${i}`} className="batch-page-ellipsis">
                …
              </span>
            ) : (
              <button
                key={p}
                type="button"
                className={`batch-page-btn ${
                  p === page ? "batch-page-current" : ""
                }`}
                onClick={() => setPage(p)}
              >
                {p + 1}
              </button>
            ),
          )}
          <button
            type="button"
            className="batch-page-btn"
            disabled={page >= pages - 1}
            onClick={() => setPage(page + 1)}
          >
            ›
          </button>
        </div>
      </div>
    </div>
  );
}
