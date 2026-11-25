import { useState } from "react";
import type { ReactNode } from "react";
import Card from "../../components/common/Card.tsx";
import type {
  DatasetSummaryGroup,
  DatasetSummaryResponse,
  SourceFilter,
} from "../../types/dataset.ts";
import { TARGETS } from "../../types/predict.ts";
import { ElementChart, HistogramChart, Legend } from "./Charts.tsx";
import {
  combineHistograms,
  count,
  decimal,
  elementColor,
  histogramAt,
  percent,
} from "./format.ts";

export function ChartCard({
  title,
  aside,
  children,
}: {
  title: string;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Card className="explorer-chart-card">
      <div className="explorer-card-head">
        <h2>{title}</h2>
        {aside && <div className="explorer-card-aside">{aside}</div>}
      </div>
      {children}
    </Card>
  );
}

function elementsOf(group: DatasetSummaryGroup) {
  return Object.entries(group.elements).map(([symbol, element]) => ({
    symbol,
    percentage: element.percentageOfMolecules,
    molecules: element.moleculesContaining,
    color: elementColor(symbol),
  }));
}

export function Overview({ group }: { group: DatasetSummaryGroup }) {
  const combined = combineHistograms(
    TARGETS.map((target) =>
      histogramAt(group.bioactivity[target].histograms, 0.5),
    ),
  );
  const preview = elementsOf(group)
    .filter((e) => e.symbol !== "H")
    .sort((a, b) => b.percentage - a.percentage)
    .slice(0, 9);
  return (
    <>
      <div className="explorer-overview-grid explorer-stat-cards">
        {[
          ["Molecules", group.counts.total, "blue"],
          [
            "With Bioactivity Labels",
            group.counts.withBioactivityLabels,
            "red",
          ],
          ["With Toxicity Labels", group.counts.withToxicityLabels, "amber"],
        ].map(([label, value, tone]) => (
          <Card key={label} className={`explorer-stat explorer-stat-${tone}`}>
            <strong>{count(value as number)}</strong>
            <span>{label}</span>
          </Card>
        ))}
      </div>
      <div className="explorer-overview-grid">
        {[
          {
            title: "pIC50 Distribution",
            label: "BACE1 + EGFR measurements",
            xLabel: "pIC50",
            histogram: combined,
          },
          {
            title: "Molecular Weight",
            label: "Molecules",
            xLabel: "Molecular Weight (Da)",
            histogram: histogramAt(group.molecularWeight.histograms, 50),
          },
        ].map(({ title, label, xLabel, histogram }) => (
          <ChartCard key={title} title={title}>
            <HistogramChart
              title={
                title === "Molecular Weight"
                  ? "Molecular weight distribution"
                  : "pIC50 distribution"
              }
              compact
              xLabel={xLabel}
              series={[{ label, color: "var(--blue-300)", histogram }]}
            />
          </ChartCard>
        ))}
        <ChartCard title="Element Composition">
          <ElementChart elements={preview} compact />
        </ChartCard>
      </div>
    </>
  );
}

function BinaryRow({
  positive,
  negative,
  color = "var(--green-500)",
  showNegative = true,
  label,
  subtitle,
}: {
  positive: number;
  negative: number;
  color?: string;
  showNegative?: boolean;
  label: string;
  subtitle: ReactNode;
}) {
  const total = positive + negative;
  const rate = total ? positive / total : null;
  return (
    <div className="explorer-binary-row">
      <div>
        <strong>{label}</strong>
        <small>{subtitle}</small>
      </div>
      <div
        className="explorer-fraction"
        role="img"
        aria-label={`${label}: ${count(positive)} positive, ${count(
          negative,
        )} negative${
          rate === null ? ", no observed labels" : `, ${percent(rate)} positive`
        }`}
      >
        {total === 0 ? (
          <span className="explorer-no-labels">No observed labels</span>
        ) : (
          <>
            <div
              className="explorer-fraction-fill"
              style={{ width: `${(rate ?? 0) * 100}%`, background: color }}
            />
            <div className="explorer-fraction-text">
              <span>
                {showNegative ? `${count(positive)} · ` : ""}
                {percent(rate)}
              </span>
              {showNegative && (
                <span>
                  {count(negative)} · {percent(1 - (rate ?? 0))}
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export function PropertyDistribution({ summary }: DatasetSummaryResponse) {
  const [source, setSource] = useState<SourceFilter>("all");
  const [binWidth, setBinWidth] = useState(0.5);
  const group = summary[source];
  const assays = Object.entries(group.tox21).sort(
    (a, b) => (b[1].positiveRate ?? -1) - (a[1].positiveRate ?? -1),
  );
  const highestRate = Math.max(
    0.01,
    ...assays.map(([, s]) => s.positiveRate ?? 0),
  );
  const ct = group.clintox.CT_TOX;
  return (
    <>
      <div className="explorer-controls">
        <label>
          Source{" "}
          <select
            value={source}
            onChange={(e) => setSource(e.target.value as SourceFilter)}
          >
            <option value="all">All sources</option>
            <option value="bace1">BACE1</option>
            <option value="egfr">EGFR</option>
            <option value="tox21">Tox21</option>
            <option value="clintox">ClinTox</option>
          </select>
        </label>
        <span className="explorer-control-static">
          Active threshold <strong>pIC50 ≥ 6.0</strong>
        </span>
        <label>
          Bin width{" "}
          <select
            value={binWidth}
            onChange={(e) => setBinWidth(Number(e.target.value))}
          >
            {group.bioactivity.BACE1.histograms.map((histogram) => (
              <option value={histogram.binWidth} key={histogram.binWidth}>
                {histogram.binWidth}
              </option>
            ))}
          </select>
        </label>
      </div>
      {source !== "all" && (
        <p className="explorer-note explorer-source-note">
          Showing {count(group.counts.total)} compounds belonging to{" "}
          {source === "tox21"
            ? "Tox21"
            : source === "clintox"
              ? "ClinTox"
              : source.toUpperCase()}
          . Sources can overlap.
        </p>
      )}
      <div className="explorer-property-grid">
        <ChartCard
          title="pIC50 Distribution by Target"
          aside={`median ${decimal(
            group.bioactivity.BACE1.statistics.median,
            2,
          )} BACE1 · ${decimal(
            group.bioactivity.EGFR.statistics.median,
            2,
          )} EGFR`}
        >
          <HistogramChart
            title="pIC50 distribution by target"
            xLabel="pIC50"
            threshold={6}
            series={TARGETS.map((target, index) => ({
              label: target,
              color: index === 0 ? "var(--blue-500)" : "var(--explorer-orange)",
              histogram: histogramAt(
                group.bioactivity[target].histograms,
                binWidth,
              ),
            }))}
          />
          {group.counts.withBioactivityLabels === 0 && (
            <p className="explorer-note">
              No observed bioactivity labels in this source.
            </p>
          )}
        </ChartCard>
        <ChartCard
          title="Active vs. Inactive"
          aside={`${count(
            group.counts.withBioactivityLabels,
          )} labeled compounds`}
        >
          <div className="explorer-binary-rows">
            {TARGETS.map((target) => (
              <BinaryRow
                key={target}
                label={target}
                subtitle={`n = ${count(group.bioactivity[target].labeled)}`}
                positive={group.bioactivity[target].active}
                negative={group.bioactivity[target].inactive}
              />
            ))}
          </div>
          <Legend
            items={[
              { label: "Active (pIC50 ≥ 6.0)", color: "var(--green-500)" },
              { label: "Inactive", color: "var(--slate-300)" },
            ]}
          />
        </ChartCard>
        <ChartCard title="Tox21 Positive Rate by Assay" aside="12 assays">
          <Legend
            items={[
              { label: "Nuclear receptor (NR)", color: "var(--violet-600)" },
              { label: "Stress response (SR)", color: "var(--red-500)" },
            ]}
          />
          <div className="explorer-assays">
            {assays.map(([name, assay]) => (
              <div className="explorer-assay" key={name}>
                <span>{name}</span>
                <div className="explorer-assay-track">
                  <i
                    style={{
                      width: `${
                        ((assay.positiveRate ?? 0) / highestRate) * 100
                      }%`,
                      background:
                        assay.family === "NR"
                          ? "var(--violet-600)"
                          : "var(--red-500)",
                    }}
                  />
                </div>
                <strong>{percent(assay.positiveRate)}</strong>
              </div>
            ))}
          </div>
        </ChartCard>
        <ChartCard title="ClinTox Labels" aside="2 tasks">
          <div className="explorer-binary-rows explorer-clintox-rows">
            {(
              [
                [
                  "FDA_APPROVED",
                  "FDA approved",
                  "approved",
                  "not approved",
                  "var(--blue-500)",
                ],
                [
                  "CT_TOX",
                  "Clinical-trial toxicity",
                  "toxic",
                  "not labeled toxic",
                  "var(--red-500)",
                ],
              ] as const
            ).map(([key, label, positiveLabel, negativeLabel, color]) => {
              const { positive, negative } = group.clintox[key];
              return (
                <BinaryRow
                  key={key}
                  label={label}
                  positive={positive}
                  negative={negative}
                  color={color}
                  showNegative={false}
                  subtitle={`${count(positive)} ${positiveLabel} · ${count(
                    negative,
                  )} ${negativeLabel}`}
                />
              );
            })}
          </div>
          <p className="explorer-note">
            {ct.labeled > 0
              ? `Class imbalance: ${count(ct.positive)} of ${count(
                  ct.labeled,
                )} observed ClinTox labels (${percent(
                  ct.positiveRate,
                )}) indicate clinical toxicity.`
              : "No observed ClinTox labels in this source."}
          </p>
        </ChartCard>
      </div>
    </>
  );
}

const HALOGENS = ["F", "Cl", "Br", "I", "At", "Ts"];
const NAMES: Record<string, string> = {
  C: "Carbon",
  H: "Hydrogen",
  N: "Nitrogen",
  O: "Oxygen",
  S: "Sulfur",
};

export function ElementComposition({ group }: { group: DatasetSummaryGroup }) {
  const [sort, setSort] = useState("frequency");
  const elements = elementsOf(group).sort((a, b) =>
    sort === "frequency"
      ? b.percentage - a.percentage
      : a.symbol.localeCompare(b.symbol),
  );
  const averageRows = ["C", "H", "N", "O", "S"].map((symbol) => ({
    symbol,
    name: NAMES[symbol],
    value: group.elements[symbol]?.averageAtomsPerMolecule ?? 0,
    color: elementColor(symbol),
  }));
  averageRows.push({
    symbol: "F/Cl/Br/I",
    name: "Halogens",
    value: HALOGENS.reduce(
      (sum, element) =>
        sum + (group.elements[element]?.averageAtomsPerMolecule ?? 0),
      0,
    ),
    color: "var(--red-500)",
  });
  const maxAtoms = Math.max(1, ...averageRows.map((row) => row.value));
  return (
    <div className="explorer-elements-grid">
      <ChartCard
        title="Molecules Containing Element (%)"
        aside={
          <label className="explorer-sort">
            Sort{" "}
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value)}
              aria-label="Element sort order"
            >
              <option value="frequency">Frequency</option>
              <option value="element">Element</option>
            </select>
          </label>
        }
      >
        <ElementChart elements={elements} />
      </ChartCard>
      <ChartCard title="Average Atoms per Molecule">
        <div className="explorer-average-rows">
          {averageRows.map((row) => (
            <div className="explorer-average-row" key={row.symbol}>
              <span
                className="explorer-element-chip"
                style={{ color: row.color }}
              >
                {row.symbol}
              </span>
              <div>
                <div className="explorer-average-label">
                  <span>{row.name}</span>
                  <strong>{decimal(row.value)}</strong>
                </div>
                <div className="explorer-average-track">
                  <i
                    style={{
                      width: `${(row.value / maxAtoms) * 100}%`,
                      background: row.color,
                    }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
        <dl className="explorer-summary-stats">
          <div>
            <dt>Avg. heavy atom count</dt>
            <dd>{decimal(group.heavyAtomCounts.mean)}</dd>
          </div>
          <div>
            <dt>Total distinct elements</dt>
            <dd>{count(elements.length)}</dd>
          </div>
        </dl>
      </ChartCard>
    </div>
  );
}
