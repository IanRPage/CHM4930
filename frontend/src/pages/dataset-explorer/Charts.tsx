import type { ReactNode } from "react";
import type { DatasetHistogram } from "../../types/dataset.ts";
import { count, decimal } from "./format.ts";

interface Series {
  label: string;
  color: string;
  histogram: DatasetHistogram;
}

const LEFT = 52,
  TOP = 18,
  HEIGHT = 180;

function Plot({
  title,
  xLabel,
  max,
  width = 550,
  compact = false,
  scroll = false,
  percentage = false,
  children,
}: {
  title: string;
  xLabel: string;
  max: number;
  width?: number;
  compact?: boolean;
  scroll?: boolean;
  percentage?: boolean;
  children: ReactNode;
}) {
  return (
    <div className="explorer-chart-scroll">
      <svg
        className={`explorer-chart${compact ? " explorer-chart-compact" : ""}`}
        style={scroll ? { minWidth: width } : undefined}
        viewBox={`0 0 ${width} 250`}
        role="img"
        aria-label={title}
      >
        <title>{title}</title>
        {Array.from({ length: 5 }, (_, index) => {
          const y = TOP + HEIGHT * (1 - index / 4);
          return (
            <g key={index}>
              <line
                x1={LEFT}
                x2={width - 18}
                y1={y}
                y2={y}
                className="explorer-gridline"
              />
              <text x={LEFT - 9} y={y + 4} textAnchor="end">
                {count((max * index) / 4)}
                {percentage ? "%" : ""}
              </text>
            </g>
          );
        })}
        {children}
        <text
          x={(LEFT + width - 18) / 2}
          y={242}
          textAnchor="middle"
          className="explorer-axis-label"
        >
          {xLabel}
        </text>
      </svg>
    </div>
  );
}

export function Legend({
  items,
}: {
  items: { label: string; color: string }[];
}) {
  return (
    <div className="explorer-legend">
      {items.map(({ label, color }) => (
        <span key={label}>
          <i style={{ background: color }} />
          {label}
        </span>
      ))}
    </div>
  );
}

export function HistogramChart({
  title,
  series,
  xLabel,
  threshold,
  compact = false,
}: {
  title: string;
  series: Series[];
  xLabel: string;
  threshold?: number;
  compact?: boolean;
}) {
  const first = Math.min(...series.map((s) => s.histogram.edges[0]));
  const last = Math.max(...series.map((s) => s.histogram.edges.at(-1)!));
  const range = last - first || 1;
  const highest = Math.max(4, ...series.flatMap((s) => s.histogram.counts));
  const step = 10 ** Math.floor(Math.log10(highest / 4));
  const max = Math.ceil(highest / (4 * step)) * 4 * step;
  const x = (value: number) => LEFT + ((value - first) / range) * 480;
  const description = `${title}. Vertical axis: count. Horizontal axis: ${xLabel}.${
    threshold === undefined ? "" : ` Active threshold: ${threshold}.`
  }`;
  return (
    <>
      {series.length > 1 && <Legend items={series} />}
      <Plot title={description} xLabel={xLabel} max={max} compact={compact}>
        {series.flatMap(({ label, color, histogram }, seriesIndex) =>
          histogram.counts.map((value, index) => {
            const width = ((histogram.binWidth / range) * 480) / series.length;
            const labelText = `${label}: ${
              histogram.edges[index]
            } to less than ${histogram.edges[index + 1]}: ${count(value)}`;
            return (
              <rect
                key={`${label}-${index}`}
                x={x(histogram.edges[index]) + width * seriesIndex + 0.5}
                y={TOP + HEIGHT * (1 - value / max)}
                width={Math.max(0.2, width - 1)}
                height={(HEIGHT * value) / max}
                fill={color}
                className="explorer-bar"
                aria-label={labelText}
              >
                <title>{labelText}</title>
              </rect>
            );
          }),
        )}
        {threshold !== undefined && threshold >= first && threshold <= last && (
          <g>
            <line
              x1={x(threshold)}
              x2={x(threshold)}
              y1={TOP}
              y2={TOP + HEIGHT}
              className="explorer-threshold"
            />
            <text
              x={x(threshold) + 5}
              y={TOP + 10}
              className="explorer-axis-label"
            >
              active ≥ {decimal(threshold)}
            </text>
          </g>
        )}
        {Array.from({ length: 6 }, (_, index) => {
          const value = first + (range * index) / 5;
          return (
            <text
              key={index}
              x={x(value)}
              y={TOP + HEIGHT + 18}
              textAnchor="middle"
            >
              {Number(value.toFixed(1))}
            </text>
          );
        })}
      </Plot>
    </>
  );
}

export function ElementChart({
  elements,
  compact = false,
}: {
  elements: {
    symbol: string;
    percentage: number;
    molecules: number;
    color: string;
  }[];
  compact?: boolean;
}) {
  const width = compact ? 550 : Math.max(550, elements.length * 34 + LEFT + 18);
  const slot = (width - LEFT - 18) / Math.max(1, elements.length);
  return (
    <Plot
      title="Percentage of molecules containing each element"
      xLabel="Element"
      max={100}
      width={width}
      compact={compact}
      scroll={!compact}
      percentage
    >
      {elements.map(({ symbol, percentage, molecules, color }, index) => (
        <g key={symbol}>
          <rect
            x={LEFT + slot * (index + 0.13)}
            y={TOP + HEIGHT * (1 - percentage / 100)}
            width={slot * 0.74}
            height={(HEIGHT * percentage) / 100}
            fill={color}
            className="explorer-bar"
          >
            <title>
              {symbol}: {decimal(percentage, 3)}% · {count(molecules)} molecules
            </title>
          </rect>
          <text
            x={LEFT + slot * (index + 0.5)}
            y={TOP + HEIGHT + 18}
            textAnchor="middle"
          >
            {symbol}
          </text>
        </g>
      ))}
    </Plot>
  );
}
