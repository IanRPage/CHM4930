import "./Meter.css";
import { classNames } from "./classNames.ts";

// Horizontal bar for a 0–1 value. Style with --meter-height, --meter-track,
// --meter-color.
export default function Meter({
  value,
  className,
}: {
  value: number;
  className?: string;
}) {
  const width = `${Math.min(Math.max(value, 0), 1) * 100}%`;
  return (
    <span className={classNames("meter", className)} aria-hidden="true">
      <span className="meter-fill" style={{ width }} />
    </span>
  );
}
