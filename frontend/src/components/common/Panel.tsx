import type { ReactNode } from "react";
import "./Panel.css";
import { classNames } from "./classNames.ts";

export type PanelTone = "plain" | "good" | "neutral";

// Titled section inside a Card. "good"/"neutral" tint the panel and set
// --tint, --line, --track, --fill, --value, --strong for its contents.
export default function Panel({
  title,
  subtitle,
  icon,
  controls,
  aside,
  tone = "plain",
  className,
  children,
}: {
  title: string;
  subtitle?: string;
  icon?: string;
  controls?: ReactNode;
  aside?: ReactNode;
  tone?: PanelTone;
  className?: string;
  children?: ReactNode;
}) {
  return (
    <section
      className={classNames(
        "panel",
        tone !== "plain" && `panel-${tone}`,
        className,
      )}
    >
      <div className="panel-head">
        {icon && (
          <img
            src={icon}
            alt=""
            className={classNames("panel-icon", subtitle && "panel-icon-top")}
          />
        )}
        <div>
          <h3>{title}</h3>
          {subtitle && <p className="panel-subtitle">{subtitle}</p>}
        </div>
        {controls}
        {aside && <div className="panel-aside">{aside}</div>}
      </div>
      {children}
    </section>
  );
}
