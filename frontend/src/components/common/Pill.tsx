import type { ReactNode } from "react";
import "./Pill.css";

export default function Pill({
  tone,
  children,
}: {
  tone: "good" | "bad";
  children: ReactNode;
}) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}
