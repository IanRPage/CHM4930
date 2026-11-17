import { useEffect, useId, useRef, useState } from "react";
import infoIcon from "../../assets/icon-info-muted.svg";
import "./Tip.css";

export default function Tip({ label, hint }: { label: string; hint: string }) {
  const id = useId();
  const button = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    function close(ev: PointerEvent) {
      if (!button.current?.contains(ev.target as Node)) setOpen(false);
    }
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);

  return (
    <button
      ref={button}
      type="button"
      className="info-tip"
      aria-label={`About ${label}`}
      aria-describedby={id}
      data-open={open || undefined}
      onClick={() => setOpen((o) => !o)}
    >
      <img src={infoIcon} alt="" />
      <span role="tooltip" id={id}>
        {hint}
      </span>
    </button>
  );
}
