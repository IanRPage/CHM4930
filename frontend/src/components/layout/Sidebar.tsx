import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import aboutIcon from "../../assets/icon-about.svg";
import explorerIcon from "../../assets/icon-explorer.svg";
import githubIcon from "../../assets/icon-github.svg";
import homeIcon from "../../assets/icon-home.svg";
import huggingFaceIcon from "../../assets/icon-hugging-face.svg";
import batchIcon from "../../assets/icon-nav-batch.svg";
import predictIcon from "../../assets/icon-nav-predict.svg";
import collapseIcon from "../../assets/icon-sidebar-collapse.svg";
import expandIcon from "../../assets/icon-sidebar-expand.svg";
import logoDark from "../../assets/mufu-logo-dark.svg";
import { classNames } from "../common/classNames.ts";
import {
  DEFAULT_MODEL_VERSION,
  fetchModelVersion,
} from "../../services/health.ts";

const NAV_ITEMS = [
  { to: "/", label: "Home", icon: homeIcon },
  { to: "/predict", label: "Single Prediction", icon: predictIcon },
  { to: "/batch", label: "Batch Prediction", icon: batchIcon },
  { to: "/explorer", label: "Dataset Explorer", icon: explorerIcon },
  { to: "/about", label: "About", icon: aboutIcon },
];

const EXTERNAL_LINKS = [
  {
    href: "https://github.com/IanRPage/CHM4930",
    label: "GitHub",
    icon: githubIcon,
  },
  {
    href: "https://huggingface.co/CHM4930/mufu",
    label: "Hugging Face",
    icon: huggingFaceIcon,
  },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const [modelVersion, setModelVersion] = useState(DEFAULT_MODEL_VERSION);

  useEffect(() => {
    const controller = new AbortController();
    void fetchModelVersion(controller.signal).then((version) => {
      if (version) setModelVersion(version);
    });
    return () => controller.abort();
  }, []);

  return (
    <aside className={classNames("sidebar", collapsed && "sidebar-collapsed")}>
      {collapsed ? (
        <button
          type="button"
          className="sidebar-expand"
          aria-label="Expand sidebar"
          onClick={() => setCollapsed(false)}
        >
          <img
            className="sidebar-logo"
            src={logoDark}
            alt=""
            width={32}
            height={32}
          />
          <img
            className="sidebar-expand-icon"
            src={expandIcon}
            alt=""
            width={20}
            height={20}
          />
        </button>
      ) : (
        <div className="sidebar-head">
          <div className="sidebar-brand">
            <img src={logoDark} alt="" width={32} height={32} />
            <span>MuFu</span>
          </div>
          <button
            type="button"
            className="sidebar-collapse"
            aria-label="Collapse sidebar"
            onClick={() => setCollapsed(true)}
          >
            <img src={collapseIcon} alt="" width={20} height={20} />
          </button>
        </div>
      )}

      <nav aria-label="Primary">
        {NAV_ITEMS.map(({ to, label, icon }) => (
          <NavLink key={to} to={to} end data-tooltip={label}>
            <img src={icon} alt="" width={20} height={20} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {!collapsed && (
        <div className="sidebar-foot">
          <span>Model: {modelVersion}</span>
        </div>
      )}

      <div className="sidebar-links">
        {EXTERNAL_LINKS.map(({ href, label, icon }) => (
          <a
            key={label}
            href={href}
            target="_blank"
            rel="noreferrer"
            data-tooltip={label}
          >
            <img src={icon} alt="" width={20} height={20} />
            <span>{label}</span>
          </a>
        ))}
      </div>
    </aside>
  );
}
