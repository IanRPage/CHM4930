import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Home" },
  { to: "/predict", label: "Single Prediction" },
  { to: "/batch", label: "Batch Prediction" },
  { to: "/explorer", label: "Dataset Explorer" },
  { to: "/about", label: "About" },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">MuFu</div>
      <nav>
        {NAV_ITEMS.map(({ to, label }) => (
          <NavLink key={to} to={to} end>
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-foot">
        <span>v1.0.0</span>
        <span>Model: MuFu-Fusion</span>
      </div>
    </aside>
  );
}
