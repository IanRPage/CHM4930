import { elementColors } from "3dmol";

const NAMES: Record<string, string> = {
  H: "Hydrogen",
  B: "Boron",
  C: "Carbon",
  N: "Nitrogen",
  O: "Oxygen",
  F: "Fluorine",
  Na: "Sodium",
  Mg: "Magnesium",
  Si: "Silicon",
  P: "Phosphorus",
  S: "Sulfur",
  Cl: "Chlorine",
  K: "Potassium",
  Ca: "Calcium",
  Fe: "Iron",
  Cu: "Copper",
  Zn: "Zinc",
  Se: "Selenium",
  Br: "Bromine",
  I: "Iodine",
};

// same table 3Dmol colors atoms with so legend can't drift from view
function color(symbol: string) {
  const rgb = Number(
    elementColors.defaultColors[symbol] ?? elementColors.defaultColor,
  );
  return `#${rgb.toString(16).padStart(6, "0")}`;
}

// elems in molecule taken from its formula
export default function ElementLegend({ formula }: { formula: string }) {
  const symbols = [...new Set(formula.match(/[A-Z][a-z]?/g))];
  return (
    <ul className="predict-info-legend" aria-label="Element colors">
      {symbols.map((symbol) => (
        <li key={symbol}>
          <span
            className="predict-info-swatch"
            style={{ backgroundColor: color(symbol) }}
            aria-hidden="true"
          />
          {NAMES[symbol] ?? symbol}
        </li>
      ))}
    </ul>
  );
}
