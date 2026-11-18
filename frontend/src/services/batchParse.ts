export const MAX_FILE_SIZE = 10 * 1024 * 1024;
export const MAX_ROWS = 5000;
const ACCEPTED_EXTENSIONS = [".csv", ".sdf", ".txt"];

function extension(name: string) {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot).toLowerCase();
}

export function validateFile(file: File): string | null {
  if (!ACCEPTED_EXTENSIONS.includes(extension(file.name))) {
    return "Unsupported file type. Use a CSV, SDF, or TXT file.";
  }
  if (file.size > MAX_FILE_SIZE) return "File is larger than 10 MB.";
  return null;
}

const SMILES_HEADER = /^(canonical_?)?smiles?$/i;

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  const endCell = () => {
    row.push(cell.trim());
    cell = "";
  };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c !== '"') cell += c;
      else if (text[i + 1] === '"') cell += text[i++];
      else quoted = false;
    } else if (c === '"') quoted = true;
    else if (c === ",") endCell();
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      endCell();
      rows.push(row);
      row = [];
    } else cell += c;
  }
  endCell();
  rows.push(row);
  return rows.filter((r) => r.some(Boolean));
}

// SDF: first non-blank line of each $$$$ block. CSV: the "smiles" column if
// the header has one, else the first. TXT: first token of each line.
export function parseSmiles(text: string, fileName: string): string[] {
  const clean = text.replace(/^\uFEFF/, "");
  const ext = extension(fileName);
  if (ext === ".sdf") {
    return clean
      .split(/^\$\$\$\$.*$/m)
      .map((block) => block.match(/\S[^\r\n]*/)?.[0].trim() ?? "")
      .filter(Boolean);
  }
  if (ext !== ".csv") {
    const lines = clean.split(/\r?\n/).filter((l) => l.trim());
    return lines.map((l) => l.trim().split(/\s+/)[0]);
  }
  const rows = parseCsv(clean);
  const col = rows[0]?.findIndex((c) => SMILES_HEADER.test(c)) ?? -1;
  return rows.slice(col < 0 ? 0 : 1).flatMap((r) => r[Math.max(col, 0)] || []);
}
