import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { predictResponse } from "../../../test/fixtures.ts";
import type { PredictResponse } from "../../../types/predict.ts";
import { NR_TASKS, SR_TASKS } from "../../../types/predict.ts";
import { downloadPredictionCsv, predictionToCsv } from "./exportCsv.ts";

// Decode records independently of the exporter, including quoted line breaks.
function parseCsv(csv: string): string[][] {
  const records: string[][] = [];
  let record: string[] = [];
  let cell = "";
  let quoted = false;
  let closedQuote = false;

  for (let i = 0; i < csv.length; i++) {
    const char = csv[i];
    if (quoted) {
      if (char !== '"') cell += char;
      else if (csv[i + 1] === '"') {
        cell += '"';
        i++;
      } else {
        quoted = false;
        closedQuote = true;
      }
    } else if (char === "," || char === "\r" || char === "\n") {
      record.push(cell);
      cell = "";
      closedQuote = false;
      if (char !== ",") {
        if (char === "\r" && csv[i + 1] === "\n") i++;
        records.push(record);
        record = [];
      }
    } else {
      expect(closedQuote, "unexpected text after a quoted cell").toBe(false);
      if (char === '"') {
        expect(cell, "quote inside an unquoted cell").toBe("");
        quoted = true;
      } else cell += char;
    }
  }

  expect(quoted, "unterminated quoted cell").toBe(false);
  if (record.length || cell || closedQuote) records.push([...record, cell]);
  return records;
}

function csvRow(result: PredictResponse): Record<string, string> {
  const csv = predictionToCsv(result);
  expect(csv).toMatch(/\r\n$/u);
  const records = parseCsv(csv);
  expect(records).toHaveLength(2);
  const [headers, values] = records;
  expect(headers).toHaveLength(31);
  expect(new Set(headers).size).toBe(31);
  expect(values).toHaveLength(31);
  return Object.fromEntries(headers.map((header, i) => [header, values[i]]));
}

describe("predictionToCsv", () => {
  it("exports the complete mapping for both targets, all assays and metadata", () => {
    const result = predictResponse();
    result.bioactivity.EGFR.pic50 = { value: 6.3, lo: 6, hi: 6.6 };
    result.bioactivity.EGFR.ic50Nm = { value: 500, lo: 250, hi: 900 };
    result.toxicity = {
      nuclearReceptor: {
        "NR-AR": { value: 0 },
        "NR-AR-LBD": { value: 0.11 },
        "NR-AhR": { value: 0.22 },
        "NR-Aromatase": { value: 0.33 },
        "NR-ER": { value: 0.44 },
        "NR-ER-LBD": { value: 0.55 },
        "NR-PPAR-gamma": { value: 0.66 },
      },
      stressResponse: {
        "SR-ARE": { value: 0.77 },
        "SR-ATAD5": { value: 0.88 },
        "SR-HSE": { value: 0.99 },
        "SR-MMP": { value: 1 },
        "SR-p53": { value: 0.123456789 },
      },
    };

    expect(csvRow(result)).toEqual({
      SMILES: "CC(=O)Oc1ccccc1C(=O)O",
      "EGFR pIC50": "6.3",
      "EGFR pIC50 CI lower": "6",
      "EGFR pIC50 CI upper": "6.6",
      "EGFR IC50 (nM)": "500",
      "EGFR IC50 (nM) CI lower": "250",
      "EGFR IC50 (nM) CI upper": "900",
      "EGFR P(active) (0-1)": "0.08",
      "BACE1 pIC50": "4.9",
      "BACE1 pIC50 CI lower": "3.8",
      "BACE1 pIC50 CI upper": "6",
      "BACE1 IC50 (nM)": "12589",
      "BACE1 IC50 (nM) CI lower": "1000",
      "BACE1 IC50 (nM) CI upper": "158489",
      "BACE1 P(active) (0-1)": "0.12",
      "Tox21 NR-AR probability (0-1)": "0",
      "Tox21 NR-AR-LBD probability (0-1)": "0.11",
      "Tox21 NR-AhR probability (0-1)": "0.22",
      "Tox21 NR-Aromatase probability (0-1)": "0.33",
      "Tox21 NR-ER probability (0-1)": "0.44",
      "Tox21 NR-ER-LBD probability (0-1)": "0.55",
      "Tox21 NR-PPAR-gamma probability (0-1)": "0.66",
      "Tox21 SR-ARE probability (0-1)": "0.77",
      "Tox21 SR-ATAD5 probability (0-1)": "0.88",
      "Tox21 SR-HSE probability (0-1)": "0.99",
      "Tox21 SR-MMP probability (0-1)": "1",
      "Tox21 SR-p53 probability (0-1)": "0.123456789",
      "FDA approval probability (0-1)": "0.91",
      "Model version": "mufu-test",
      "MC-Dropout passes": "50",
      "CI coverage (0-1)": "0.95",
    });
  });

  it.each([0, 1])(
    "preserves probability boundary %s without percentage scaling",
    (value) => {
      const result = predictResponse();
      result.bioactivity.EGFR.pActive = { value };
      result.bioactivity.BACE1.pActive = { value };
      result.fdaApproved = { value };
      for (const task of NR_TASKS)
        result.toxicity.nuclearReceptor[task] = { value };
      for (const task of SR_TASKS)
        result.toxicity.stressResponse[task] = { value };

      const row = csvRow(result);
      const probabilities = Object.entries(row).filter(([header]) =>
        /P\(active\)|probability/u.test(header),
      );
      expect(probabilities).toHaveLength(15);
      expect(probabilities.map(([, cell]) => cell)).toEqual(
        Array(15).fill(String(value)),
      );
    },
  );

  it("leaves unavailable outputs, SMILES and all metadata blank", () => {
    const result = predictResponse();
    const missing = { value: null, reason: "Not returned by the server" };
    result.smiles = null;
    for (const bio of Object.values(result.bioactivity)) {
      bio.pic50 = missing;
      bio.ic50Nm = missing;
      bio.pActive = missing;
    }
    result.toxicity.nuclearReceptor["NR-AR"] = missing;
    result.toxicity.stressResponse["SR-ARE"] = missing;
    result.fdaApproved = missing;
    result.metadata = {
      modelVersion: null,
      mcDropoutPasses: null,
      ciCoverage: null,
    };

    const row = csvRow(result);
    const availableAssays = Object.entries(row).filter(
      ([header]) =>
        header.startsWith("Tox21 ") &&
        header !== "Tox21 NR-AR probability (0-1)" &&
        header !== "Tox21 SR-ARE probability (0-1)",
    );
    expect(availableAssays).toHaveLength(10);
    expect(availableAssays.map(([, cell]) => cell)).toEqual(
      Array(10).fill("0.1"),
    );
    for (const [header, cell] of Object.entries(row)) {
      if (!availableAssays.some(([assay]) => assay === header))
        expect(cell).toBe("");
    }
  });

  it.each([
    'C"N',
    "C,N",
    'C,"N"',
    "C\rN",
    "C\nN",
    "C\r\nN",
    "\u03bcM \u2014 \u03b3 caf\u00e9 \ud83e\uddea",
    'C,"N"\r\n=SUM(1,2)',
  ])("round-trips text %j without adding records or columns", (text) => {
    const result = predictResponse();
    result.smiles = text;
    result.metadata.modelVersion = text;

    const row = csvRow(result);
    expect(row.SMILES).toBe(text);
    expect(row["Model version"]).toBe(text);
    expect(row["EGFR P(active) (0-1)"]).toBe("0.08");
    expect(row["MC-Dropout passes"]).toBe("50");
  });

  it.each(
    ["", " ", "\t", "\r\n"].flatMap((prefix) =>
      [
        '=HYPERLINK("https://example.com","click")',
        "+1+1",
        "-1+1",
        "@SUM(1,2)",
      ].map((formula) => prefix + formula),
    ),
  )("neutralizes formula-like text %j in both text columns", (text) => {
    const result = predictResponse();
    result.smiles = text;
    result.metadata.modelVersion = text;

    const row = csvRow(result);
    for (const cell of [row.SMILES, row["Model version"]]) {
      expect(cell).toBe("'" + text);
      expect(cell).not.toMatch(/^\s*[=+@-]/u);
    }
  });

  it("preserves negative numeric estimates and scientific notation as numbers", () => {
    const result = predictResponse();
    result.bioactivity.EGFR.pic50 = { value: -1.5, lo: -2.5, hi: -0.5 };
    result.bioactivity.BACE1.pic50 = { value: 17, lo: 16, hi: 18 };
    result.bioactivity.BACE1.ic50Nm = { value: 1e-8, lo: 1e-9, hi: 1e-7 };

    const row = csvRow(result);
    for (const [header, value] of Object.entries({
      "EGFR pIC50": -1.5,
      "EGFR pIC50 CI lower": -2.5,
      "EGFR pIC50 CI upper": -0.5,
      "BACE1 IC50 (nM)": 1e-8,
      "BACE1 IC50 (nM) CI lower": 1e-9,
      "BACE1 IC50 (nM) CI upper": 1e-7,
    })) {
      expect(row[header]).toBe(String(value));
      expect(Number(row[header])).toBe(value);
    }
  });
});

describe("downloadPredictionCsv", () => {
  const url = "blob:prediction-csv";
  const createObjectURL = vi.fn<(blob: Blob) => string>();
  const revokeObjectURL = vi.fn<(url: string) => void>();

  beforeEach(() => {
    createObjectURL.mockReturnValue(url);
    vi.stubGlobal(
      "URL",
      class extends URL {
        static createObjectURL = createObjectURL;
        static revokeObjectURL = revokeObjectURL;
      },
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.resetAllMocks();
  });

  it("downloads a UTF-8 CSV and removes the anchor and object URL", async () => {
    const result = predictResponse();
    result.metadata.modelVersion = "mufu-\u03bc-\u03b3";
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        expect(this).toBeInTheDocument();
        expect(this.hidden).toBe(true);
        expect(this.href).toBe(url);
        expect(this.download).toBe("mufu-prediction.csv");
        expect(createObjectURL).toHaveBeenCalledOnce();
        expect(revokeObjectURL).not.toHaveBeenCalled();
      });

    downloadPredictionCsv(result);

    expect(createObjectURL).toHaveBeenCalledOnce();
    const [blob] = createObjectURL.mock.calls[0];
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe("text/csv;charset=utf-8");
    const bytes = new Uint8Array(await blob.arrayBuffer());
    expect([...bytes.slice(0, 3)]).toEqual([0xef, 0xbb, 0xbf]);
    expect(
      new TextDecoder("utf-8", { ignoreBOM: true }).decode(bytes.slice(3)),
    ).toBe(predictionToCsv(result));
    expect(click).toHaveBeenCalledOnce();
    expect(click.mock.contexts[0]).not.toBeInTheDocument();
    expect(document.querySelector("a[download]")).toBeNull();
    expect(revokeObjectURL).toHaveBeenCalledExactlyOnceWith(url);
  });

  it("removes the anchor and revokes the URL even when clicking throws", () => {
    const failure = new Error("Click failed");
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        expect(this).toBeInTheDocument();
        throw failure;
      });

    expect(() => downloadPredictionCsv(predictResponse())).toThrow(failure);

    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(click).toHaveBeenCalledOnce();
    expect(click.mock.contexts[0]).not.toBeInTheDocument();
    expect(document.querySelector("a[download]")).toBeNull();
    expect(revokeObjectURL).toHaveBeenCalledExactlyOnceWith(url);
  });
});
