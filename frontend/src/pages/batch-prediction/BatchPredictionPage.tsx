import { useEffect, useRef, useState } from "react";
import PageHeader from "../../components/common/PageHeader.tsx";
import { MAX_ROWS, parseSmiles } from "../../services/batchParse.ts";
import { batchPredict } from "../../services/batchPredict.ts";
import type { BatchProgress } from "../../services/batchPredict.ts";
import type { BatchResult } from "../../types/batch.ts";
import "./BatchPredictionPage.css";
import { formatDuration } from "./format.ts";
import ProcessingPanel from "./ProcessingPanel.tsx";
import ResultsPanel from "./ResultsPanel.tsx";
import UploadPanel from "./UploadPanel.tsx";

type State =
  | { phase: "upload"; error: string | null }
  | {
      phase: "processing";
      file: File;
      rows: number;
      progress: BatchProgress;
      elapsedMs: number;
    }
  | {
      phase: "done";
      file: File;
      results: BatchResult[];
      elapsedMs: number;
    };

const UPLOAD: State = { phase: "upload", error: null };

export default function BatchPredictionPage() {
  const [state, setState] = useState<State>(UPLOAD);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => controller.current?.abort(), []);

  async function run(file: File) {
    let rows: string[];
    try {
      rows = parseSmiles(await file.text(), file.name);
    } catch {
      setState({ phase: "upload", error: "Couldn't read the file." });
      return;
    }
    if (rows.length === 0 || rows.length > MAX_ROWS) {
      setState({
        phase: "upload",
        error:
          rows.length === 0
            ? "No SMILES found in the file."
            : `The file has ${rows.length} rows. The limit is ${MAX_ROWS}.`,
      });
      return;
    }

    const unique = new Set(rows).size;
    const started = Date.now();
    const abort = new AbortController();
    controller.current = abort;
    setState({
      phase: "processing",
      file,
      rows: rows.length,
      progress: { processed: 0, total: unique, invalid: 0 },
      elapsedMs: 0,
    });
    void batchPredict(
      rows,
      {
        onProgress: (progress) =>
          setState((s) =>
            s.phase === "processing"
              ? { ...s, progress, elapsedMs: Date.now() - started }
              : s,
          ),
        onComplete: (results) =>
          setState({
            phase: "done",
            file,
            results,
            elapsedMs: Date.now() - started,
          }),
        onError: (err) => setState({ phase: "upload", error: err.message }),
      },
      abort.signal,
    );
  }

  function reset() {
    controller.current?.abort();
    setState(UPLOAD);
  }

  if (state.phase === "done") {
    return (
      <div className="batch batch-fixed">
        <div className="batch-head">
          <PageHeader
            title="Batch Results"
            subtitle={`${state.file.name} · ${
              state.results.length
            } molecules · ${formatDuration(state.elapsedMs)}`}
          />
          <button type="button" onClick={reset}>
            New batch
          </button>
        </div>
        <ResultsPanel results={state.results} />
      </div>
    );
  }

  return (
    <div className="batch">
      <PageHeader
        title="Batch Prediction"
        subtitle="Upload a file with multiple molecules for large-scale prediction"
      />
      {state.phase === "upload" && (
        <UploadPanel onRun={run} error={state.error} />
      )}
      {state.phase === "processing" && (
        <ProcessingPanel
          file={state.file}
          rows={state.rows}
          progress={state.progress}
          elapsedMs={state.elapsedMs}
          onCancel={reset}
        />
      )}
    </div>
  );
}
