"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Verdict = "Real" | "Misinformation" | "Disinformation";

interface Evidence {
  title: string;
  url: string;
  snippet: string;
  source_is_fulltext: boolean;
  entailment: number;
  neutral: number;
  contradiction: number;
}

interface SubResult {
  sub_claim: string;
  verdict: Verdict | "";
  confidence: number;
  probs: Record<string, number>;
  reason: string;
  evidence_count: number;
  top_evidence: Evidence[];
}

interface ClaimResponse {
  claim: string;
  sub_claims: string[];
  verdict: Verdict;
  confidence: number;
  probs: Record<string, number>;
  reason: string;
  sub_results: SubResult[];
  duration_ms: number;
}

const verdictClasses = (v: Verdict | string) => {
  switch (v) {
    case "Real":
      return "bg-emerald-50 text-emerald-900 border-emerald-200";
    case "Misinformation":
      return "bg-amber-50 text-amber-900 border-amber-200";
    case "Disinformation":
      return "bg-rose-50 text-rose-900 border-rose-200";
    default:
      return "bg-slate-50 text-slate-700 border-slate-200";
  }
};

export default function Home() {
  const [claim, setClaim] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClaimResponse | null>(null);

  const submit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/api/v1/claims`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim: claim.trim() }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }
      const data: ClaimResponse = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      <main className="mx-auto max-w-3xl px-6 py-12 sm:px-8">
        <header className="mb-10">
          <h1 className="text-4xl font-bold tracking-tight text-slate-900">
            factforge
          </h1>
          <p className="mt-2 text-slate-600">
            Multimodal fact-checking agent. Submit a claim, get an
            evidence-grounded verdict with citations.
          </p>
        </header>

        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <label
            htmlFor="claim-input"
            className="block text-sm font-medium text-slate-700"
          >
            Your claim
          </label>
          <textarea
            id="claim-input"
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
            disabled={loading}
            placeholder="e.g. Humans only use 10 percent of their brain"
            rows={3}
            className="mt-2 w-full resize-none rounded-lg border border-slate-300 px-4 py-3 text-slate-900 placeholder:text-slate-400 focus:border-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-900/10 disabled:bg-slate-50"
          />
          <div className="mt-3 flex items-center justify-between gap-4">
            <p className="text-xs text-slate-500">
              ~30s per claim (DDG search + NLI on CPU). First request after
              idle may be slower.
            </p>
            <button
              onClick={submit}
              disabled={loading || claim.trim().length < 3}
              className="shrink-0 rounded-lg bg-slate-900 px-6 py-2.5 text-sm font-medium text-white transition-colors hover:bg-slate-700 disabled:bg-slate-300 disabled:cursor-not-allowed"
            >
              {loading ? "Checking…" : "Fact-check"}
            </button>
          </div>
        </section>

        {error && (
          <section className="mt-4 rounded-lg border border-rose-200 bg-rose-50 p-4">
            <p className="font-medium text-rose-900">Something went wrong</p>
            <p className="mt-1 text-sm text-rose-800">{error}</p>
          </section>
        )}

        {loading && (
          <section className="mt-6 rounded-lg border border-slate-200 bg-white p-6">
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 animate-pulse rounded-full bg-slate-900" />
              <p className="text-sm font-medium text-slate-900">
                Agent working…
              </p>
            </div>
            <ul className="mt-3 space-y-1 text-xs text-slate-500">
              <li>1. Decompose claim into sub-claims (Gemini)</li>
              <li>2. Retrieve evidence (DuckDuckGo)</li>
              <li>3. Score evidence against claim (DeBERTa NLI)</li>
              <li>4. Aggregate verdict</li>
            </ul>
          </section>
        )}

        {result && (
          <section className="mt-6 space-y-6">
            {/* Verdict header */}
            <div
              className={`rounded-2xl border-2 p-6 ${verdictClasses(result.verdict)}`}
            >
              <div className="flex items-baseline justify-between">
                <h2 className="text-2xl font-semibold">{result.verdict}</h2>
                <span className="text-sm opacity-75 tabular-nums">
                  {Math.round(result.confidence * 100)}% confidence ·{" "}
                  {(result.duration_ms / 1000).toFixed(1)}s
                </span>
              </div>
              <p className="mt-2 text-sm opacity-75">{result.reason}</p>

              {/* Probability bars */}
              <div className="mt-5 space-y-2">
                {(["Real", "Misinformation", "Disinformation"] as const).map(
                  (cls) => (
                    <div
                      key={cls}
                      className="flex items-center gap-3 text-xs"
                    >
                      <span className="w-28 shrink-0 font-medium">{cls}</span>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/50">
                        <div
                          className="h-full bg-current opacity-70 transition-all"
                          style={{
                            width: `${(result.probs[cls] || 0) * 100}%`,
                          }}
                        />
                      </div>
                      <span className="w-12 text-right tabular-nums">
                        {((result.probs[cls] || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  )
                )}
              </div>
            </div>

            {/* Sub-claims (only if decomposed into >1) */}
            {result.sub_results.length > 1 && (
              <div>
                <h3 className="mb-2 text-sm font-medium uppercase tracking-wider text-slate-500">
                  Sub-claims ({result.sub_results.length})
                </h3>
                <div className="space-y-3">
                  {result.sub_results.map((sr, i) => (
                    <div
                      key={i}
                      className="rounded-lg border border-slate-200 bg-white p-4"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <p className="text-sm text-slate-900">
                          &ldquo;{sr.sub_claim}&rdquo;
                        </p>
                        <span
                          className={`shrink-0 rounded-full border px-2.5 py-0.5 text-xs font-medium ${verdictClasses(sr.verdict)}`}
                        >
                          {sr.verdict || "—"}
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-slate-500">
                        {sr.reason}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Top evidence */}
            <div>
              <h3 className="mb-2 text-sm font-medium uppercase tracking-wider text-slate-500">
                Top evidence
              </h3>
              <div className="space-y-3">
                {result.sub_results
                  .flatMap((sr) => sr.top_evidence)
                  .slice(0, 6)
                  .map((ev, i) => (
                    <a
                      key={i}
                      href={ev.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block rounded-lg border border-slate-200 bg-white p-4 transition-colors hover:border-slate-300 hover:bg-slate-50"
                    >
                      <p className="text-sm font-medium text-slate-900">
                        {ev.title}
                      </p>
                      <p className="mt-0.5 truncate text-xs text-slate-500">
                        {ev.url}
                      </p>
                      <p className="mt-2 text-sm text-slate-700 line-clamp-3">
                        {ev.snippet}
                      </p>
                      <div className="mt-3 flex gap-4 text-xs tabular-nums text-slate-500">
                        <span>
                          entail{" "}
                          <span className="font-medium text-emerald-700">
                            {(ev.entailment * 100).toFixed(0)}%
                          </span>
                        </span>
                        <span>
                          neutral{" "}
                          <span className="font-medium text-slate-700">
                            {(ev.neutral * 100).toFixed(0)}%
                          </span>
                        </span>
                        <span>
                          contra{" "}
                          <span className="font-medium text-rose-700">
                            {(ev.contradiction * 100).toFixed(0)}%
                          </span>
                        </span>
                      </div>
                    </a>
                  ))}
              </div>
            </div>
          </section>
        )}

        <footer className="mt-16 border-t border-slate-200 pt-6 text-center text-xs text-slate-400">
          <p>
            Research demo. Not journalism. Do not use to fact-check named
            individuals.
          </p>
        </footer>
      </main>
    </div>
  );
}
