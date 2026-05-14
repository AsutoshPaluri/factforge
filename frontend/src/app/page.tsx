"use client";

import { useEffect, useRef, useState } from "react";

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
  summary: string;
  sub_results: SubResult[];
  duration_ms: number;
  was_multimodal: boolean;
}

const EXAMPLE_CLAIMS = [
  "Humans only use 10% of their brain",
  "5G networks cause COVID-19",
  "The earth is flat when viewed from space",
  "Drinking warm water in the morning aids digestion",
];

const AGENT_STEPS = [
  {
    label: "Decompose",
    desc: "Break into atomic sub-claims (Gemini, vision-capable)",
  },
  { label: "Retrieve", desc: "Search the open web via DuckDuckGo" },
  { label: "Verify", desc: "Score each snippet with DeBERTa-MNLI-FEVER" },
  { label: "Synthesize", desc: "Aggregate to final verdict + confidence" },
  {
    label: "Summarize",
    desc: "Write plain-English explanation citing sources (Gemini)",
  },
];

const STEP_DURATIONS_S = [3, 8, 15, 3, 4];
const ACCEPTED_MIME = "image/jpeg,image/png,image/webp,image/gif";

const verdictStyles = (v: Verdict | string) => {
  switch (v) {
    case "Real":
      return {
        card: "bg-emerald-50 border-emerald-200 text-emerald-900",
        bar: "bg-emerald-500",
        pill: "bg-emerald-100 text-emerald-800 border-emerald-200",
      };
    case "Misinformation":
      return {
        card: "bg-amber-50 border-amber-200 text-amber-900",
        bar: "bg-amber-500",
        pill: "bg-amber-100 text-amber-800 border-amber-200",
      };
    case "Disinformation":
      return {
        card: "bg-rose-50 border-rose-200 text-rose-900",
        bar: "bg-rose-500",
        pill: "bg-rose-100 text-rose-800 border-rose-200",
      };
    default:
      return {
        card: "bg-slate-50 border-slate-200 text-slate-700",
        bar: "bg-slate-500",
        pill: "bg-slate-100 text-slate-700 border-slate-200",
      };
  }
};

const verdictIcon = (v: Verdict | string) => {
  switch (v) {
    case "Real":
      return "✓";
    case "Misinformation":
      return "?";
    case "Disinformation":
      return "✕";
    default:
      return "—";
  }
};

function domain(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

// Favicon via Google's free service. No auth, no rate limit issues for our scale.
function faviconUrl(url: string, size: 16 | 32 | 64 = 32): string {
  const d = domain(url);
  return `https://www.google.com/s2/favicons?domain=${encodeURIComponent(d)}&sz=${size}`;
}

// Minimal inline-markdown renderer: supports **bold** and *italic*.
// Returns a list of React nodes so we can drop into <p>.
function renderInlineMarkdown(text: string): React.ReactNode[] {
  // Split into alternating plain / **bold** / *italic* segments
  const parts = text.split(/(\*\*[^*\n]+\*\*|\*[^*\n]+\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-stone-900">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (
      part.startsWith("*") &&
      part.endsWith("*") &&
      !part.startsWith("**")
    ) {
      return (
        <em key={i} className="italic">
          {part.slice(1, -1)}
        </em>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

// Read a File into a base64 string (no data URL prefix)
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.split(",")[1] || "");
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

export default function Home() {
  const [claim, setClaim] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClaimResponse | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Animate through agent steps while waiting on the API
  useEffect(() => {
    if (!loading) {
      setCurrentStep(0);
      return;
    }
    const timers: ReturnType<typeof setTimeout>[] = [];
    let cumulative = 0;
    for (let i = 1; i < STEP_DURATIONS_S.length; i++) {
      cumulative += STEP_DURATIONS_S[i - 1];
      timers.push(
        setTimeout(() => setCurrentStep(i), cumulative * 1000),
      );
    }
    return () => timers.forEach(clearTimeout);
  }, [loading]);

  // Revoke object URL on unmount / change to avoid memory leak
  useEffect(() => {
    return () => {
      if (imagePreview) URL.revokeObjectURL(imagePreview);
    };
  }, [imagePreview]);

  const handleFileSelect = (file: File | null) => {
    if (!file) {
      if (imagePreview) URL.revokeObjectURL(imagePreview);
      setImageFile(null);
      setImagePreview(null);
      return;
    }
    if (!file.type.startsWith("image/")) {
      setError("Please choose an image file (JPEG / PNG / WebP / GIF).");
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setError("Image must be under 5 MB.");
      return;
    }
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setError(null);
  };

  const canSubmit =
    !loading && (claim.trim().length >= 3 || imageFile !== null);

  const submit = async (claimToCheck?: string) => {
    const c = (claimToCheck ?? claim).trim();
    if (c.length < 3 && !imageFile) return;
    if (claimToCheck !== undefined) setClaim(claimToCheck);
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let image_b64: string | null = null;
      let image_mime: string | undefined;
      if (imageFile) {
        image_b64 = await fileToBase64(imageFile);
        image_mime = imageFile.type;
      }

      const res = await fetch(`${API_URL}/api/v1/claims`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          claim: c,
          image_b64,
          image_mime,
        }),
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
    <div className="min-h-screen bg-gradient-to-b from-stone-50 via-white to-stone-50">
      {/* Header */}
      <header className="border-b border-stone-200/60 bg-white/70 backdrop-blur-sm">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2.5">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2.5}
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-5 w-5 text-indigo-600"
              aria-hidden
            >
              <circle cx="10.5" cy="10.5" r="6.5" />
              <path d="m21 21-5.5-5.5" />
              <path d="m8 10.5 2 2 4-4" />
            </svg>
            <span className="text-xl font-semibold tracking-tight text-stone-900">
              factforge
            </span>
            <span className="font-mono text-xs text-stone-400">v0.1</span>
          </div>
          <nav className="flex items-center gap-5 text-sm">
            <a
              href={`${API_URL}/docs`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-stone-500 transition hover:text-stone-900"
            >
              API
            </a>
            <a
              href="https://github.com/asutoshpaluri/factforge"
              target="_blank"
              rel="noopener noreferrer"
              className="text-stone-500 transition hover:text-stone-900"
            >
              GitHub
            </a>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-12">
        {/* Hero */}
        <section className="mb-10">
          <h1 className="text-4xl font-semibold tracking-tight text-stone-900 sm:text-5xl">
            Fact-check{" "}
            <span className="bg-gradient-to-r from-indigo-600 to-rose-500 bg-clip-text text-transparent">
              anything
            </span>
            .
          </h1>
          <p className="mt-4 max-w-2xl text-lg leading-relaxed text-stone-600">
            Multimodal agent — submit a claim as text, an image (screenshot,
            chart, infographic), or both. Retrieves real web evidence, scores
            it with a fine-tuned NLI model, and returns an auditable verdict
            with a plain-English explanation.
          </p>
        </section>

        {/* Input card */}
        <section className="rounded-2xl border border-stone-200 bg-white p-6 shadow-sm">
          <label
            htmlFor="claim-input"
            className="block text-sm font-medium text-stone-700"
          >
            Your claim {imageFile && <span className="text-stone-400">(optional — image will be analyzed)</span>}
          </label>
          <textarea
            id="claim-input"
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
            disabled={loading}
            placeholder={
              imageFile
                ? "Optional context about the image (e.g. 'tweet by X saying Y')"
                : "e.g. Humans only use 10 percent of their brain"
            }
            rows={3}
            className="mt-2 w-full resize-none rounded-lg border border-stone-300 bg-white px-4 py-3 text-stone-900 placeholder:text-stone-400 focus:border-indigo-500 focus:outline-none focus:ring-4 focus:ring-indigo-500/15 disabled:bg-stone-50"
          />

          {/* Image input row */}
          <div className="mt-3">
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_MIME}
              onChange={(e) => handleFileSelect(e.target.files?.[0] ?? null)}
              className="hidden"
            />

            {!imageFile ? (
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
                className="inline-flex items-center gap-2 rounded-lg border border-stone-200 bg-stone-50 px-3 py-2 text-xs font-medium text-stone-700 transition hover:border-stone-300 hover:bg-stone-100 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.8}
                  stroke="currentColor"
                  className="h-4 w-4"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="m2.25 15.75 5.159-5.159a2.25 2.25 0 0 1 3.182 0l5.159 5.159m-1.5-1.5 1.409-1.409a2.25 2.25 0 0 1 3.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 0 0 1.5-1.5V6a1.5 1.5 0 0 0-1.5-1.5H3.75A1.5 1.5 0 0 0 2.25 6v12a1.5 1.5 0 0 0 1.5 1.5Zm10.5-11.25h.008v.008h-.008V8.25Zm.375 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Z"
                  />
                </svg>
                Add image (screenshot, chart, infographic)
              </button>
            ) : (
              <div className="flex items-start gap-3 rounded-lg border border-stone-200 bg-stone-50 p-3">
                {imagePreview && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={imagePreview}
                    alt="claim preview"
                    className="h-20 w-20 shrink-0 rounded-md border border-stone-200 object-cover"
                  />
                )}
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-medium text-stone-900">
                    {imageFile.name}
                  </p>
                  <p className="mt-0.5 font-mono text-[11px] text-stone-500">
                    {imageFile.type} · {(imageFile.size / 1024).toFixed(0)} KB
                  </p>
                  <button
                    type="button"
                    onClick={() => handleFileSelect(null)}
                    disabled={loading}
                    className="mt-2 text-xs font-medium text-rose-600 hover:text-rose-700 disabled:opacity-50"
                  >
                    Remove
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Example chips */}
          <div className="mt-4 flex flex-wrap gap-2">
            <span className="text-xs font-medium text-stone-500">Try:</span>
            {EXAMPLE_CLAIMS.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => !loading && submit(ex)}
                disabled={loading}
                className="rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-xs text-stone-700 transition hover:border-stone-300 hover:bg-stone-100 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {ex}
              </button>
            ))}
          </div>

          <div className="mt-5 flex items-center justify-between gap-4">
            <p className="text-xs text-stone-500">
              ~30s per claim · DDG retrieval + DeBERTa NLI on free-tier CPU
            </p>
            <button
              onClick={() => submit()}
              disabled={!canSubmit}
              className="shrink-0 rounded-lg bg-stone-900 px-6 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:bg-stone-300"
            >
              {loading ? "Checking…" : "Fact-check"}
            </button>
          </div>
        </section>

        {/* Error */}
        {error && (
          <section className="mt-4 animate-fade-up rounded-xl border border-rose-200 bg-rose-50 p-4">
            <p className="text-sm font-medium text-rose-900">
              Something went wrong
            </p>
            <p className="mt-1 break-words text-xs text-rose-800">{error}</p>
          </section>
        )}

        {/* Loading — animated agent trace */}
        {loading && (
          <section className="mt-6 animate-fade-up rounded-2xl border border-stone-200 bg-white p-6 shadow-sm">
            <div className="mb-4 flex items-center gap-2">
              <span className="block h-2 w-2 animate-pulse-dot rounded-full bg-indigo-500" />
              <p className="text-sm font-medium text-stone-900">
                Agent working
                {imageFile && (
                  <span className="ml-2 rounded-full bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-700">
                    multimodal
                  </span>
                )}
              </p>
            </div>
            <ol className="space-y-2">
              {AGENT_STEPS.map((step, i) => {
                const done = i < currentStep;
                const active = i === currentStep;
                return (
                  <li
                    key={step.label}
                    className={`flex items-start gap-3 rounded-lg p-2 transition ${
                      active ? "bg-indigo-50" : ""
                    }`}
                  >
                    <span
                      className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full font-mono text-xs ${
                        done
                          ? "bg-indigo-600 text-white"
                          : active
                            ? "bg-indigo-100 text-indigo-700"
                            : "bg-stone-100 text-stone-400"
                      }`}
                    >
                      {done ? "✓" : i + 1}
                    </span>
                    <div className="min-w-0">
                      <p
                        className={`text-sm font-medium ${
                          active
                            ? "text-indigo-900"
                            : done
                              ? "text-stone-700"
                              : "text-stone-400"
                        }`}
                      >
                        {step.label}
                      </p>
                      <p className="text-xs text-stone-500">{step.desc}</p>
                    </div>
                  </li>
                );
              })}
            </ol>
          </section>
        )}

        {/* Result */}
        {result && (
          <section className="mt-6 animate-fade-up space-y-6">
            {/* Verdict hero card */}
            <div
              className={`rounded-2xl border-2 p-6 shadow-sm ${verdictStyles(result.verdict).card}`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-3">
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-white/60 font-mono text-xl">
                      {verdictIcon(result.verdict)}
                    </span>
                    <h2 className="text-3xl font-semibold tracking-tight">
                      {result.verdict}
                    </h2>
                    {result.was_multimodal && (
                      <span className="rounded-full bg-white/60 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider">
                        multimodal
                      </span>
                    )}
                  </div>
                  <p className="mt-2 text-sm opacity-80">{result.reason}</p>
                </div>
                <div className="shrink-0 text-right font-mono text-xs opacity-75">
                  <div className="tabular-nums">
                    {Math.round(result.confidence * 100)}%
                  </div>
                  <div className="mt-0.5 text-[10px] uppercase tracking-wider">
                    confidence
                  </div>
                  <div className="mt-2 tabular-nums">
                    {(result.duration_ms / 1000).toFixed(1)}s
                  </div>
                </div>
              </div>

              {/* Probability bars */}
              <div className="mt-6 space-y-2.5">
                {(["Real", "Misinformation", "Disinformation"] as const).map(
                  (cls) => {
                    const pct = (result.probs[cls] || 0) * 100;
                    return (
                      <div
                        key={cls}
                        className="flex items-center gap-3 text-xs"
                      >
                        <span className="w-28 shrink-0 font-medium">
                          {cls}
                        </span>
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/50">
                          <div
                            className={`h-full transition-all duration-700 ${verdictStyles(cls).bar}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="w-12 text-right font-mono tabular-nums">
                          {pct.toFixed(1)}%
                        </span>
                      </div>
                    );
                  },
                )}
              </div>
            </div>

            {/* Summary — plain-English explanation */}
            {result.summary && (
              <div className="rounded-2xl border border-stone-200 bg-white p-6 shadow-sm">
                <div className="mb-3 flex items-center gap-2">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-4 w-4 text-indigo-600"
                    aria-hidden
                  >
                    <path d="M14 4.1V2H4a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-9.9" />
                    <path d="M16 2v6h6" />
                    <path d="M8 13h6" />
                    <path d="M8 17h8" />
                  </svg>
                  <h3 className="text-xs font-medium uppercase tracking-wider text-stone-500">
                    Summary
                  </h3>
                </div>
                <div className="space-y-3 text-[15px] leading-relaxed text-stone-800">
                  {result.summary
                    .split(/\n\n+/)
                    .filter((p) => p.trim().length > 0)
                    .map((para, i) => (
                      <p key={i}>{renderInlineMarkdown(para.trim())}</p>
                    ))}
                </div>
              </div>
            )}

            {/* Sub-claims (only when >1) */}
            {result.sub_results.length > 1 && (
              <div>
                <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-stone-500">
                  Sub-claims · {result.sub_results.length}
                </h3>
                <div className="space-y-2">
                  {result.sub_results.map((sr, i) => (
                    <div
                      key={i}
                      className="rounded-xl border border-stone-200 bg-white p-4 shadow-sm"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <p className="text-sm text-stone-900">
                          &ldquo;{sr.sub_claim}&rdquo;
                        </p>
                        <span
                          className={`shrink-0 rounded-full border px-2.5 py-0.5 text-xs font-medium ${verdictStyles(sr.verdict).pill}`}
                        >
                          {sr.verdict || "—"}{" "}
                          <span className="font-mono opacity-70">
                            {Math.round(sr.confidence * 100)}%
                          </span>
                        </span>
                      </div>
                      <p className="mt-2 text-xs text-stone-500">
                        {sr.reason}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Top evidence */}
            <div>
              <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-stone-500">
                Evidence
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
                      style={{ animationDelay: `${i * 80}ms` }}
                      className="animate-fade-up block rounded-xl border border-stone-200 bg-white p-4 shadow-sm transition hover:-translate-y-0.5 hover:border-stone-300 hover:shadow-md"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex min-w-0 items-start gap-3">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={faviconUrl(ev.url, 32)}
                            alt=""
                            aria-hidden
                            width={20}
                            height={20}
                            className="mt-0.5 h-5 w-5 shrink-0 rounded-sm bg-stone-100"
                            onError={(e) => {
                              (e.currentTarget as HTMLImageElement).style.visibility = "hidden";
                            }}
                          />
                          <div className="min-w-0">
                            <p className="truncate text-sm font-medium text-stone-900">
                              {ev.title}
                            </p>
                            <p className="mt-0.5 truncate font-mono text-[11px] text-stone-500">
                              {domain(ev.url)}
                            </p>
                          </div>
                        </div>
                        <span
                          className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider ${
                            ev.source_is_fulltext
                              ? "bg-stone-100 text-stone-600"
                              : "bg-amber-50 text-amber-700"
                          }`}
                        >
                          {ev.source_is_fulltext ? "fulltext" : "snippet"}
                        </span>
                      </div>
                      <p className="mt-2.5 line-clamp-3 text-sm leading-relaxed text-stone-700">
                        {ev.snippet}
                      </p>
                      <div className="mt-3 flex gap-4 font-mono text-[11px] tabular-nums text-stone-500">
                        <span>
                          entail{" "}
                          <span className="font-semibold text-emerald-700">
                            {(ev.entailment * 100).toFixed(0)}%
                          </span>
                        </span>
                        <span>
                          neutral{" "}
                          <span className="font-semibold text-stone-600">
                            {(ev.neutral * 100).toFixed(0)}%
                          </span>
                        </span>
                        <span>
                          contra{" "}
                          <span className="font-semibold text-rose-700">
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

        {/* How factforge works — agent architecture explainer */}
        <section className="mt-20">
          <h2 className="text-xs font-medium uppercase tracking-wider text-stone-500">
            How it works
          </h2>
          <p className="mt-2 text-lg font-medium text-stone-900">
            Five specialized models in a graph, not one black-box LLM.
          </p>
          <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {[
              {
                num: "1",
                label: "Decompose",
                desc: "Splits compound claims into atomic, individually-verifiable sub-claims. Handles image inputs.",
                tech: "Gemini 2.5 Flash · vision",
                tint: "from-indigo-500/10 to-indigo-500/0",
              },
              {
                num: "2",
                label: "Retrieve",
                desc: "Two web queries per sub-claim (raw + fact-check-biased). Page text extracted in parallel.",
                tech: "DuckDuckGo · no API key",
                tint: "from-sky-500/10 to-sky-500/0",
              },
              {
                num: "3",
                label: "Verify",
                desc: "Scores each (evidence, sub-claim) pair as entail / neutral / contradict.",
                tech: "DeBERTa-v3-large MNLI+FEVER",
                tint: "from-violet-500/10 to-violet-500/0",
              },
              {
                num: "4",
                label: "Synthesize",
                desc: "Topical-relevance gate + claim-weighted max/mean blend → normalized 3-class verdict.",
                tech: "deterministic aggregation",
                tint: "from-amber-500/10 to-amber-500/0",
              },
              {
                num: "5",
                label: "Summarize",
                desc: "Generates the plain-English explanation citing named sources.",
                tech: "Gemini 2.5 Flash",
                tint: "from-rose-500/10 to-rose-500/0",
              },
            ].map((step) => (
              <div
                key={step.num}
                className={`relative overflow-hidden rounded-xl border border-stone-200 bg-gradient-to-br ${step.tint} bg-white p-4 shadow-sm transition hover:-translate-y-0.5 hover:shadow-md`}
              >
                <div className="font-mono text-xs font-medium text-stone-400">
                  0{step.num}
                </div>
                <h3 className="mt-1 text-sm font-semibold text-stone-900">
                  {step.label}
                </h3>
                <p className="mt-1.5 text-xs leading-relaxed text-stone-600">
                  {step.desc}
                </p>
                <p className="mt-3 font-mono text-[10px] uppercase tracking-wider text-stone-400">
                  {step.tech}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Footer */}
        <footer className="mt-16 border-t border-stone-200 pt-6 text-center">
          <p className="text-xs text-stone-500">
            Built by{" "}
            <a
              href="https://github.com/asutoshpaluri"
              className="font-medium text-stone-700 hover:text-stone-900"
            >
              Asutosh Paluri
            </a>{" "}
            · MS Computational Linguistics, UNT 2026
          </p>
          <p className="mt-2 text-[11px] text-stone-400">
            Research demo. Not journalism. Do not use to fact-check named
            individuals.
          </p>
        </footer>
      </main>
    </div>
  );
}
