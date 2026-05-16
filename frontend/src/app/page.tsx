"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Verdict = "Credible" | "Uncertain" | "Not Credible";

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
  claim_id: string | null;
  parent_claim_id: string | null;
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

type FeedbackStatus =
  | { state: "idle" }
  | { state: "submitting" }
  | { state: "thanked" }
  | { state: "error"; message: string };

type RefineStatus =
  | { state: "idle" }
  | { state: "submitting" }
  | { state: "error"; message: string };

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
    case "Credible":
      return {
        glow: "glow-real",
        text: "text-emerald-300",
        accent: "text-emerald-400",
        bar: "bg-gradient-to-r from-emerald-500 to-emerald-400",
        pill: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
        ring: "ring-emerald-500/30",
      };
    case "Uncertain":
      return {
        glow: "glow-misinfo",
        text: "text-amber-300",
        accent: "text-amber-400",
        bar: "bg-gradient-to-r from-amber-500 to-amber-400",
        pill: "bg-amber-500/10 text-amber-300 border-amber-500/30",
        ring: "ring-amber-500/30",
      };
    case "Not Credible":
      return {
        glow: "glow-disinfo",
        text: "text-rose-300",
        accent: "text-rose-400",
        bar: "bg-gradient-to-r from-rose-500 to-rose-400",
        pill: "bg-rose-500/10 text-rose-300 border-rose-500/30",
        ring: "ring-rose-500/30",
      };
    default:
      return {
        glow: "",
        text: "text-zinc-300",
        accent: "text-zinc-400",
        bar: "bg-zinc-600",
        pill: "bg-zinc-800/60 text-zinc-300 border-zinc-700",
        ring: "ring-zinc-700",
      };
  }
};

const verdictIcon = (v: Verdict | string) => {
  switch (v) {
    case "Credible":
      return "✓";
    case "Uncertain":
      return "?";
    case "Not Credible":
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
function renderInlineMarkdown(text: string): React.ReactNode[] {
  const parts = text.split(/(\*\*[^*\n]+\*\*|\*[^*\n]+\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={i} className="font-semibold text-zinc-100">
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
        <em key={i} className="italic text-zinc-100">
          {part.slice(1, -1)}
        </em>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

// Block-level renderer for the summary: handles paragraphs, "Key reasons:"
// style headers, and dash-bulleted lists with **bold lead-ins**.
function renderSummary(text: string): React.ReactNode[] {
  const blocks = text.split(/\n\n+/).filter((b) => b.trim().length > 0);
  return blocks.map((block, i) => {
    const lines = block
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    // List block: every line starts with "- " or "• "
    if (
      lines.length > 0 &&
      lines.every((l) => l.startsWith("- ") || l.startsWith("• "))
    ) {
      return (
        <ul key={i} className="mt-1 space-y-3">
          {lines.map((line, j) => {
            const content = line.replace(/^(-|•)\s+/, "");
            return (
              <li
                key={j}
                className="flex gap-3 text-[15px] leading-relaxed text-zinc-300"
              >
                <span
                  className="mt-2.5 block h-[5px] w-[5px] shrink-0 rounded-full bg-indigo-400"
                  aria-hidden
                />
                <span className="min-w-0">{renderInlineMarkdown(content)}</span>
              </li>
            );
          })}
        </ul>
      );
    }

    // Section header: short line ending in ":" with no other content
    if (
      lines.length === 1 &&
      /^[A-Z][^.!?]+:$/.test(lines[0]) &&
      lines[0].length < 40
    ) {
      return (
        <p
          key={i}
          className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-300"
        >
          {lines[0].replace(/:$/, "")}
        </p>
      );
    }

    // Regular paragraph — join multi-line into single flow
    return (
      <p key={i} className="text-[15px] leading-relaxed text-zinc-300">
        {renderInlineMarkdown(lines.join(" "))}
      </p>
    );
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

  // Feedback / refinement state — resets every time a new verdict lands
  const [feedbackKind, setFeedbackKind] = useState<"good" | "bad" | null>(null);
  const [feedbackComment, setFeedbackComment] = useState("");
  const [feedbackStatus, setFeedbackStatus] = useState<FeedbackStatus>({
    state: "idle",
  });
  const [refineStatus, setRefineStatus] = useState<RefineStatus>({
    state: "idle",
  });

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

  // Revoke object URL on unmount / change
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

  const resetFeedbackState = () => {
    setFeedbackKind(null);
    setFeedbackComment("");
    setFeedbackStatus({ state: "idle" });
    setRefineStatus({ state: "idle" });
  };

  const submit = async (claimToCheck?: string) => {
    const c = (claimToCheck ?? claim).trim();
    if (c.length < 3 && !imageFile) return;
    if (claimToCheck !== undefined) setClaim(claimToCheck);
    setLoading(true);
    setError(null);
    setResult(null);
    resetFeedbackState();

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
        body: JSON.stringify({ claim: c, image_b64, image_mime }),
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

  /** POST /claims/{id}/feedback — store thumbs + comment, no agent re-run. */
  const submitFeedback = async (kind: "good" | "bad") => {
    if (!result?.claim_id) {
      setFeedbackStatus({
        state: "error",
        message: "No claim_id — feedback can't be saved.",
      });
      return;
    }
    setFeedbackStatus({ state: "submitting" });
    try {
      const res = await fetch(
        `${API_URL}/api/v1/claims/${result.claim_id}/feedback`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            kind,
            comment: feedbackComment.trim() || null,
          }),
        },
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }
      setFeedbackStatus({ state: "thanked" });
    } catch (e) {
      setFeedbackStatus({
        state: "error",
        message: e instanceof Error ? e.message : String(e),
      });
    }
  };

  /** POST /claims/{id}/refine — re-run agent with feedback in context. */
  const submitRefine = async () => {
    if (!result?.claim_id) return;
    const fb = feedbackComment.trim();
    if (fb.length < 10) {
      setRefineStatus({
        state: "error",
        message: "Please write at least 10 characters of feedback.",
      });
      return;
    }
    setRefineStatus({ state: "submitting" });
    setLoading(true);
    setCurrentStep(0);

    try {
      const res = await fetch(
        `${API_URL}/api/v1/claims/${result.claim_id}/refine`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ feedback: fb }),
        },
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const msg = body?.detail || `HTTP ${res.status}`;
        throw new Error(msg);
      }
      const data: ClaimResponse = await res.json();
      setResult(data);
      // Clear feedback state — but result now has parent_claim_id, which
      // we render as a "Refined" badge so the user knows what happened.
      setFeedbackKind(null);
      setFeedbackComment("");
      setFeedbackStatus({ state: "idle" });
      setRefineStatus({ state: "idle" });
    } catch (e) {
      setRefineStatus({
        state: "error",
        message: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen text-zinc-100">
      {/* Animated aurora background */}
      <div className="aurora" aria-hidden />

      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-white/5 backdrop-blur-md">
        <div className="absolute inset-0 bg-zinc-950/50" aria-hidden />
        <div className="relative mx-auto flex max-w-4xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2.5">
            <span className="relative flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-indigo-500 to-fuchsia-500 shadow-[0_0_20px_-2px_rgba(129,140,248,0.6)]">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth={2.5}
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4 text-white"
                aria-hidden
              >
                <circle cx="10.5" cy="10.5" r="6.5" />
                <path d="m21 21-5.5-5.5" />
                <path d="m8 10.5 2 2 4-4" />
              </svg>
            </span>
            <span className="text-lg font-semibold tracking-tight text-zinc-100">
              factforge
            </span>
            <span className="rounded-full bg-zinc-800/80 px-1.5 py-0.5 font-mono text-[10px] text-zinc-400">
              v0.1
            </span>
          </div>
          <nav className="flex items-center gap-5 text-sm">
            <a
              href={`${API_URL}/docs`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-zinc-400 transition hover:text-zinc-100"
            >
              API
            </a>
            <a
              href="https://github.com/asutoshpaluri"
              target="_blank"
              rel="noopener noreferrer"
              className="text-zinc-400 transition hover:text-zinc-100"
            >
              GitHub
            </a>
          </nav>
        </div>
      </header>

      <main className="relative mx-auto max-w-3xl px-6 py-16">
        {/* Hero */}
        <section className="mb-12">
          <p className="mb-3 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-zinc-300 backdrop-blur-sm">
            <span className="block h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
            Multimodal · Evidence-grounded · No tracking
          </p>
          <h1 className="text-5xl font-bold tracking-tight sm:text-6xl">
            Fact-check{" "}
            <span className="gradient-text">anything</span>
            <span className="text-zinc-100">.</span>
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-relaxed text-zinc-400">
            Submit a claim as text or an image. An AI agent decomposes it,
            retrieves real web evidence, scores it with a fine-tuned NLI
            model, and writes a plain-English verdict citing the sources.
          </p>
        </section>

        {/* Input card */}
        <section className="glass animate-fade-up rounded-2xl p-6">
          <label
            htmlFor="claim-input"
            className="block text-sm font-medium text-zinc-200"
          >
            Your claim{" "}
            {imageFile && (
              <span className="text-zinc-500">
                (optional — image will be analyzed)
              </span>
            )}
          </label>
          <textarea
            id="claim-input"
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
            disabled={loading}
            placeholder={
              imageFile
                ? "Optional context about the image"
                : "e.g. Humans only use 10 percent of their brain"
            }
            rows={3}
            className="mt-2 w-full resize-none rounded-lg border border-white/10 bg-zinc-900/60 px-4 py-3 text-zinc-100 placeholder:text-zinc-500 focus:border-indigo-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/20 disabled:opacity-60"
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
                className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-zinc-900/60 px-3 py-2 text-xs font-medium text-zinc-300 transition hover:border-white/20 hover:bg-zinc-800/80 hover:text-zinc-100 disabled:cursor-not-allowed disabled:opacity-50"
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
              <div className="flex items-start gap-3 rounded-lg border border-white/10 bg-zinc-900/60 p-3">
                {imagePreview && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={imagePreview}
                    alt="claim preview"
                    className="h-20 w-20 shrink-0 rounded-md border border-white/10 object-cover"
                  />
                )}
                <div className="min-w-0 flex-1">
                  <p className="truncate text-xs font-medium text-zinc-200">
                    {imageFile.name}
                  </p>
                  <p className="mt-0.5 font-mono text-[11px] text-zinc-500">
                    {imageFile.type} · {(imageFile.size / 1024).toFixed(0)} KB
                  </p>
                  <button
                    type="button"
                    onClick={() => handleFileSelect(null)}
                    disabled={loading}
                    className="mt-2 text-xs font-medium text-rose-400 transition hover:text-rose-300 disabled:opacity-50"
                  >
                    Remove
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Example chips */}
          <div className="mt-4 flex flex-wrap gap-2">
            <span className="text-xs font-medium text-zinc-500">Try:</span>
            {EXAMPLE_CLAIMS.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => !loading && submit(ex)}
                disabled={loading}
                className="rounded-full border border-white/10 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-300 transition hover:border-white/20 hover:bg-zinc-800/80 hover:text-zinc-100 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {ex}
              </button>
            ))}
          </div>

          <div className="mt-5 flex items-center justify-between gap-4">
            <p className="text-xs text-zinc-500">
              ~30s per claim · DDG + DeBERTa NLI on free-tier CPU
            </p>
            <button
              onClick={() => submit()}
              disabled={!canSubmit}
              className="group relative shrink-0 overflow-hidden rounded-lg bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-6 py-2.5 text-sm font-medium text-white shadow-[0_0_30px_-8px_rgba(129,140,248,0.6)] transition hover:from-indigo-400 hover:to-fuchsia-400 hover:shadow-[0_0_40px_-6px_rgba(217,70,239,0.7)] disabled:cursor-not-allowed disabled:from-zinc-700 disabled:to-zinc-700 disabled:text-zinc-500 disabled:shadow-none"
            >
              <span className="relative z-10">
                {loading ? "Checking…" : "Fact-check"}
              </span>
            </button>
          </div>
        </section>

        {/* Error */}
        {error && (
          <section className="mt-4 animate-fade-up rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 backdrop-blur-sm">
            <p className="text-sm font-medium text-rose-300">
              Something went wrong
            </p>
            <p className="mt-1 break-words text-xs text-rose-200/80">
              {error}
            </p>
          </section>
        )}

        {/* Loading — animated agent trace */}
        {loading && (
          <section className="glass animate-fade-up mt-6 rounded-2xl p-6">
            <div className="mb-4 flex items-center gap-2">
              <span className="block h-2 w-2 animate-pulse-dot rounded-full bg-indigo-400" />
              <p className="text-sm font-medium text-zinc-200">
                Agent working
                {imageFile && (
                  <span className="ml-2 rounded-full border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-xs font-medium text-indigo-300">
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
                    className={`relative flex items-start gap-3 overflow-hidden rounded-lg p-2.5 transition ${
                      active
                        ? "border border-indigo-500/30 bg-indigo-500/10"
                        : "border border-transparent"
                    }`}
                  >
                    {active && (
                      <span className="shimmer pointer-events-none absolute inset-0" />
                    )}
                    <span
                      className={`relative mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full font-mono text-xs ${
                        done
                          ? "bg-indigo-500 text-white shadow-[0_0_12px_-2px_rgba(129,140,248,0.8)]"
                          : active
                            ? "border border-indigo-500/50 bg-indigo-500/20 text-indigo-200"
                            : "border border-white/10 bg-zinc-900/60 text-zinc-500"
                      }`}
                    >
                      {done ? "✓" : i + 1}
                    </span>
                    <div className="relative min-w-0">
                      <p
                        className={`text-sm font-medium ${
                          active
                            ? "text-indigo-200"
                            : done
                              ? "text-zinc-300"
                              : "text-zinc-500"
                        }`}
                      >
                        {step.label}
                      </p>
                      <p className="text-xs text-zinc-500">{step.desc}</p>
                    </div>
                  </li>
                );
              })}
            </ol>
          </section>
        )}

        {/* Result */}
        {result && (
          <section className="mt-6 space-y-6">
            {/* Verdict hero card */}
            <div
              className={`glass animate-fade-up rounded-2xl p-6 ${verdictStyles(result.verdict).glow}`}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex items-center gap-3">
                    <span
                      className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-full ring-2 ${verdictStyles(result.verdict).ring} bg-zinc-900/60 font-mono text-2xl ${verdictStyles(result.verdict).accent}`}
                    >
                      {verdictIcon(result.verdict)}
                    </span>
                    <div>
                      <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-zinc-500">
                        Verdict
                      </p>
                      <h2
                        className={`text-3xl font-bold tracking-tight ${verdictStyles(result.verdict).text}`}
                      >
                        {result.verdict}
                      </h2>
                    </div>
                    {result.was_multimodal && (
                      <span className="rounded-full border border-indigo-500/30 bg-indigo-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-indigo-300">
                        multimodal
                      </span>
                    )}
                    {result.parent_claim_id && (
                      <span className="inline-flex items-center gap-1 rounded-full border border-fuchsia-500/30 bg-fuchsia-500/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-fuchsia-300">
                        <svg
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth={2.5}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          className="h-2.5 w-2.5"
                          aria-hidden
                        >
                          <path d="M3 12a9 9 0 0 1 15-6.7l3 3" />
                          <path d="M21 3v6h-6" />
                        </svg>
                        refined
                      </span>
                    )}
                  </div>
                  <p className="mt-3 text-sm text-zinc-400">{result.reason}</p>
                </div>
                <div className="shrink-0 text-right">
                  <div className="font-mono text-3xl font-semibold tabular-nums text-zinc-100">
                    {Math.round(result.confidence * 100)}
                    <span className="text-lg text-zinc-500">%</span>
                  </div>
                  <div className="mt-0.5 text-[10px] uppercase tracking-wider text-zinc-500">
                    confidence
                  </div>
                  <div className="mt-3 font-mono text-xs tabular-nums text-zinc-500">
                    {(result.duration_ms / 1000).toFixed(1)}s
                  </div>
                </div>
              </div>

              {/* Probability bars */}
              <div className="mt-6 space-y-2.5">
                {(["Credible", "Uncertain", "Not Credible"] as const).map(
                  (cls) => {
                    const pct = (result.probs[cls] || 0) * 100;
                    return (
                      <div
                        key={cls}
                        className="flex items-center gap-3 text-xs"
                      >
                        <span className="w-28 shrink-0 font-medium text-zinc-300">
                          {cls}
                        </span>
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-800/80">
                          <div
                            className={`h-full transition-all duration-1000 ease-out ${verdictStyles(cls).bar}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="w-12 text-right font-mono tabular-nums text-zinc-400">
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
              <div className="glass animate-fade-up rounded-2xl p-6 [animation-delay:80ms]">
                <div className="mb-3 flex items-center gap-2">
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="h-4 w-4 text-indigo-400"
                    aria-hidden
                  >
                    <path d="M14 4.1V2H4a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-9.9" />
                    <path d="M16 2v6h6" />
                    <path d="M8 13h6" />
                    <path d="M8 17h8" />
                  </svg>
                  <h3 className="text-[11px] font-medium uppercase tracking-[0.18em] text-zinc-500">
                    Summary
                  </h3>
                </div>
                <div className="space-y-4">
                  {renderSummary(result.summary)}
                </div>
              </div>
            )}

            {/* Sub-claims (only when >1) */}
            {result.sub_results.length > 1 && (
              <div className="animate-fade-up [animation-delay:160ms]">
                <h3 className="mb-3 text-[11px] font-medium uppercase tracking-[0.18em] text-zinc-500">
                  Sub-claims · {result.sub_results.length}
                </h3>
                <div className="space-y-2">
                  {result.sub_results.map((sr, i) => (
                    <div
                      key={i}
                      className="glass rounded-xl p-4"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <p className="text-sm text-zinc-200">
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
                      <p className="mt-2 text-xs text-zinc-500">{sr.reason}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Strongest evidence + All retrieved evidence (mirrors the
                misinfo_detection capstone's "Strongest evidence" card +
                "All retrieved evidence" expander pattern). */}
            {(() => {
              const allEvidence = result.sub_results.flatMap((sr) => sr.top_evidence);
              if (allEvidence.length === 0) return null;

              const informativeness = (e: Evidence) =>
                e.entailment + e.contradiction;
              const sorted = [...allEvidence].sort(
                (a, b) => informativeness(b) - informativeness(a),
              );
              const strongest = sorted[0];

              const accentBorder = (() => {
                switch (result.verdict) {
                  case "Credible":
                    return "border-l-emerald-500";
                  case "Uncertain":
                    return "border-l-amber-500";
                  case "Not Credible":
                    return "border-l-rose-500";
                  default:
                    return "border-l-zinc-700";
                }
              })();

              return (
                <>
                  {/* Strongest evidence — hero card */}
                  <div
                    className={`glass animate-fade-up rounded-2xl border-l-4 ${accentBorder} p-6 [animation-delay:240ms]`}
                  >
                    <div className="mb-4 flex items-center gap-2">
                      <svg
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth={2}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="h-4 w-4 text-indigo-400"
                        aria-hidden
                      >
                        <path d="M3 11l3 3 8-8" />
                        <path d="M3 17l3 3 8-8" />
                      </svg>
                      <h3 className="text-[11px] font-medium uppercase tracking-[0.18em] text-zinc-500">
                        Strongest evidence
                      </h3>
                    </div>

                    <blockquote className="border-l-2 border-zinc-700 pl-4 text-[15px] italic leading-relaxed text-zinc-200">
                      &ldquo;
                      {strongest.snippet.length > 360
                        ? strongest.snippet.slice(0, 360).trimEnd() + "..."
                        : strongest.snippet}
                      &rdquo;
                    </blockquote>

                    <div className="mt-4 flex flex-wrap items-center gap-2 text-sm">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={faviconUrl(strongest.url, 32)}
                        alt=""
                        aria-hidden
                        width={18}
                        height={18}
                        className="h-[18px] w-[18px] shrink-0 rounded-sm bg-white/10"
                        onError={(e) => {
                          (e.currentTarget as HTMLImageElement).style.visibility = "hidden";
                        }}
                      />
                      <a
                        href={strongest.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="font-medium text-indigo-300 underline-offset-4 transition hover:text-indigo-200 hover:underline"
                      >
                        {strongest.title}
                      </a>
                      <span className="font-mono text-[11px] text-zinc-500">
                        · {domain(strongest.url)}
                      </span>
                    </div>

                    <div className="mt-3 flex gap-4 font-mono text-[11px] tabular-nums text-zinc-500">
                      <span>
                        NLI · entail{" "}
                        <span className="font-semibold text-emerald-400">
                          {(strongest.entailment * 100).toFixed(0)}%
                        </span>
                      </span>
                      <span>
                        neutral{" "}
                        <span className="font-semibold text-zinc-400">
                          {(strongest.neutral * 100).toFixed(0)}%
                        </span>
                      </span>
                      <span>
                        contra{" "}
                        <span className="font-semibold text-rose-400">
                          {(strongest.contradiction * 100).toFixed(0)}%
                        </span>
                      </span>
                    </div>
                  </div>

                  {/* All retrieved sources — collapsible compact list */}
                  <details className="glass group animate-fade-up rounded-2xl [animation-delay:320ms]">
                    <summary className="flex cursor-pointer list-none items-center justify-between px-6 py-4 select-none">
                      <h3 className="text-[11px] font-medium uppercase tracking-[0.18em] text-zinc-500">
                        All retrieved sources · {sorted.length}
                      </h3>
                      <span className="flex items-center gap-2 text-xs text-zinc-500">
                        <span className="hidden sm:inline group-open:hidden">
                          show table
                        </span>
                        <span className="hidden sm:inline group-open:inline-block">
                          hide
                        </span>
                        <svg
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth={2.2}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          className="h-4 w-4 transition-transform group-open:rotate-180"
                          aria-hidden
                        >
                          <path d="M6 9l6 6 6-6" />
                        </svg>
                      </span>
                    </summary>

                    <div className="border-t border-white/5">
                      <div className="hidden grid-cols-[1fr_auto_auto_auto] gap-4 border-b border-white/5 px-6 py-2 font-mono text-[10px] uppercase tracking-wider text-zinc-500 sm:grid">
                        <span>Source</span>
                        <span className="w-12 text-right">entail</span>
                        <span className="w-12 text-right">neutral</span>
                        <span className="w-12 text-right">contra</span>
                      </div>

                      <div className="divide-y divide-white/5">
                        {sorted.map((ev, i) => (
                          <a
                            key={i}
                            href={ev.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="grid grid-cols-[1fr_auto] gap-4 px-6 py-3 transition hover:bg-white/[0.02] sm:grid-cols-[1fr_auto_auto_auto]"
                          >
                            <div className="flex min-w-0 items-center gap-3">
                              {/* eslint-disable-next-line @next/next/no-img-element */}
                              <img
                                src={faviconUrl(ev.url, 32)}
                                alt=""
                                aria-hidden
                                width={16}
                                height={16}
                                className="h-4 w-4 shrink-0 rounded-sm bg-white/10"
                                onError={(e) => {
                                  (e.currentTarget as HTMLImageElement).style.visibility = "hidden";
                                }}
                              />
                              <div className="min-w-0">
                                <p className="truncate text-sm text-zinc-200">
                                  {ev.title}
                                </p>
                                <p className="truncate font-mono text-[11px] text-zinc-500">
                                  {domain(ev.url)}
                                </p>
                              </div>
                            </div>
                            <div className="hidden w-12 text-right font-mono text-xs tabular-nums text-emerald-400 sm:block">
                              {(ev.entailment * 100).toFixed(0)}%
                            </div>
                            <div className="hidden w-12 text-right font-mono text-xs tabular-nums text-zinc-400 sm:block">
                              {(ev.neutral * 100).toFixed(0)}%
                            </div>
                            <div className="hidden w-12 text-right font-mono text-xs tabular-nums text-rose-400 sm:block">
                              {(ev.contradiction * 100).toFixed(0)}%
                            </div>
                            {/* Mobile: stacked scores */}
                            <div className="flex gap-3 font-mono text-[11px] tabular-nums sm:hidden">
                              <span className="text-emerald-400">
                                {(ev.entailment * 100).toFixed(0)}%
                              </span>
                              <span className="text-zinc-500">
                                {(ev.neutral * 100).toFixed(0)}%
                              </span>
                              <span className="text-rose-400">
                                {(ev.contradiction * 100).toFixed(0)}%
                              </span>
                            </div>
                          </a>
                        ))}
                      </div>
                      <p className="border-t border-white/5 px-6 py-3 text-[11px] text-zinc-500">
                        <span className="text-emerald-400">entail</span> = source
                        supports claim &middot;{" "}
                        <span className="text-rose-400">contra</span> = source
                        refutes claim &middot;{" "}
                        <span className="text-zinc-400">neutral</span> = source is
                        unrelated or inconclusive
                      </p>
                    </div>
                  </details>
                </>
              );
            })()}

            {/* Feedback bar — collects user signal + can trigger a refine
                (agent re-runs with feedback in context). The "true agent"
                piece: human-in-the-loop refinement loop. */}
            {result.claim_id && (
              <div className="glass animate-fade-up rounded-2xl p-6 [animation-delay:400ms]">
                {feedbackStatus.state === "thanked" ? (
                  <div className="flex items-center gap-2 text-sm text-emerald-300">
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={2.2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-4 w-4"
                      aria-hidden
                    >
                      <path d="M20 6 9 17l-5-5" />
                    </svg>
                    Thanks — your feedback is recorded.
                  </div>
                ) : (
                  <>
                    <div className="flex items-center justify-between gap-4">
                      <h3 className="text-[11px] font-medium uppercase tracking-[0.18em] text-zinc-500">
                        Was this verdict helpful?
                      </h3>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            setFeedbackKind("good");
                            submitFeedback("good");
                          }}
                          disabled={feedbackStatus.state === "submitting"}
                          className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition disabled:opacity-50 ${
                            feedbackKind === "good"
                              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                              : "border-white/10 bg-zinc-900/60 text-zinc-300 hover:border-emerald-500/30 hover:text-emerald-300"
                          }`}
                        >
                          <span aria-hidden>👍</span> Looks right
                        </button>
                        <button
                          type="button"
                          onClick={() => setFeedbackKind("bad")}
                          disabled={feedbackStatus.state === "submitting"}
                          className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition disabled:opacity-50 ${
                            feedbackKind === "bad"
                              ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
                              : "border-white/10 bg-zinc-900/60 text-zinc-300 hover:border-rose-500/30 hover:text-rose-300"
                          }`}
                        >
                          <span aria-hidden>👎</span> Something&rsquo;s off
                        </button>
                      </div>
                    </div>

                    {feedbackKind === "bad" && (
                      <div className="mt-4 space-y-3">
                        <label
                          htmlFor="feedback-input"
                          className="block text-xs font-medium text-zinc-300"
                        >
                          What did the agent miss? <span className="text-zinc-500">(specific feedback gives a better refined verdict)</span>
                        </label>
                        <textarea
                          id="feedback-input"
                          value={feedbackComment}
                          onChange={(e) => setFeedbackComment(e.target.value)}
                          placeholder="e.g. 'you ignored peer-reviewed studies', 'the source on point 2 is unreliable', 'this misreads the claim — it's actually about X'"
                          rows={2}
                          maxLength={500}
                          disabled={
                            feedbackStatus.state === "submitting" ||
                            refineStatus.state === "submitting"
                          }
                          className="w-full resize-none rounded-lg border border-white/10 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-rose-400/60 focus:outline-none focus:ring-4 focus:ring-rose-500/15 disabled:opacity-50"
                        />
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <p className="text-[11px] text-zinc-500">
                            <span className="text-zinc-400">{feedbackComment.length}/500</span> &middot; min 10 chars to refine
                          </p>
                          <div className="flex gap-2">
                            <button
                              type="button"
                              onClick={() => submitFeedback("bad")}
                              disabled={
                                feedbackStatus.state === "submitting" ||
                                refineStatus.state === "submitting"
                              }
                              className="rounded-lg border border-white/10 bg-zinc-900/60 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-white/20 hover:text-zinc-100 disabled:opacity-50"
                            >
                              {feedbackStatus.state === "submitting"
                                ? "Saving…"
                                : "Just submit feedback"}
                            </button>
                            <button
                              type="button"
                              onClick={submitRefine}
                              disabled={
                                refineStatus.state === "submitting" ||
                                feedbackComment.trim().length < 10
                              }
                              className="rounded-lg bg-gradient-to-r from-indigo-500 to-fuchsia-500 px-3 py-1.5 text-xs font-medium text-white shadow-[0_0_20px_-6px_rgba(129,140,248,0.6)] transition hover:from-indigo-400 hover:to-fuchsia-400 disabled:cursor-not-allowed disabled:from-zinc-700 disabled:to-zinc-700 disabled:text-zinc-500 disabled:shadow-none"
                            >
                              {refineStatus.state === "submitting"
                                ? "Refining…"
                                : "Refine verdict ↻"}
                            </button>
                          </div>
                        </div>
                        {feedbackStatus.state === "error" && (
                          <p className="text-xs text-rose-300">
                            {feedbackStatus.message}
                          </p>
                        )}
                        {refineStatus.state === "error" && (
                          <p className="text-xs text-rose-300">
                            {refineStatus.message}
                          </p>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </section>
        )}

        {/* How factforge works — agent architecture explainer */}
        <section className="mt-24">
          <h2 className="text-[11px] font-medium uppercase tracking-[0.18em] text-zinc-500">
            How it works
          </h2>
          <p className="mt-2 text-2xl font-semibold tracking-tight text-zinc-100">
            Five specialized stages in a graph.
            <span className="block text-zinc-500">
              Not one black-box LLM.
            </span>
          </p>
          <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            {[
              {
                num: "01",
                label: "Decompose",
                desc: "Splits compound claims into atomic, individually-verifiable sub-claims. Handles image inputs.",
                tech: "Gemini · vision",
                color: "from-indigo-500/30 to-indigo-500/0",
                ring: "ring-indigo-500/30",
              },
              {
                num: "02",
                label: "Retrieve",
                desc: "Two web queries per sub-claim (raw + fact-check-biased). Pages fetched + cleaned in parallel.",
                tech: "DuckDuckGo · no API key",
                color: "from-sky-500/30 to-sky-500/0",
                ring: "ring-sky-500/30",
              },
              {
                num: "03",
                label: "Verify",
                desc: "Scores each (snippet, claim) pair: entail / neutral / contradict.",
                tech: "DeBERTa-v3-large MNLI+FEVER",
                color: "from-violet-500/30 to-violet-500/0",
                ring: "ring-violet-500/30",
              },
              {
                num: "04",
                label: "Synthesize",
                desc: "Topical-relevance gate + claim-weighted max/mean blend → normalized 3-class verdict.",
                tech: "deterministic aggregation",
                color: "from-amber-500/30 to-amber-500/0",
                ring: "ring-amber-500/30",
              },
              {
                num: "05",
                label: "Summarize",
                desc: "Writes the plain-English explanation citing named sources.",
                tech: "Gemini",
                color: "from-rose-500/30 to-rose-500/0",
                ring: "ring-rose-500/30",
              },
            ].map((step) => (
              <div
                key={step.num}
                className={`glass group relative overflow-hidden rounded-xl p-4 transition hover:-translate-y-0.5 hover:ring-1 ${step.ring}`}
              >
                <div
                  className={`absolute inset-0 -z-10 bg-gradient-to-br ${step.color} opacity-50 transition group-hover:opacity-100`}
                />
                <div className="font-mono text-xs font-medium text-zinc-500">
                  {step.num}
                </div>
                <h3 className="mt-1 text-sm font-semibold text-zinc-100">
                  {step.label}
                </h3>
                <p className="mt-1.5 text-xs leading-relaxed text-zinc-400">
                  {step.desc}
                </p>
                <p className="mt-3 font-mono text-[10px] uppercase tracking-wider text-zinc-500">
                  {step.tech}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Footer */}
        <footer className="mt-20 border-t border-white/5 pt-6 text-center">
          <p className="text-xs text-zinc-500">
            Built by{" "}
            <a
              href="https://github.com/asutoshpaluri"
              className="font-medium text-zinc-300 transition hover:text-zinc-100"
            >
              Asutosh Paluri
            </a>{" "}
            · MS Computational Linguistics, UNT 2026
          </p>
          <p className="mt-2 text-[11px] text-zinc-600">
            Research demo. Not journalism. Do not use to fact-check named
            individuals.
          </p>
        </footer>
      </main>
    </div>
  );
}
