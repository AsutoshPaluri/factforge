-- =========================================================================
-- factforge — initial schema: claims + feedback
--
-- Run once in your Supabase project:
--   Dashboard -> SQL Editor -> New query -> paste this whole file -> Run
--
-- After running, verify in Dashboard -> Table Editor that two tables exist:
--   - claims
--   - feedback
-- =========================================================================

-- pgcrypto gives us gen_random_uuid(). Free, standard.
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- -------------------------------------------------------------------------
-- claims: every verdict the agent produces (original + refined re-runs)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.claims (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Inputs
    claim           TEXT        NOT NULL,
    was_multimodal  BOOLEAN     NOT NULL DEFAULT FALSE,

    -- Agent outputs
    sub_claims      JSONB       NOT NULL DEFAULT '[]'::jsonb,
    verdict         TEXT        NOT NULL CHECK (
                                  verdict IN ('Real', 'Misinformation', 'Disinformation')
                                ),
    confidence      REAL        NOT NULL,
    probs           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    reason          TEXT,
    summary         TEXT,
    sub_results     JSONB       NOT NULL DEFAULT '[]'::jsonb,
    duration_ms     INTEGER     NOT NULL DEFAULT 0,

    -- Linkage: when an agent run is a *refinement* of a previous verdict,
    -- parent_claim_id points back to the original. feedback_used stores
    -- the user feedback that triggered the re-run.
    parent_claim_id UUID        REFERENCES public.claims(id) ON DELETE SET NULL,
    feedback_used   TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_claims_created_at
    ON public.claims (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_claims_parent
    ON public.claims (parent_claim_id)
    WHERE parent_claim_id IS NOT NULL;


-- -------------------------------------------------------------------------
-- feedback: thumbs + comment from users on specific verdicts.
-- One claim can have many feedback rows. Cascade-delete with the claim.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.feedback (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id      UUID         NOT NULL
                                REFERENCES public.claims(id) ON DELETE CASCADE,
    kind          TEXT         NOT NULL CHECK (kind IN ('good', 'bad')),
    comment       TEXT,

    -- SHA-256(ip)[:16] for soft per-IP spam prevention. NOT for identifying users.
    user_ip_hash  TEXT,

    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_claim_id
    ON public.feedback (claim_id);

CREATE INDEX IF NOT EXISTS idx_feedback_created_at
    ON public.feedback (created_at DESC);


-- -------------------------------------------------------------------------
-- Row-Level Security
-- -------------------------------------------------------------------------
-- The backend uses SUPABASE_SECRET_KEY (service_role) which bypasses RLS,
-- so we don't strictly need policies right now. Enabling RLS as a safety
-- net so that if anyone ever accidentally uses the publishable key
-- against these tables, they hit "no policy" deny-by-default rather than
-- a wide-open read/write.
ALTER TABLE public.claims   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.feedback ENABLE ROW LEVEL SECURITY;

-- (No public policies. Service-role key bypasses RLS — backend will work.
--  Add explicit "anon can read public claims" policies in v2 if/when we
--  want to expose a /v/{slug} shareable URL endpoint.)
