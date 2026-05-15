-- =========================================================================
-- factforge — migration 003: rename verdict labels
--
-- Schema:
--   OLD:  Real          | Misinformation | Disinformation
--   NEW:  Credible      | Uncertain      | Not Credible
--
-- Why: 'Disinformation' implies intent; 'Credible/Not Credible' is honest
-- about what an evidence-grounded fact-checker actually does. 'Uncertain'
-- as an explicit abstain label lets us measure clean binary accuracy on
-- the cases where the agent commits, while transparently flagging
-- genuinely-ambiguous claims to users.
--
-- Run once in Supabase SQL Editor (new tab as before).
-- =========================================================================

-- 1. Drop the old CHECK constraint (so the UPDATE doesn't fail mid-flight).
ALTER TABLE public.claims
    DROP CONSTRAINT IF EXISTS claims_verdict_check;

-- 2. Migrate existing rows to new labels.
UPDATE public.claims
SET verdict = CASE verdict
    WHEN 'Real'            THEN 'Credible'
    WHEN 'Disinformation'  THEN 'Not Credible'
    WHEN 'Misinformation'  THEN 'Uncertain'
    ELSE verdict
END;

-- 3. Re-add CHECK constraint with new vocabulary.
ALTER TABLE public.claims
    ADD CONSTRAINT claims_verdict_check
    CHECK (verdict IN ('Credible', 'Not Credible', 'Uncertain'));
