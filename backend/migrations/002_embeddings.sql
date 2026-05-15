-- =========================================================================
-- factforge — migration 002: claim embeddings for memory-augmented learning
--
-- Adds a 384-dim vector column on `claims` and an HNSW index for fast
-- cosine-similarity lookups. Used by the agent to find semantically
-- similar past claims and pull their user feedback as additional context
-- on every new agent run.
--
-- Run once in Supabase SQL Editor (same flow as 001).
-- =========================================================================

-- pgvector should already be enabled from the Database -> Extensions step.
-- Safe to declare again (no-op if already on).
CREATE EXTENSION IF NOT EXISTS vector;

-- 384 dims matches sentence-transformers/all-MiniLM-L6-v2 output.
ALTER TABLE public.claims
    ADD COLUMN IF NOT EXISTS claim_embedding vector(384);

-- HNSW index for fast approximate nearest-neighbour search by cosine distance.
-- Order matters: cosine works well for sentence embeddings; pgvector also
-- supports L2 + inner product, but cosine is the default for text similarity.
CREATE INDEX IF NOT EXISTS idx_claims_embedding
    ON public.claims
    USING hnsw (claim_embedding vector_cosine_ops);

-- Optional: convenience function — given a text embedding, return the top-K
-- similar past claims AND their associated bad-feedback comments. We do
-- this in Python for now, but exposing it as a SQL function later would
-- let us push the join into the DB. Skipping for v2; keeping the join in
-- application code for visibility.
