# NewsProof Architecture

NewsProof is a monorepo with a Next.js frontend and a FastAPI backend.

## Frontend

- `app/` contains the home, result, history, and admin pages.
- `components/` contains reusable UI for status labels, result sections, and publisher management.
- `lib/api.ts` calls the live backend endpoints under `/api/v1`.

## Backend

- `app/routers/` exposes verification, result, audit, and publisher endpoints.
- `app/services/verification.py` runs the 4-layer verification pipeline.
- `app/services/content_extraction.py` fetches and normalizes URL content.
- `app/services/ocr.py` extracts text from uploaded images with Tesseract.
- `app/services/cohere_service.py` uses Cohere when configured and falls back gracefully otherwise.
- `app/services/audit.py` writes append-only hash-chained audit entries.

## Data flow

1. The client submits a URL, text payload, or image upload.
2. The backend extracts or normalizes content.
3. Layer 1 checks trusted publisher proofs, hashes, signatures, and source metadata.
4. Layer 2 finds semantic candidate matches with Cohere or a lexical fallback.
5. Layer 3 computes supporting consensus metrics from stored votes.
6. Layer 4 persists a tamper-evident audit trail for each step.
7. The API returns a consistent verification result payload for the UI.
