# NewsProof

NewsProof is an MVP web application for verifying the provenance of news content. It accepts article URLs, pasted text, and uploaded images, then runs a 4-layer verification pipeline and presents a clear verification label, proof summary, and audit trail.

## Stack

- Frontend: Next.js, React, TypeScript, Tailwind CSS
- Backend: FastAPI, Python 3.12
- Data: PostgreSQL, Redis, Alembic, local upload storage
- Verification support: Tesseract OCR, Cohere Embed/Rerank with graceful fallback

## Repository layout

- `frontend/` Next.js application
- `backend/` FastAPI API, migrations, tests, and verification services
- `docs/` architecture notes
- `extension/` Chrome extension for one-click verification previews

## Features

- URL verification with server-side extraction
- Pasted text verification
- Image upload verification with OCR
- Audio and video upload verification with graceful media preprocessing
- Trusted publisher management
- Verification history
- Append-only hash-chained audit trail
- Platform-issued cryptographic provenance manifests and version chains
- Chrome extension popup for quick current-tab verification

## API routes

- `POST /api/v1/verify/url`
- `POST /api/v1/verify/text`
- `POST /api/v1/verify/image`
- `POST /api/v1/verify/audio`
- `POST /api/v1/verify/video`
- `GET /api/v1/results/{job_id}`
- `GET /api/v1/history`
- `GET /api/v1/audit/{job_id}`
- `GET /api/v1/publishers`
- `POST /api/v1/publishers`
- `PUT /api/v1/publishers/{id}`
- `DELETE /api/v1/publishers/{id}`
- `GET /api/v1/crypto/issuers`
- `GET /api/v1/crypto/issuers/{id}`
- `POST /api/v1/crypto/issuers`
- `POST /api/v1/crypto/issuers/{id}/revoke`
- `POST /api/v1/crypto/article-snapshots`
- `POST /api/v1/crypto/evidence-bundles`
- `POST /api/v1/crypto/review-events`
- `POST /api/v1/crypto/manifests/sign`
- `POST /api/v1/crypto/manifests/verify`
- `GET /api/v1/crypto/manifests/{id}`
- `GET /api/v1/crypto/records/{object_type}/{record_id}/history`
- `GET /api/v1/crypto/records/{object_type}/{record_id}/compare`

## Environment variables

See `.env.example`.

- `DATABASE_URL` PostgreSQL connection string
- `REDIS_URL` Redis connection string
- `COHERE_API_KEY` optional Cohere API key
- `NEXT_PUBLIC_API_BASE_URL` frontend API base URL, for example `http://localhost:8000` locally or a temporary backend ngrok HTTPS URL
- `INTERNAL_API_BASE_URL` server-side frontend-to-backend URL inside Docker
- `BACKEND_CORS_ORIGINS_RAW` comma-separated allowed frontend origins
- `BACKEND_CORS_ORIGIN_REGEX` optional development-only CORS regex for temporary tunnel origins
- `UPLOAD_DIR` local storage directory for uploaded images
- `UPLOAD_MAX_BYTES` max image upload size in bytes
- `REQUEST_TIMEOUT_SECONDS` URL fetch timeout
- `PLATFORM_SIGNING_PRIVATE_KEY_B64` optional Ed25519 private key for the platform signer in base64 raw-byte format
- `PLATFORM_ISSUER_NAME` platform issuer name
- `TIMESTAMP_MODE` `internal` by default, or `rfc3161` for the production integration point
- `TSA_URL` optional RFC 3161 TSA endpoint URL

## Local development

### Docker Compose

1. Ensure Docker Desktop is running.
2. Review `.env` or copy from `.env.example`.
3. Start the stack:

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

The backend container runs `alembic upgrade head` before starting the API.

### Backend without Docker

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

### Frontend without Docker

```bash
cd frontend
npm install
npm run dev
```

When running the frontend outside Docker, put `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000` in `frontend/.env.local`. For temporary external testing through ngrok, see `LOCAL_TUNNEL_SETUP.md`.

## Tests

Backend:

```bash
cd backend
pytest
```

Frontend:

```bash
cd frontend
npm test
```

## Chrome extension

1. Start the NewsProof stack.
2. Open `chrome://extensions`.
3. Enable `Developer mode`.
4. Click `Load unpacked`.
5. Select the `extension/` folder.

The extension popup verifies the active tab URL and shows:

- verification label
- short reason
- link to the full report page

Default local targets:

- API: `http://127.0.0.1:8000/api/v1`
- Web app: `http://127.0.0.1:3000`

You can override these in the popup settings.

## Cryptographic provenance layer

NewsProof now includes a platform-issued cryptographic provenance subsystem. It does not decide whether a claim is true. Its job is to answer:

- who issued this record
- what exact version was recorded
- when it was recorded
- whether it has changed

Supported record types:

- `article_snapshot`
- `evidence_bundle`
- `review_event`

Provenance flow:

1. Build deterministic canonical JSON for the record.
2. Hash the canonical JSON bytes with SHA-256.
3. Sign the digest with a server-side Ed25519 key.
4. Attach issuer identity and timestamp token.
5. Store a new append-only provenance manifest.
6. Verify later by rebuilding canonical JSON, recomputing the digest, checking the signature, checking issuer status, checking the timestamp token, and validating the version chain.

Verification statuses:

- `valid`
- `hash_mismatch`
- `signature_invalid`
- `issuer_revoked`
- `timestamp_invalid`
- `version_chain_broken`

UI:

- Use `http://127.0.0.1:3000/provenance` for the provenance studio.
- It can create a signed article snapshot, verify the manifest, and display version history and changed fields.

## Verification decision rules

- `Verified` requires strong Layer 1 proof.
- `Verification Failed` is used when a trusted proof exists but does not match.
- `Not Verifiable` is used when no reliable provenance proof is found.
- `Timeout` is used when extraction or verification fails to complete in time.

Layers 2 and 3 add supporting evidence only and cannot independently produce `Verified`.

## Notes

- Cohere support degrades gracefully to a lexical overlap ranking when `COHERE_API_KEY` is not configured.
- OCR requires Tesseract; the backend Docker image installs it.
- Video verification tries keyframe OCR first and falls back to media metadata if no readable frame text is found.
- Audio verification currently uses metadata-based preprocessing until speech transcription is added.
- If `PLATFORM_SIGNING_PRIVATE_KEY_B64` is not set, the backend generates an ephemeral development signing key. That is acceptable for local development but not for production.
- Blockchain integration is still not included.
