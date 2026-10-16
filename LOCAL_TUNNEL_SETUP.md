# Local Tunnel Setup

Use this only for temporary local testing. Do not commit ngrok auth tokens, API keys, secrets, or generated public ngrok URLs.

## 1. Start the Backend

With Docker Compose from the repository root:

```bash
docker compose up --build backend
```

Or run the backend directly:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend should be available at:

```text
http://localhost:8000
```

## 2. Start the Frontend

If you run the frontend directly, create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Then start Next.js:

```bash
cd frontend
npm install
npm run dev
```

The frontend should be available at:

```text
http://localhost:3000
```

If you run with Docker Compose, keep the values in the repository-root `.env` because Compose injects that file into the containers.

## 3. Start ngrok for the Backend

In a new terminal:

```bash
ngrok http 8000
```

Copy the HTTPS forwarding URL. It will look like:

```text
https://BACKEND-NGROK-URL
```

Do not commit this generated URL.

## 4. Point the Frontend at the Backend Tunnel

For direct local frontend development, update `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=https://BACKEND-NGROK-URL
```

Restart `npm run dev` after changing this file. Next.js reads `NEXT_PUBLIC_*` values at startup/build time.

For Docker Compose, put the same value in the repository-root `.env` temporarily, then rebuild/restart the frontend container.

```bash
docker compose up --build -d frontend
```

The frontend Docker build receives `NEXT_PUBLIC_API_BASE_URL` as a build argument, because Next.js embeds public environment variables into the browser bundle.

## 5. Start ngrok for the Frontend

In another terminal:

```bash
ngrok http 3000
```

Share the frontend HTTPS forwarding URL with the external tester.

## CORS

The backend allows local frontend origins from `BACKEND_CORS_ORIGINS_RAW`, for example:

```env
BACKEND_CORS_ORIGINS_RAW=http://localhost:3000,http://127.0.0.1:3000
```

For temporary ngrok testing in development, the backend also accepts ngrok frontend origins through:

```env
BACKEND_CORS_ORIGIN_REGEX=^https://[a-zA-Z0-9-]+\.(ngrok-free\.app|ngrok\.io)$
```

This regex is disabled automatically when `APP_ENV=production`. For production, use explicit allowed origins instead of wildcard or broad tunnel patterns.

## Cleanup

When testing is done:

1. Stop both ngrok processes.
2. Restore `NEXT_PUBLIC_API_BASE_URL` to `http://localhost:8000`.
3. Remove any temporary public URLs from `.env`, `frontend/.env.local`, shell history snippets, or notes.
