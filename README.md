# OZC V89 AI Server

Render Web Service (Python):
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment: `OPENAI_API_KEY` (new secret key, never commit it), `OPENAI_MODEL=gpt-4.1-mini`
- `ALLOWED_ORIGINS`: comma-separated trusted HTTPS website origins. For a local `file://` V89 HTML page, the browser typically sends origin `null`; temporary testing can set `ALLOWED_ORIGINS=null`, but do not use this as a production security boundary. Host the V89 frontend on HTTPS and restrict to its exact origin for normal use.
- Health: `/health`; analysis URL for V89: `https://YOUR-RENDER-HOST.onrender.com/analyze`

Important: No authentication is built into this starter service. Do not expose publicly with an active billable key beyond a controlled test. Add authentication and rate limiting before production. The V89 client must not contain the API key.
