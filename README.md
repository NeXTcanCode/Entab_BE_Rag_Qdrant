# NeXTCodeNavigator Render Backend

This directory is a standalone Render deployment package. It reads school and
file metadata from Qdrant, reads exact source files from the private GitHub
repository `likhilbalakrishnan/Websites`, and uses Ollama first with Groq as the
chat fallback.

## Read-only GitHub access

Create a fine-grained GitHub token restricted to the `Websites` repository with
only **Contents: Read-only** permission. Configure it as `GITHUB_TOKEN` in
Render. The backend only sends HTTP GET requests to GitHub's Contents and Git
Trees APIs. It contains no GitHub create, update, commit, or delete operations.

## Render deployment

1. Push the contents of this directory to a private GitHub repository.
2. In Render, create a Web Service from that repository.
3. Render will detect `render.yaml`. If configuring manually, use:

   ```text
   Build command: pip install -r requirements.txt
   Start command: uvicorn user_embed_question:app --host 0.0.0.0 --port $PORT
   Health check: /health
   ```

4. Add the secret environment variables shown in `.env.example` through the
   Render dashboard. Never commit `.env`.

## Source-path mapping

Qdrant payloads currently contain Mac paths such as:

```text
/Users/.../Websites-main/ABSB-Anand-Bhawan
```

The Render backend ignores that machine-specific prefix. It uses `school_name`
as the GitHub directory and retains Qdrant's relative file path:

```text
ABSB-Anand-Bhawan + src/Component/Header.js
→ ABSB-Anand-Bhawan/src/Component/Header.js
```

## Frontend and CORS

Localhost origins are enabled by default. After Netlify deployment, configure:

```text
ALLOWED_ORIGINS=https://next-doesnt-like-css.netlify.app
```

Set the Netlify frontend environment variable to the Render service URL:

```text
VITE_BACKEND_URL=https://your-service.onrender.com
```
