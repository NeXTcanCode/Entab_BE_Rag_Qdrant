#!/usr/bin/env python3
"""ABSB local RAG backend for Continue.

This server exposes an OpenAI-compatible API for Continue:
- /v1/models
- /v1/embeddings
- /v1/chat/completions

Important behavior:
- Direct source requests such as "show me header of ABSB" return the real
  source file content verbatim.
- Follow-up requests like "css", "code", or "both" reuse the original section
  and switch the returned source type.
- Qdrant is used for fuzzy retrieval. Exact section names are resolved from
  absb_embeddings.jsonl first, because "Header" should never retrieve "News".
"""

from __future__ import annotations

import base64
import json
import os
import re
import ssl
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import certifi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, SearchParams

APP_DIR = Path(__file__).resolve().parent
ENV_FILE = APP_DIR / ".env"
CATALOGUE_FILE = APP_DIR / "absb_embeddings.jsonl"

DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
DEFAULT_CHAT_MODEL = "qwen2.5-coder:1.5b"
DEFAULT_COLLECTION = "absb_school_code"
DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_GROQ_BASE = "https://api.groq.com/openai/v1"
DEFAULT_GITHUB_API_BASE = "https://api.github.com"

MAX_SOURCE_CHARS = 40_000
OUTPUT_MODES = {"code", "css", "both"}
OUT_OF_SCOPE_ANSWER = "I can only answer questions supported by the selected school codebase."
CODE_QUERY_TERMS = {
    "api", "class", "code", "component", "css", "dependency", "element",
    "file", "function", "header", "html", "import", "javascript", "jsx",
    "layout", "page", "prop", "react", "section", "source", "style",
}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(ENV_FILE)

EMBED_MODEL = os.environ.get("ABSB_EMBED_MODEL", DEFAULT_EMBED_MODEL)
CHAT_MODEL = os.environ.get("ABSB_CHAT_MODEL", DEFAULT_CHAT_MODEL)
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", DEFAULT_COLLECTION)
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", DEFAULT_OLLAMA_BASE)
OLLAMA_CHAT_TIMEOUT = float(os.environ.get("OLLAMA_CHAT_TIMEOUT", "30"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
GROQ_BASE = os.environ.get("GROQ_BASE", DEFAULT_GROQ_BASE)
GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "likhilbalakrishnan")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "Websites")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_API_BASE = os.environ.get("GITHUB_API_BASE", DEFAULT_GITHUB_API_BASE)
GITHUB_CACHE_SECONDS = int(os.environ.get("GITHUB_CACHE_SECONDS", "300"))
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

LOCAL_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
EXTRA_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
ALLOWED_ORIGINS = list(dict.fromkeys(LOCAL_ORIGINS + EXTRA_ORIGINS))

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env or environment")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
app = FastAPI(title="NeXTCodeNavigator Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    school_code: str = Field(default="ABSB")
    section_name: str | None = None
    output_mode: str | None = Field(default="code")
    top_k: int = Field(default=5, ge=1, le=20)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] | str
    content: str | list[Any] | None = ""


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="absb-rag")
    messages: list[ChatMessage]
    stream: bool = Field(default=False)
    temperature: float | None = None
    school_code: str = Field(default="ABSB")
    output_mode: str | None = Field(default="code")
    top_k: int = Field(default=5, ge=1, le=20)


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]


class SourceRequest(BaseModel):
    school_code: str = Field(..., min_length=1)
    source_file_path: str = Field(..., min_length=1)
    output_mode: Literal["code", "css", "both"] = "code"


class SourceChatRequest(SourceRequest):
    messages: list[ChatMessage]


def message_text(content: str | list[Any] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def load_catalogue_records() -> list[dict[str, Any]]:
    if not CATALOGUE_FILE.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in CATALOGUE_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        record.pop("embedding_vector", None)
        records.append(record)
    return records


CATALOGUE_RECORDS = load_catalogue_records()
_INDEXED_SCHOOL_CODES: set[str] | None = None
_INDEXED_SCHOOL_CODES_AT = 0.0
SCHOOL_CODE_CACHE_SECONDS = 30.0


def qdrant_payloads(
    payload_fields: list[str],
    scroll_filter: Filter | None = None,
) -> list[dict[str, Any]]:
    """Read matching payload metadata from the complete Qdrant collection."""
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=payload_fields,
            with_vectors=False,
        )
        payloads.extend(point.payload or {} for point in points)
        if offset is None:
            break
    return payloads


def source_ref_path(source_ref: Any) -> str:
    """Extract the relative file path from SCHOOL:path:lines:hash metadata."""
    parts = str(source_ref).split(":", 3)
    return parts[1].strip() if len(parts) == 4 else ""


def non_css_source_paths(payload: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for field_name in ("primary_files", "snippet_files"):
        value = payload.get(field_name, [])
        if isinstance(value, list):
            paths.update(str(path).strip() for path in value if str(path).strip())
    source_refs = payload.get("source_refs", [])
    if isinstance(source_refs, list):
        paths.update(path for path in map(source_ref_path, source_refs) if path)
    return {path for path in paths if not path.lower().endswith(".css")}


def local_school_codes() -> set[str]:
    return {str(record.get("school_code", "")).upper() for record in CATALOGUE_RECORDS if record.get("school_code")}


def catalogue_school_codes() -> set[str]:
    """Return school codes that actually exist in Qdrant.

    The pipeline intentionally reuses ``absb_embeddings.jsonl`` for each school,
    so that local file contains only the most recently processed school. It must
    not be treated as the registry for the complete Qdrant collection.
    """
    global _INDEXED_SCHOOL_CODES, _INDEXED_SCHOOL_CODES_AT

    now = time.monotonic()
    if (
        _INDEXED_SCHOOL_CODES is not None
        and now - _INDEXED_SCHOOL_CODES_AT < SCHOOL_CODE_CACHE_SECONDS
    ):
        return set(_INDEXED_SCHOOL_CODES)

    codes: set[str] = set()
    offset = None
    try:
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=["school_code"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                code = str(payload.get("school_code", "")).strip().upper()
                if code:
                    codes.add(code)
            if offset is None:
                break
    except Exception:
        # Keep the API usable during a temporary Qdrant outage, but only expose
        # the local school's data in that degraded mode.
        codes = local_school_codes()

    _INDEXED_SCHOOL_CODES = codes
    _INDEXED_SCHOOL_CODES_AT = now
    return set(codes)


def known_school_code(question: str, default: str) -> str:
    upper_question = question.upper()
    codes = catalogue_school_codes()
    for code in sorted(codes, key=len, reverse=True):
        if code and re.search(rf"\b{re.escape(code)}\b", upper_question):
            return code
    return default.upper()


def requested_school_code(question: str) -> str | None:
    """Return an explicit school code from the user's wording, even if unknown.

    Examples:
    - "show me header of ABSB" -> ABSB
    - "show me header of jhgjhgjgj" -> JHGJHGJGJ

    This prevents unknown codes from silently falling back to ABSB.
    """
    known_sections = {
        normalize_token(str(record.get("section_name", "")))
        for record in CATALOGUE_RECORDS
        if record.get("section_name")
    }
    ignored = known_sections | {
        "code",
        "css",
        "both",
        "header",
        "footer",
        "section",
        "component",
        "source",
        "school",
    }

    patterns = [
        r"\b(?:of|for|from)\s+([A-Za-z][A-Za-z0-9_-]{2,})\b",
        r"\bschool(?:\s+code)?\s*[:=-]?\s*([A-Za-z][A-Za-z0-9_-]{2,})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, question, flags=re.I):
            candidate = match.group(1).strip().upper()
            if normalize_token(candidate) not in ignored:
                return candidate
    return None


def resolve_school_code(question: str, default: str) -> tuple[str, str | None]:
    """Return (school_code_to_use, unknown_requested_code)."""
    codes = catalogue_school_codes()
    explicit_code = requested_school_code(question)
    if explicit_code:
        if explicit_code in codes:
            return explicit_code, None
        return default.upper(), explicit_code

    resolved = known_school_code(question, default)
    if resolved in codes:
        return resolved, None
    return default.upper(), resolved


def unknown_school_answer(requested_code: str, available_codes: set[str] | None = None) -> str:
    codes = sorted(catalogue_school_codes() if available_codes is None else available_codes)
    available = ", ".join(codes) if codes else "none"
    return (
        f"I do not have vector/source data for school code `{requested_code.upper()}`.\n\n"
        f"Available indexed school codes: `{available}`.\n\n"
        "Please check the school code, then ask again. Example:\n\n"
        "`show me header of ABSB`\n\n"
        "No code or CSS was returned because using a fallback school would give the wrong project files."
    )


def find_explicit_section(question: str, school_code: str = "ABSB") -> str | None:
    normalized_question = normalize_token(question)
    if not normalized_question:
        return None

    sections = {
        str(record.get("section_name", ""))
        for record in CATALOGUE_RECORDS
        if str(record.get("school_code", "")).upper() == school_code.upper()
    }
    for section in sorted(sections, key=len, reverse=True):
        normalized_section = normalize_token(section)
        if normalized_section and normalized_section in normalized_question:
            return section
    return None


def embed_text(text: str) -> list[float]:
    payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_BASE.rstrip("/") + "/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    vector = data.get("embedding")
    if not isinstance(vector, list):
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            vector = embeddings[0]

    if not isinstance(vector, list):
        raise RuntimeError(f"Embedding response missing vector: {data}")
    return vector


def build_filter(school_code: str, section_name: str | None = None) -> Filter:
    must = [FieldCondition(key="school_code", match=MatchValue(value=school_code.upper()))]
    if section_name:
        must.append(FieldCondition(key="section_name", match=MatchValue(value=section_name)))
    return Filter(must=must)


def safe_source_path(source_root: str, relative_path: str) -> Path | None:
    if not source_root or not relative_path:
        return None

    root = Path(source_root).expanduser().resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def read_text_file(path: Path) -> str:
    content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) > MAX_SOURCE_CHARS:
        return content[:MAX_SOURCE_CHARS] + "\n\n/* truncated by backend */"
    return content


_GITHUB_JSON_CACHE: dict[str, tuple[float, Any]] = {}
_GITHUB_TEXT_CACHE: dict[str, tuple[float, str]] = {}


def github_configured() -> bool:
    return bool(GITHUB_OWNER and GITHUB_REPO and GITHUB_BRANCH and GITHUB_TOKEN)


def github_headers() -> dict[str, str]:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is not configured on the backend")
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "NeXTCodeNavigator/1.0 (read-only source client)",
    }


def normalized_repo_path(path: str) -> str:
    value = str(path).strip().replace("\\", "/").lstrip("/")
    parts = PurePosixPath(value).parts
    if not value or ".." in parts or "." in parts:
        raise ValueError(f"Unsafe GitHub repository path: {path}")
    return "/".join(parts)


def github_api_json(api_path: str, *, use_cache: bool = True) -> Any:
    path = "/" + api_path.lstrip("/")
    now = time.monotonic()
    cached = _GITHUB_JSON_CACHE.get(path)
    if use_cache and cached and now - cached[0] < GITHUB_CACHE_SECONDS:
        return cached[1]

    request = urllib.request.Request(
        GITHUB_API_BASE.rstrip("/") + path,
        headers=github_headers(),
        method="GET",
    )
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"GitHub read failed with HTTP {exc.code}: {detail}") from exc

    if use_cache:
        _GITHUB_JSON_CACHE[path] = (now, data)
    return data


def github_file_text(repo_path: str) -> str:
    path = normalized_repo_path(repo_path)
    now = time.monotonic()
    cached = _GITHUB_TEXT_CACHE.get(path)
    if cached and now - cached[0] < GITHUB_CACHE_SECONDS:
        return cached[1]

    encoded_path = urllib.parse.quote(path, safe="/")
    encoded_ref = urllib.parse.quote(GITHUB_BRANCH, safe="")
    data = github_api_json(
        f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{encoded_path}?ref={encoded_ref}"
    )
    if not isinstance(data, dict) or data.get("type") != "file":
        raise FileNotFoundError(f"GitHub source file was not found: {path}")

    encoded_content = data.get("content")
    if not encoded_content and data.get("git_url"):
        blob_url = str(data["git_url"])
        api_prefix = GITHUB_API_BASE.rstrip("/")
        if not blob_url.startswith(api_prefix + "/"):
            raise RuntimeError("GitHub returned an unexpected blob URL")
        data = github_api_json(blob_url[len(api_prefix):])
        encoded_content = data.get("content") if isinstance(data, dict) else None
    if not isinstance(encoded_content, str):
        raise RuntimeError(f"GitHub response did not contain file content: {path}")

    content = base64.b64decode(encoded_content).decode("utf-8", errors="replace")
    _GITHUB_TEXT_CACHE[path] = (now, content)
    return content


def source_folder(payload: dict[str, Any]) -> str:
    school_code = str(payload.get("school_code") or "").strip()
    school_name = str(payload.get("school_name") or "").strip()
    if school_name:
        # GitHub stores each project under ``{school_code}-{school_name}``.
        # Existing Qdrant records may contain either the complete directory
        # name (for example ``ABSB-Anand-Bhawan``) or only the readable name.
        folder = school_name
        if school_code and not school_name.upper().startswith(
            f"{school_code.upper()}-"
        ):
            folder = f"{school_code}-{school_name}"
        return normalized_repo_path(folder)

    # source_root contains a local Mac path in older Qdrant records. Keep it
    # only as a compatibility fallback; it must never override school_name.
    source_root = str(payload.get("source_root") or "").strip()
    if source_root:
        return normalized_repo_path(PurePosixPath(source_root.replace("\\", "/")).name)
    raise RuntimeError("Qdrant payload is missing school_name/source_root")


def github_source_text(
    payload: dict[str, Any],
    relative_path: str,
    *,
    truncate: bool = True,
) -> str:
    relative = normalized_repo_path(relative_path)
    repo_path = f"{source_folder(payload)}/{relative}"
    content = github_file_text(repo_path)
    if truncate and len(content) > MAX_SOURCE_CHARS:
        return content[:MAX_SOURCE_CHARS] + "\n\n/* truncated by backend */"
    return content


def github_school_files(payload: dict[str, Any]) -> list[str]:
    folder = source_folder(payload)
    encoded_ref = urllib.parse.quote(GITHUB_BRANCH, safe="")
    branch_data = github_api_json(
        f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/branches/{encoded_ref}"
    )
    commit_data = branch_data.get("commit", {}) if isinstance(branch_data, dict) else {}
    commit_details = commit_data.get("commit", {}) if isinstance(commit_data, dict) else {}
    tree_details = commit_details.get("tree", {}) if isinstance(commit_details, dict) else {}
    root_sha = str(tree_details.get("sha") or "")
    if not root_sha:
        raise RuntimeError(f"GitHub branch tree was not found: {GITHUB_BRANCH}")

    root_data = github_api_json(
        f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/git/trees/{root_sha}"
    )
    root_tree = root_data.get("tree", []) if isinstance(root_data, dict) else []
    school_entry = next(
        (
            item
            for item in root_tree
            if item.get("type") == "tree" and item.get("path") == folder
        ),
        None,
    )
    if not school_entry:
        raise RuntimeError(f"GitHub school directory was not found: {folder}")
    sha = str(school_entry.get("sha") or "")
    if not sha:
        raise RuntimeError(f"GitHub school directory is missing its tree SHA: {folder}")

    tree_data = github_api_json(
        f"/repos/{GITHUB_OWNER}/{GITHUB_REPO}/git/trees/{sha}?recursive=1"
    )
    if not isinstance(tree_data, dict) or not isinstance(tree_data.get("tree"), list):
        raise RuntimeError(f"GitHub returned an invalid tree for {folder}")
    if tree_data.get("truncated"):
        raise RuntimeError(f"GitHub tree is truncated for {folder}")

    ignored_parts = {"node_modules", "dist", "build", ".git"}
    return sorted(
        normalized_repo_path(str(item.get("path", "")))
        for item in tree_data["tree"]
        if item.get("type") == "blob"
        and item.get("path")
        and not ignored_parts.intersection(PurePosixPath(str(item["path"])).parts)
    )


def source_paths_from_payload(payload: dict[str, Any], output_mode: str = "both") -> list[str]:
    primary_files = list(payload.get("primary_files") or [])
    style_files = list(payload.get("style_files") or [])
    snippet_files = list(payload.get("snippet_files") or [])

    if output_mode == "code":
        paths = primary_files + snippet_files
    elif output_mode == "css":
        paths = style_files
    else:
        paths = primary_files + style_files + snippet_files

    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def load_source_files(payload: dict[str, Any], output_mode: str = "both") -> list[dict[str, str]]:
    files: list[dict[str, str]] = []

    for relative_path in source_paths_from_payload(payload, output_mode):
        try:
            content = github_source_text(
                payload,
                relative_path,
                truncate=not str(relative_path).lower().endswith(".css"),
            )
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        files.append({"path": relative_path, "content": content})
    return files


def context_from_payload(payload: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    return {
        "score": score,
        "school_code": payload.get("school_code"),
        "school_name": payload.get("school_name"),
        "source_root": payload.get("source_root", ""),
        "bundle_id": payload.get("bundle_id"),
        "section_name": payload.get("section_name"),
        "output_mode": payload.get("output_mode"),
        "primary_files": payload.get("primary_files", []),
        "style_files": payload.get("style_files", []),
        "snippet_files": payload.get("snippet_files", []),
        "dependencies": payload.get("dependencies", []),
        "source_refs": payload.get("source_refs", []),
        "embedding_text": payload.get("embedding_text", ""),
        "source_files": load_source_files(payload, "both"),
    }


def local_section_contexts(school_code: str, section_name: str) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for record in CATALOGUE_RECORDS:
        if str(record.get("school_code", "")).upper() != school_code.upper():
            continue
        if str(record.get("section_name", "")) != section_name:
            continue
        contexts.append(context_from_payload(record, score=1.0))
    return contexts


def search_qdrant(query_vector: list[float], school_code: str, section_name: str | None, top_k: int) -> list[dict[str, Any]]:
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=build_filter(school_code, section_name),
        limit=top_k,
        search_params=SearchParams(hnsw_ef=64),
        with_payload=True,
    )

    points = getattr(response, "points", response)
    items: list[dict[str, Any]] = []
    for point in points:
        payload = point.payload or {}
        items.append(context_from_payload(payload, score=getattr(point, "score", None)))
    return items


def extract_class_names(context: dict[str, Any]) -> set[str]:
    class_names: set[str] = set()
    for source_file in context.get("source_files", []):
        if Path(source_file.get("path", "")).suffix.lower() == ".css":
            continue
        content = source_file.get("content", "")
        for match in re.findall(r"className\s*=\s*[\"']([^\"']+)[\"']", content):
            for class_name in re.split(r"\s+", match.strip()):
                if class_name and re.match(r"^[A-Za-z_][A-Za-z0-9_-]*$", class_name):
                    class_names.add(class_name)
        for match in re.findall(r"\bstyles\.([A-Za-z_][A-Za-z0-9_-]*)", content):
            class_names.add(match)
        for match in re.findall(r"\bstyles\[['\"]([A-Za-z_][A-Za-z0-9_-]*)['\"]\]", content):
            class_names.add(match)
    return class_names


def matching_brace(text: str, opening: int) -> int | None:
    """Find a CSS block's closing brace while respecting strings/comments."""
    depth = 0
    quote: str | None = None
    escaped = False
    in_comment = False
    index = opening
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_comment:
            if char == "*" and next_char == "/":
                in_comment = False
                index += 2
                continue
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char == "/" and next_char == "*":
            in_comment = True
            index += 2
            continue
        elif char in {"'", '"'}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def css_blocks(css: str) -> list[tuple[str, str]]:
    """Return top-level ``(header, body)`` CSS blocks."""
    blocks: list[tuple[str, str]] = []
    start = 0
    index = 0
    quote: str | None = None
    escaped = False
    in_comment = False
    while index < len(css):
        char = css[index]
        next_char = css[index + 1] if index + 1 < len(css) else ""

        if in_comment:
            if char == "*" and next_char == "/":
                in_comment = False
                index += 2
                continue
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char == "/" and next_char == "*":
            in_comment = True
            index += 2
            continue
        elif char in {"'", '"'}:
            quote = char
        elif char == ";":
            # Skip blockless at-rules such as @import.
            start = index + 1
        elif char == "{":
            closing = matching_brace(css, index)
            if closing is None:
                break
            header = css[start:index].strip()
            if header:
                blocks.append((header, css[index + 1 : closing]))
            index = closing
            start = closing + 1
        index += 1
    return blocks


def selector_uses_class(selector: str, class_names: set[str]) -> bool:
    return any(
        re.search(rf"(?<![A-Za-z0-9_-])\.{re.escape(name)}(?![A-Za-z0-9_-])", selector)
        for name in class_names
    )


def filter_css_blocks(css: str, class_names: set[str]) -> str:
    """Select component rules and preserve their enclosing conditional blocks."""
    selected: list[str] = []
    for header, body in css_blocks(css):
        normalized_header = re.sub(r"^(?:/\*.*?\*/\s*)+", "", header, flags=re.S).strip()
        lower_header = normalized_header.lower()

        if lower_header.startswith("@"):
            if re.match(r"@(?:-[a-z]+-)?keyframes\b", lower_header):
                continue
            nested = filter_css_blocks(body, class_names)
            if nested:
                selected.append(f"{normalized_header} {{\n{nested}\n}}")
        elif selector_uses_class(normalized_header, class_names):
            selected.append(f"{normalized_header} {{\n{body.strip()}\n}}")
    return "\n\n".join(dict.fromkeys(selected))


def related_keyframes(css: str, selected_css: str) -> str:
    """Include keyframes whose names are referenced by selected declarations."""
    blocks: list[str] = []
    for header, body in css_blocks(css):
        match = re.match(
            r"@(?:-[A-Za-z]+-)?keyframes\s+([^\s{]+)",
            header.strip(),
            flags=re.I,
        )
        if not match:
            continue
        name = match.group(1).strip("'\"")
        if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])", selected_css):
            blocks.append(f"{header.strip()} {{\n{body.strip()}\n}}")
    return "\n\n".join(dict.fromkeys(blocks))


def extract_css_blocks(css: str, class_names: set[str]) -> str:
    if not class_names:
        return ""
    selected = filter_css_blocks(css, class_names)
    keyframes = related_keyframes(css, selected)
    return "\n\n".join(part for part in (selected, keyframes) if part)


def select_relevant_css_sources(
    context: dict[str, Any], css_files: list[dict[str, str]]
) -> list[dict[str, str]]:
    class_names = extract_class_names(context)
    if not class_names:
        # Without any statically discoverable class names, explicit style_files
        # are the only reliable association available for this bundle.
        return css_files

    selected: list[dict[str, str]] = []
    for source_file in css_files:
        content = extract_css_blocks(source_file.get("content", ""), class_names)
        if content:
            selected.append({"path": source_file.get("path", ""), "content": content})
    return selected


def discover_style_sources(context: dict[str, Any]) -> list[dict[str, str]]:
    class_names = extract_class_names(context)
    if not class_names:
        return []

    discovered: list[dict[str, str]] = []
    try:
        css_paths = [
            path for path in github_school_files(context) if path.lower().endswith(".css")
        ]
    except (RuntimeError, ValueError):
        return []

    for relative_path in css_paths:
        try:
            css = github_source_text(context, relative_path, truncate=False)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue

        snippets = extract_css_blocks(css, class_names)
        if not snippets:
            continue
        discovered.append({"path": relative_path, "content": snippets})
    return discovered


def exact_source_context(
    school_code: str,
    source_file_path: str,
    output_mode: str,
) -> dict[str, Any]:
    """Load an exact Qdrant-listed source file and its relevant project CSS."""
    payload_fields = [
        "school_code",
        "school_name",
        "source_root",
        "section_name",
        "bundle_id",
        "output_mode",
        "primary_files",
        "style_files",
        "snippet_files",
        "source_refs",
    ]
    payloads = qdrant_payloads(
        payload_fields,
        Filter(
            must=[
                FieldCondition(
                    key="school_code",
                    match=MatchValue(value=school_code),
                )
            ]
        ),
    )
    matches = [
        payload
        for payload in payloads
        if source_file_path in non_css_source_paths(payload)
    ]
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"{source_file_path} is not indexed for school {school_code}",
        )

    selected_payload = next(
        (payload for payload in matches if payload.get("source_root")),
        matches[0],
    )
    source_root = str(selected_payload.get("source_root") or "")
    try:
        code_content = github_source_text(selected_payload, source_file_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=404,
            detail=f"The GitHub source file is unavailable: {source_file_path}: {exc}",
        ) from exc

    code_file = {
        "path": source_file_path,
        "content": code_content,
    }
    context: dict[str, Any] = {
        "school_code": school_code,
        "school_name": selected_payload.get("school_name", ""),
        "source_root": source_root,
        "section_name": selected_payload.get("section_name", ""),
        "bundle_id": selected_payload.get("bundle_id", ""),
        "primary_files": [source_file_path],
        "style_files": [],
        "snippet_files": [],
        "source_refs": selected_payload.get("source_refs", []),
        "source_files": [code_file],
    }

    section_names = {
        str(payload.get("section_name", "")) for payload in matches
    }
    explicit_style_paths: list[str] = []
    for payload in payloads:
        if str(payload.get("section_name", "")) not in section_names:
            continue
        for path in payload.get("style_files") or []:
            path = str(path).strip()
            if path and path not in explicit_style_paths:
                explicit_style_paths.append(path)

    explicit_css: list[dict[str, str]] = []
    for relative_path in explicit_style_paths:
        try:
            css_content = github_source_text(
                selected_payload,
                relative_path,
                truncate=False,
            )
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        explicit_css.append(
            {
                "path": relative_path,
                "content": css_content,
            }
        )

    selected_css = select_relevant_css_sources(context, explicit_css)
    discovered_css = discover_style_sources(context)
    css_files: list[dict[str, str]] = []
    seen_css: set[str] = set()
    for source_file in selected_css + discovered_css:
        path = str(source_file.get("path", ""))
        content = str(source_file.get("content", ""))
        if not path or not content or path in seen_css:
            continue
        seen_css.add(path)
        css_files.append({"path": path, "content": content})

    context["style_files"] = [source_file["path"] for source_file in css_files]
    if output_mode == "code":
        context["source_files"] = [code_file]
    elif output_mode == "css":
        context["source_files"] = css_files
    else:
        context["source_files"] = [code_file] + css_files
    return context


def infer_output_selection(message: str, configured_mode: str | None = None) -> str:
    normalized = normalize_token(message)
    configured = (configured_mode or "code").lower()
    if configured in OUTPUT_MODES and normalized not in {"css", "code", "both"}:
        default = configured
    else:
        default = "code"

    if normalized in {"css", "style", "styles", "onlycss", "cssonly"}:
        return "css"
    if normalized in {"code", "js", "jsx", "component", "onlycode", "codeonly"}:
        return "code"
    if normalized in {"both", "codecss", "csscode", "codeandcss", "cssandcode"}:
        return "both"

    lower = message.lower()
    if re.search(r"\b(css|style|styles|stylesheet)\b", lower):
        if re.search(r"\b(code|jsx|js|component)\b", lower):
            return "both"
        return "css"
    if re.search(r"\b(both|code\s+and\s+css|css\s+and\s+code)\b", lower):
        return "both"
    return default


def apply_output_selection(contexts: list[dict[str, Any]], output_mode: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for context in contexts:
        item = dict(context)
        source_files = list(context.get("source_files") or [])

        if output_mode == "code":
            item["source_files"] = [
                source_file
                for source_file in source_files
                if Path(source_file.get("path", "")).suffix.lower() != ".css"
            ]
        elif output_mode == "css":
            css_files = [
                source_file
                for source_file in source_files
                if Path(source_file.get("path", "")).suffix.lower() == ".css"
            ]
            selected_css = select_relevant_css_sources(context, css_files)
            item["source_files"] = selected_css or discover_style_sources(context)
        else:
            code_files = [
                source_file
                for source_file in source_files
                if Path(source_file.get("path", "")).suffix.lower() != ".css"
            ]
            css_files = [
                source_file
                for source_file in source_files
                if Path(source_file.get("path", "")).suffix.lower() == ".css"
            ]
            selected_css = select_relevant_css_sources(context, css_files)
            item["source_files"] = code_files + (
                selected_css or discover_style_sources(context)
            )

        selected.append(item)
    return selected


def retrieve_contexts(question: str, school_code: str, output_mode: str, top_k: int = 5) -> tuple[str | None, list[dict[str, Any]]]:
    school_code = school_code.upper()
    section_name = find_explicit_section(question, school_code)

    if section_name:
        local_contexts = local_section_contexts(school_code, section_name)
        if local_contexts:
            return section_name, apply_output_selection(local_contexts, output_mode)

    query_vector = embed_text(question)
    fuzzy_contexts = search_qdrant(query_vector, school_code, section_name, top_k)
    if not fuzzy_contexts:
        return section_name, []

    best_section = str(fuzzy_contexts[0].get("section_name") or "")
    if best_section:
        strict_contexts = search_qdrant(query_vector, school_code, best_section, top_k)
        return best_section, apply_output_selection(strict_contexts or fuzzy_contexts, output_mode)
    return section_name, apply_output_selection(fuzzy_contexts, output_mode)


def language_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".js": "jsx",
        ".jsx": "jsx",
        ".ts": "tsx",
        ".tsx": "tsx",
        ".css": "css",
        ".html": "html",
        ".json": "json",
        ".md": "markdown",
    }.get(suffix, "text")


def format_source_answer(school_code: str, section_name: str | None, output_mode: str, contexts: list[dict[str, Any]]) -> str:
    files: list[tuple[str, str]] = []
    seen: set[str] = set()
    for context in contexts:
        for source_file in context.get("source_files", []):
            path = source_file.get("path", "")
            content = source_file.get("content", "")
            if not path or not content or path in seen:
                continue
            seen.add(path)
            files.append((path, content))

    section_label = section_name or "matched section"
    if not files:
        return f"No {output_mode} source file was found for {school_code} {section_label}."

    parts = [f"{school_code.upper()} {section_label} {output_mode} source:"]
    for path, content in files:
        safe_content = content.replace("```", "`\u200b``")
        parts.append(f"\nFile: `{path}`\n\n```{language_for_path(path)}\n{safe_content}\n```")
    return "\n".join(parts)


def should_return_source_directly(user_messages: list[str], latest_question: str) -> bool:
    if len(user_messages) <= 1:
        return True
    normalized = normalize_token(latest_question)
    if normalized in {"css", "style", "styles", "code", "js", "jsx", "both", "codecss", "csscode"}:
        return True
    lower = latest_question.lower()
    return bool(re.search(r"\b(show|give|return|send|provide|get)\b.*\b(code|css|source|header|footer|section|component)\b", lower))


def context_for_llm(contexts: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        source_blocks: list[str] = []
        for source_file in item.get("source_files", []):
            path = source_file.get("path", "")
            content = source_file.get("content", "")
            source_blocks.append(f"File: {path}\n```{language_for_path(path)}\n{content}\n```")
        blocks.append(
            f"Context {idx}\n"
            f"School: {item.get('school_code', '')}\n"
            f"Section: {item.get('section_name', '')}\n"
            f"Primary Files: {', '.join(item.get('primary_files', [])) or 'none'}\n"
            f"Style Files: {', '.join(item.get('style_files', [])) or 'none'}\n"
            f"Source:\n{chr(10).join(source_blocks) or 'none'}"
        )
    return "\n\n".join(blocks)


def is_context_question(question: str, contexts: list[dict[str, Any]]) -> bool:
    """Reject unrelated prompts before they can consume a hosted LLM token."""
    words = set(re.findall(r"[a-z][a-z0-9_-]{2,}", question.lower()))
    if words & CODE_QUERY_TERMS:
        return True

    identifiers: set[str] = set()
    for context in contexts:
        metadata = [
            context.get("school_code", ""),
            context.get("school_name", ""),
            context.get("section_name", ""),
            *context.get("primary_files", []),
            *context.get("style_files", []),
            *context.get("snippet_files", []),
            *context.get("dependencies", []),
        ]
        identifiers.update(
            re.findall(r"[a-z][a-z0-9_-]{2,}", " ".join(map(str, metadata)).lower())
        )
    if words & identifiers:
        return True

    # Permit narrow conversational follow-ups after a codebase answer.
    return bool(re.fullmatch(
        r"\s*(why|how|explain|continue|more|show more|what does (it|this) do)\??\s*",
        question,
        flags=re.IGNORECASE,
    ))


def grounded_chat_messages(
    question: str,
    contexts: list[dict[str, Any]],
    conversation: list[ChatMessage] | None = None,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are a frontend-code assistant. The selected codebase can contain "
        "JS, JSX, HTML, CSS, and JSON. Never invent Python files. "
        "Treat the retrieved files as persistent context for this conversation. "
        "Answer only from that source. If the answer is not present, say what is missing."
    )
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Retrieved source context:\n{context_for_llm(contexts)}"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if conversation:
        history = list(conversation[-6:])
        if (
            history
            and history[-1].role == "user"
            and message_text(history[-1].content).strip() == question.strip()
        ):
            history = history[:-1]
        for message in history:
            if message.role in {"user", "assistant"}:
                content = message_text(message.content)
                if content:
                    messages.append({"role": message.role, "content": content})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def chat_with_ollama(question: str, contexts: list[dict[str, Any]], conversation: list[ChatMessage] | None = None) -> str:
    messages = grounded_chat_messages(question, contexts, conversation)

    payload = json.dumps({"model": CHAT_MODEL, "messages": messages, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_BASE.rstrip("/") + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=OLLAMA_CHAT_TIMEOUT) as response:
        data = json.loads(response.read().decode("utf-8"))

    message = data.get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RuntimeError(f"Chat response missing content: {data}")
    return content


def chat_with_groq(question: str, contexts: list[dict[str, Any]], conversation: list[ChatMessage] | None = None) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured on the backend")

    messages = grounded_chat_messages(question, contexts, conversation)
    payload = json.dumps(
        {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        GROQ_BASE.rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "NeXTCodeNavigator/1.0 (Groq API client)",
        },
        method="POST",
    )
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(
            request,
            timeout=120,
            context=ssl_context,
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Groq returned HTTP {exc.code}: {detail}") from exc

    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Groq response missing assistant content")
    return content


def chat_with_fallback(
    question: str,
    contexts: list[dict[str, Any]],
    conversation: list[ChatMessage] | None = None,
) -> tuple[str, str]:
    try:
        return chat_with_ollama(question, contexts, conversation), "ollama"
    except Exception as ollama_error:
        print(f"Ollama chat unavailable; falling back to Groq: {ollama_error!r}")
        return chat_with_groq(question, contexts, conversation), "groq"


def openai_chat_response(answer: str, model: str = "absb-rag") -> dict[str, Any]:
    now = int(time.time())
    return {
        "id": f"chatcmpl-absb-{now}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def openai_streaming_response(answer: str, model: str = "absb-rag") -> StreamingResponse:
    now = int(time.time())
    response_id = f"chatcmpl-absb-{now}"

    def events():
        first = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n"

        step = 1200
        for start in range(0, len(answer), step):
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": now,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": answer[start:start + step]}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        final = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "embedding_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "groq_model": GROQ_MODEL,
        "groq_fallback_configured": str(bool(GROQ_API_KEY)).lower(),
        "github_source": f"{GITHUB_OWNER}/{GITHUB_REPO}@{GITHUB_BRANCH}",
        "github_read_configured": str(github_configured()).lower(),
    }


@app.get("/inventory/schools")
def inventory_schools() -> dict[str, Any]:
    """Return every school stored in Qdrant, sorted by school code."""
    try:
        payloads = qdrant_payloads(["school_code", "school_name"])
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load the school inventory from Qdrant: {exc}",
        ) from exc

    schools_by_code: dict[str, str] = {}
    for payload in payloads:
        code = str(payload.get("school_code", "")).strip().upper()
        if not code:
            continue
        name = str(payload.get("school_name", "")).strip()
        if code not in schools_by_code or (name and not schools_by_code[code]):
            schools_by_code[code] = name

    schools = [
        {"school_code": code, "school_name": schools_by_code[code]}
        for code in sorted(schools_by_code)
    ]
    return {"schools": schools, "count": len(schools)}


@app.get("/inventory/schools/{school_code}/files")
def inventory_school_files(school_code: str) -> dict[str, Any]:
    """Return exact non-CSS source paths stored for one school."""
    code = school_code.strip().upper()
    if not code:
        raise HTTPException(status_code=400, detail="school_code is required")

    try:
        payloads = qdrant_payloads(
            [
                "school_code",
                "section_name",
                "primary_files",
                "style_files",
                "snippet_files",
                "source_refs",
            ],
            Filter(
                must=[
                    FieldCondition(
                        key="school_code",
                        match=MatchValue(value=code),
                    )
                ]
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load source files for {code} from Qdrant: {exc}",
        ) from exc

    if not payloads:
        raise HTTPException(status_code=404, detail=f"School {code} is not present in Qdrant")

    section_files: dict[str, set[str]] = {}
    section_styles: dict[str, set[str]] = {}
    for payload in payloads:
        section = str(payload.get("section_name", "")).strip()
        section_files.setdefault(section, set()).update(non_css_source_paths(payload))
        styles = payload.get("style_files", [])
        if isinstance(styles, list):
            section_styles.setdefault(section, set()).update(
                str(path).strip() for path in styles if str(path).strip()
            )

    styles_by_file: dict[str, set[str]] = {}
    for section, source_paths in section_files.items():
        styles = section_styles.get(section, set())
        for path in source_paths:
            styles_by_file.setdefault(path, set()).update(styles)

    files = [
        {
            "path": path,
            "filename": PurePosixPath(path).name,
            "style_files": sorted(styles_by_file[path]),
            # Project-wide styles can be associated through selectors/imports
            # even when this Qdrant bundle's style_files field is empty.
            "has_css": None,
        }
        for path in sorted(styles_by_file, key=lambda value: value.lower())
    ]
    return {"school_code": code, "files": files, "count": len(files)}


@app.post("/source")
def source(req: SourceRequest) -> dict[str, Any]:
    code = req.school_code.strip().upper()
    relative_path = req.source_file_path.strip()
    try:
        context = exact_source_context(code, relative_path, req.output_mode)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load source for {code}: {exc}",
        ) from exc

    contexts = [context]
    return {
        "school_code": code,
        "source_file_path": relative_path,
        "output_mode": req.output_mode,
        "answer": format_source_answer(
            code,
            str(context.get("section_name") or relative_path),
            req.output_mode,
            contexts,
        ),
        "contexts": contexts,
        "has_css": bool(context.get("style_files")),
    }


@app.post("/source/chat")
def source_chat(req: SourceChatRequest) -> dict[str, Any]:
    user_messages = [
        message_text(message.content)
        for message in req.messages
        if message.role == "user" and message_text(message.content).strip()
    ]
    if not user_messages:
        raise HTTPException(status_code=400, detail="At least one user message is required")

    code = req.school_code.strip().upper()
    relative_path = req.source_file_path.strip()
    try:
        context = exact_source_context(code, relative_path, req.output_mode)
        if not is_context_question(user_messages[-1], [context]):
            answer = OUT_OF_SCOPE_ANSWER
            provider = "local"
        else:
            answer, provider = chat_with_fallback(user_messages[-1], [context], req.messages)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not answer from the selected source: {exc}",
        ) from exc

    return {
        "school_code": code,
        "source_file_path": relative_path,
        "output_mode": req.output_mode,
        "answer": answer,
        "contexts": [context],
        "provider": provider,
    }


@app.get("/v1/models")
def v1_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "absb-rag",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/embeddings")
def v1_embeddings(req: EmbeddingRequest) -> dict[str, Any]:
    inputs = req.input if isinstance(req.input, list) else [req.input]
    data = []
    for idx, text in enumerate(inputs):
        data.append({"object": "embedding", "index": idx, "embedding": embed_text(text)})
    return {"object": "list", "model": req.model or EMBED_MODEL, "data": data}


@app.post("/v1/chat/completions")
def v1_chat_completions(req: ChatCompletionRequest):
    try:
        user_messages = [message_text(message.content) for message in req.messages if message.role == "user"]
        user_messages = [message for message in user_messages if message.strip()]
        if not user_messages:
            answer = "Please ask a question about an ABSB section."
            return openai_streaming_response(answer, req.model) if req.stream else JSONResponse(openai_chat_response(answer, req.model))

        latest_question = user_messages[-1]
        retrieval_anchor = user_messages[0] if len(user_messages) > 1 else latest_question
        school_code, unknown_code = resolve_school_code(retrieval_anchor + "\n" + latest_question, req.school_code)
        output_mode = infer_output_selection(latest_question, req.output_mode)

        if unknown_code:
            answer = unknown_school_answer(unknown_code)
            if req.stream:
                return openai_streaming_response(answer, req.model)
            return JSONResponse(openai_chat_response(answer, req.model))

        section_name, contexts = retrieve_contexts(retrieval_anchor, school_code, output_mode, req.top_k)
        if not is_context_question(latest_question, contexts):
            answer = OUT_OF_SCOPE_ANSWER
        elif should_return_source_directly(user_messages, latest_question):
            answer = format_source_answer(school_code, section_name, output_mode, contexts)
        else:
            answer, _provider = chat_with_fallback(latest_question, contexts, req.messages)

        if req.stream:
            return openai_streaming_response(answer, req.model)
        return JSONResponse(openai_chat_response(answer, req.model))
    except Exception as exc:
        print(f"ERROR in /v1/chat/completions: {exc!r}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    try:
        output_mode = infer_output_selection(req.question, req.output_mode)
        school_code, unknown_code = resolve_school_code(req.question, req.school_code)
        if unknown_code:
            return {
                "school_code": unknown_code.upper(),
                "section_name": None,
                "output_mode": output_mode,
                "answer": unknown_school_answer(unknown_code),
                "contexts": [],
                "collection": COLLECTION_NAME,
                "embedding_model": EMBED_MODEL,
                "chat_model": CHAT_MODEL,
            }

        section_name = req.section_name or find_explicit_section(req.question, school_code)
        if section_name:
            contexts = apply_output_selection(local_section_contexts(school_code, section_name), output_mode)
        else:
            section_name, contexts = retrieve_contexts(req.question, school_code, output_mode, req.top_k)

        answer = format_source_answer(school_code, section_name, output_mode, contexts)
        return {
            "school_code": school_code,
            "section_name": section_name,
            "output_mode": output_mode,
            "answer": answer,
            "contexts": contexts,
            "collection": COLLECTION_NAME,
            "embedding_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
