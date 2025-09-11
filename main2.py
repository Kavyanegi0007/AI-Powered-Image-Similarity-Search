import os
import sys
import json
import time
import uuid
import hashlib
import asyncio
import mimetypes
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# MCP (official SDK)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Cloudinary
import cloudinary
import cloudinary.uploader

# HTTP downloads
import requests

# Azure OpenAI (chat + responses)
from openai import AzureOpenAI

# ===============================
# ENV & GLOBALS
# ===============================
load_dotenv(override=True)

# ---- Azure OpenAI ----
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "BmaiYil8P7o3Dgv0JzIEIA4JYd3AHl7Jh6SzBdjkwXfF4DNxCzC3JQQJ99BGACYeBjFXJ3w3AAABACOGZkhi")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://gpt-4o-intern.openai.azure.com/")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-08-06")
DEFAULT_MODEL = AZURE_DEPLOYMENT

oai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# ---- Cloudinary ----
CLOUDINARY_CLOUD_NAME = "dczody87a"
CLOUDINARY_API_KEY = "553446782129746"
CLOUDINARY_API_SECRET = "Nd48WZ0KFadHbUdZY_jnpTAKeRo"

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True,
)

def upload_image_to_cloudinary(image_path: str) -> Optional[str]:
    """Uploads a local image file to Cloudinary and returns the secure URL, or None on failure."""
    try:
        resp = cloudinary.uploader.upload(image_path)
        print(f"✅ Uploaded image to Cloudinary: {resp.get('secure_url')}")
        return resp.get("secure_url")
    except Exception as e:
        print(f"❌ Cloudinary upload failed for {image_path}: {e}")
        return None

def ensure_remote_url(image_path_or_url: str) -> Optional[str]:
    """If input looks like a URL, return as-is. If it's a local path, upload to Cloudinary and return the secure URL."""
    s = (image_path_or_url or "").strip().strip('"').strip("'")
    if s.lower().startswith(("http://", "https://")):
        print(f"📎 Using existing URL: {s}")
        return s
    print(f"📤 Uploading local image: {s}")
    return upload_image_to_cloudinary(s)

# ===============================
# Tool Schemas (for LLM tool-calling)
# ===============================
IMAGE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "image_search",
        "description": "Return N related images for an input image URL via SerpAPI's Google Lens.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {"type": "string", "description": "URL of the image to search for similar images."},
                "num": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "How many images to return (1–10).",
                },
            },
            "required": ["image_url"],
        },
    },
}

SEARCH_BY_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_by_image",
        "description": "Return top-K similar images from Azure AI Search over the local dataset.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_image_path": {"type": "string", "description": "Local path or URL to the query image."},
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "How many similar images to return from the DB.",
                },
            },
            "required": ["query_image_path"],
        },
    },
}

SYSTEM_PROMPT = (
    "You have access to two tools via MCP: `search_by_image` (Azure Cognitive Search over our indexed dataset) and "
    "`image_search` (SerpAPI Google Lens). "
    "When the user provides an image URL and asks for similar images, call both `search_by_image` and `image_search` with "
    "`query_image_path` or `image_url` set to the provided URL and `k` or `num` set to the desired count. "
    "Present results with DB matches first (include local paths/descriptions and scores if provided), "
    "then web results (titles and URLs). Keep raw JSON structured and concise."
)

# ===============================
# Helpers: output folder, downloads, parsing
# ===============================
def make_run_folder(output_root: Path) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}-{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "db").mkdir(exist_ok=True)
    (run_dir / "web").mkdir(exist_ok=True)
    (run_dir / "all").mkdir(exist_ok=True)
    (run_dir / "kept").mkdir(exist_ok=True)
    print(f"📂 Created run folder: {run_dir}")
    return run_dir

def safe_name_from_url(url: str) -> str:
    ext = Path(url).suffix
    if not ext or len(ext) > 6:
        guess = mimetypes.guess_extension(mimetypes.guess_type(url)[0] or "") or ".jpg"
        ext = guess
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{h}{ext}"

def download_one(url: str, out_path: Path, timeout: int = 20, retries: int = 3) -> Optional[Path]:
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, stream=True, headers=headers)
            r.raise_for_status()
            content_type = r.headers.get("content-type", "").lower()
            print(f"Downloading {url}: Content-Type = {content_type}")
            if "image" not in content_type and "octet-stream" not in content_type:
                print(f"⚠️ Skipped {url}: Content-Type {content_type} is not an image")
                return None
            first_chunk = next(r.iter_content(8192))
            valid_signatures = [
                b"\xff\xd8",  # JPEG
                b"\x89PNG",  # PNG
                b"\x52\x49\x46\x46"  # WebP (RIFF)
            ]
            if not any(first_chunk.startswith(sig) for sig in valid_signatures):
                print(f"⚠️ Skipped {url}: Content does not start with valid image signature (first 4 bytes: {first_chunk[:4].hex()})")
                return None
            with open(out_path, "wb") as f:
                f.write(first_chunk)
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"✅ Downloaded: {url} -> {out_path}")
            return out_path
        except Exception as e:
            print(f"⚠️ Failed to download {url} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(1)
    return None

def download_or_copy(paths_or_urls: List[str], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for s in paths_or_urls:
        if s.lower().startswith(("http://", "https://")):
            name = safe_name_from_url(s)
            p = out_dir / name
            saved = download_one(s, p)
            if saved:
                results.append(saved)
        else:
            src = Path(s)
            if src.exists():
                target = out_dir / src.name
                target.write_bytes(src.read_bytes())
                print(f"✅ Copied local: {src} -> {target}")
                results.append(target)
            else:
                print(f"⚠️ Local file not found: {s}")
        time.sleep(0.5)  # Avoid rate limiting
    return results

def extract_urls_from_items(items: Any, source: str = "web") -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    print(f"Extracting URLs from items: {items}")

    def maybe_add(v: Optional[str], score: Optional[float] = None):
        if isinstance(v, str) and (v.lower().startswith(("http://", "https://")) or source == "db"):
            results.append({"url": v, "score": score})
            print(f"Added URL: {v}")

    if isinstance(items, dict) and "result" in items:
        items = items.get("result", [])
        print(f"Extracted 'result' key: {items}")

    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        print(f"Items is not a list or dict: {items}")
        return results

    for it in items:
        if not isinstance(it, dict):
            print(f"Skipping non-dict item: {it}")
            continue
        score = it.get("score") if source == "db" else None
        if source == "web":
            v = it.get("thumbnail")
            if isinstance(v, str):
                maybe_add(v, score)
                continue
        for k in ("decoded_path", "url", "image_url", "contentUrl", "link", "src", "image", "thumbnail"):
            v = it.get(k)
            if isinstance(v, str):
                maybe_add(v, score)
            elif isinstance(v, dict):
                for kk in ("url", "contentUrl", "src"):
                    maybe_add(v.get(kk), score)
        for k in ("results", "images", "items", "data"):
            sub = it.get(k)
            if isinstance(sub, list):
                results.extend(extract_urls_from_items(sub, source))
            elif isinstance(sub, dict):
                results.extend(extract_urls_from_items(sub, source))
    seen = set()
    uniq = []
    for r in results:
        u = r["url"]
        if u not in seen:
            uniq.append(r)
            seen.add(u)
    print(f"Extracted URLs: {uniq}")
    return uniq

# ===============================
# Core session runner (MCP + chat completions)
# ===============================
async def run_session(user_text: str, model: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    model = model or DEFAULT_MODEL

    server_python = sys.executable
    venv_python = Path(os.getcwd()) / "proj_env" / "Scripts" / "python.exe"
    if venv_python.exists():
        server_python = str(venv_python)

    server = StdioServerParameters(
        command=server_python,
        args=["-u", "server.py"],
        env=os.environ.copy(),
        cwd=os.getcwd(),
    )

    tool_payloads: Dict[str, Any] = {"search_by_image": None, "image_search": None}

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]

            try:
                resp = oai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=[IMAGE_SEARCH_TOOL, SEARCH_BY_IMAGE_TOOL],
                    tool_choice="auto",
                )
                msg = resp.choices[0].message
            except Exception as e:
                print(f"❌ Azure OpenAI chat completion failed: {e}")
                return "", tool_payloads

            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    if tc.type != "function":
                        continue
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                        print(f"⚠️ Failed to parse tool args for {name}")

                    tool_result = await session.call_tool(name, args)
                    payload = getattr(tool_result, "structuredContent", None) or getattr(tool_result, "content", None) or tool_result

                    try:
                        payload_serializable = json.loads(payload) if isinstance(payload, str) else payload
                    except Exception:
                        payload_serializable = payload

                    tool_payloads[name] = payload_serializable
                    print(f"🛠️ Tool {name} called with args {args}, payload: {json.dumps(payload_serializable, indent=2)}")

                    messages.append({"role": "assistant", "tool_calls": [tc], "content": None})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(payload_serializable, ensure_ascii=False),
                    })

                try:
                    final = oai_client.chat.completions.create(model=model, messages=messages)
                    final_text = final.choices[0].message.content or ""
                    print(f"📝 Final LLM response: {final_text}")
                    return final_text, tool_payloads
                except Exception as e:
                    print(f"❌ Azure OpenAI final completion failed: {e}")
                    return "", tool_payloads
            else:
                final_text = msg.content or ""
                print(f"📝 LLM response (no tools): {final_text}")
                return final_text, tool_payloads

# ===============================
# Minimal Azure comparator (Chat Completions with base64 images)
# ===============================
def _download_query_if_needed(url: str, out_dir: Path) -> Path:
    if url.startswith("file://"):
        p = Path(url[7:])
        print(f"📎 Using local query image: {p}")
        return p
    if url.startswith(("http://", "https://")):
        ext = mimetypes.guess_extension(mimetypes.guess_type(url)[0] or "") or ".jpg"
        p = out_dir / f"query{ext}"
        r = requests.get(url, stream=True, timeout=20)
        r.raise_for_status()
        with open(p, "wb") as f:
            for c in r.iter_content(8192):
                if c: f.write(c)
        print(f"✅ Downloaded query image: {url} -> {p}")
        return p
    p = Path(url)
    print(f"📎 Using local query image: {p}")
    return p

def get_base64(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"⚠️ Failed to encode {path} to base64: {e}")
        return ""

def score_one(candidate: Path, source: str, qpath: Path) -> Optional[Dict]:
    try:
        q_b64 = get_base64(qpath)
        c_b64 = get_base64(candidate)
        if not q_b64 or not c_b64:
            print(f"⚠️ Skipping {candidate.name}: Failed to encode images")
            return None
        q_ext = qpath.suffix[1:] or "jpeg"
        c_ext = candidate.suffix[1:] or "jpeg"
        content = [
            {"type": "text", "text": "Compare these two images and give a similarity score from 0 to 1, where 1 is identical and 0 is completely different. Respond only with JSON: {\"similarity_score\": float}"},
            {"type": "image_url", "image_url": {"url": f"data:image/{q_ext};base64,{q_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/{c_ext};base64,{c_b64}"}},
        ]
        resp = oai_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=200,
        )
        out = resp.choices[0].message.content.strip()
        if out.startswith("```json") and out.endswith("```"):
            out = out[7:-3].strip()
        try:
            data = json.loads(out)
            s = float(data.get("similarity_score", 0.0))
            print(f"🔍 {source} {candidate.name}: GPT similarity score = {s:.4f}")
            return {"path": str(candidate), "source": source, "score": s}
        except json.JSONDecodeError as e:
            print(f"⚠️ Invalid JSON response for {candidate.name}: {out}, error: {e}")
            return None
    except Exception as e:
        print(f"⚠️ Compare failed for {candidate.name}: {e}")
        return None

def compare_all(query_url: str, db_paths: List[Path], web_paths: List[Path], run_dir: Path,
                model: str = AZURE_DEPLOYMENT, threshold: float = 0.50) -> dict:
    qpath = _download_query_if_needed(query_url, run_dir)
    scores = {"db": [], "web": [], "kept": []}
    kept_dir = run_dir / "kept"

    print(f"🔍 Comparing {len(db_paths)} DB images and {len(web_paths)} web images")
    for p in db_paths:
        res = score_one(p, "db", qpath)
        if res:
            scores["db"].append(res)
            if res["score"] >= threshold:
                scores["kept"].append(res)
                target = kept_dir / p.name
                target.write_bytes(p.read_bytes())
                print(f"✅ Kept: {target} (score {res['score']:.4f})")

    for p in web_paths:
        res = score_one(p, "web", qpath)
        if res:
            scores["web"].append(res)
            if res["score"] >= threshold:
                scores["kept"].append(res)
                target = kept_dir / p.name
                target.write_bytes(p.read_bytes())
                print(f"✅ Kept: {target} (score {res['score']:.4f})")

    with open(run_dir / "scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print(f"📄 Saved scores to {run_dir / 'scores.json'}")

    return scores

# ===============================
# CLI entrypoint
# ===============================
def main() -> int:
    """
    Usage:
        python main.py <image_path_or_url> [k] [output_root] [threshold] [azure_cos_threshold]

    Examples:
        python main.py "path/to/image.jpg" 5 outputs 0.5 0.8
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path_or_url> [k] [output_root] [threshold] [azure_cos_threshold]")
        return 1

    image_arg = sys.argv[1]
    try:
        k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    except Exception:
        k = 5
    output_root = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("outputs")
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.50
    azure_cos_threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.80

    output_root.mkdir(parents=True, exist_ok=True)

    # Always get a remote URL (upload locals to Cloudinary)
    remote_url = ensure_remote_url(image_arg)
    if not remote_url:
        print("❌ Could not resolve a usable image URL.")
        return 2

    # Trigger tools
    user_text = f"Find {k} similar images for this image: {remote_url}"

    try:
        final_text, tool_payloads = asyncio.run(run_session(user_text))

        # Prepare output folders
        run_dir = make_run_folder(output_root)
        db_dir = run_dir / "db"
        web_dir = run_dir / "web"
        kept_dir = run_dir / "kept"

        # Extract URLs and scores from tool payloads
        db_results = tool_payloads.get("search_by_image", {}).get("result", []) if isinstance(tool_payloads.get("search_by_image"), dict) else tool_payloads.get("search_by_image", [])
        if isinstance(db_results, dict): db_results = [db_results]
        filtered_db_urls = []
        for res in db_results:
            if not isinstance(res, dict):
                print(f"⚠️ Skipping invalid DB result: {res}")
                continue
            url = res.get("decoded_path")
            score = res.get("score", 0.0)
            if url:
                print(f"DB result: {url}, score: {score:.4f}")
                if score >= azure_cos_threshold:
                    filtered_db_urls.append({"url": url, "score": score})
                    print(f"✅ Passed Azure cosine threshold: {url} (score {score:.4f})")
                else:
                    print(f"❌ Filtered out: {url} (score {score:.4f} < {azure_cos_threshold})")
            else:
                print(f"⚠️ No decoded_path in DB result: {res}")

        web_results = extract_urls_from_items(tool_payloads.get("image_search"), source="web")
        web_urls = [r["url"] for r in web_results]
        for url in web_urls:
            print(f"Web result: {url}")

        # Download/copy candidates
        saved_db = download_or_copy([r["url"] for r in filtered_db_urls], db_dir)
        saved_web = download_or_copy(web_urls, web_dir)

        # Copy everything into all/
        all_dir = run_dir / "all"
        for p in saved_db + saved_web:
            try:
                target = all_dir / p.name
                if not target.exists():
                    target.write_bytes(p.read_bytes())
                    print(f"✅ Copied to all: {target}")
            except Exception as e:
                print(f"⚠️ Could not copy {p} to all/: {e}")

        # Build manifest
        manifest = {
            "query_image_url": remote_url,
            "k": k,
            "run_dir": str(run_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "db_result_count": len(filtered_db_urls),
            "web_result_count": len(web_urls),
            "db_downloaded_files": [str(p) for p in saved_db],
            "web_downloaded_files": [str(p) for p in saved_web],
            "raw": {
                "search_by_image": tool_payloads.get("search_by_image"),
                "image_search": tool_payloads.get("image_search"),
            },
            "final_text": final_text,
        }

        # Save manifest
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"📄 Saved manifest to {run_dir / 'manifest.json'}")

        # Run comparator
        scores = compare_all(
            query_url=remote_url,
            db_paths=saved_db,
            web_paths=saved_web,
            run_dir=run_dir,
            model=AZURE_DEPLOYMENT,
            threshold=threshold,
        )

        # Attach scores to manifest and re-save
        manifest["scores"] = scores
        manifest["kept_files"] = [r["path"] for r in scores["kept"]]

        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        # Console summary
        print("\n=== RESULTS SAVED ===")
        print(f"Run folder: {run_dir}")
        print(f"DB images:  {len(saved_db)} saved to {db_dir}")
        print(f"Web images: {len(saved_web)} saved to {web_dir}")
        print(f"Kept images: {len(manifest['kept_files'])} in {kept_dir}")
        print(f"Manifest:   {run_dir / 'manifest.json'}\n")
        print("=== MODEL SUMMARY ===")
        print(final_text or "")
        print(f"\n=== KEPT IMAGES (passed GPT threshold {threshold}) ===")
        for kf in manifest["kept_files"]:
            print(kf)
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 3

if __name__ == "__main__":
    sys.exit(main())