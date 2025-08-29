# app.py — Async TEXT + IMAGE (Cloud Tasks + Firestore + GCS) + LOCAL_MODE za razvoj (bez timeouta)
# app.py — Async TEXT + IMAGE (Cloud Tasks + Firestore + GCS) + HYBRID (auto|sync|async) + LOCAL_MODE za razvoj
from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory, jsonify
from dotenv import load_dotenv
import os, re, base64, json, html, datetime, logging, mimetypes, threading, traceback
@@ -55,12 +55,25 @@
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- TIMEOUT ---
HARD_TIMEOUT_S = float(os.getenv("HARD_TIMEOUT_S", "120"))
OPENAI_TIMEOUT = HARD_TIMEOUT_S
OPENAI_MAX_RETRIES = 2

# --- HYBRID pragovi/timeouti ---
SYNC_SOFT_TIMEOUT_S = float(os.getenv("SYNC_SOFT_TIMEOUT_S", "8"))          # meki rok za sinhroni pokušaj
HEAVY_TOKEN_THRESHOLD = int(os.getenv("HEAVY_TOKEN_THRESHOLD", "1500"))     # grubi prag za “težak” tekst

def _budgeted_timeout(default: float | int = None, margin: float = 5.0) -> float:
    run_lim = float(os.getenv("RUN_TIMEOUT_SECONDS", "300") or 300)
    want = float(default if default is not None else OPENAI_TIMEOUT)
    return max(5.0, min(want, run_lim - margin))

# --- OpenAI ---
_OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
if not _OPENAI_API_KEY:
    log.error("OPENAI_API_KEY nije postavljen u okruženju.")
# Uklonjeni timeouti i custom retry-jevi
client = OpenAI(api_key=_OPENAI_API_KEY)
client = OpenAI(api_key=_OPENAI_API_KEY, timeout=OPENAI_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)

MODEL_VISION_LIGHT = os.getenv("OPENAI_MODEL_VISION_LIGHT") or os.getenv("OPENAI_MODEL_VISION", "gpt-5")
MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT", "gpt-5-mini")
@@ -169,7 +182,6 @@
    if fs_db:
        fs_db.collection("jobs").document(job_id).set(data, merge=merge)
    else:
        # merge simulacija
        JOB_STORE[job_id] = {**JOB_STORE.get(job_id, {}), **data}

def read_job(job_id: str) -> dict:
@@ -242,23 +254,26 @@
        return expr
    return None

# ===== OpenAI helper (bez timeouta) =====
def _openai_chat(model: str, messages: list, max_tokens: int | None = None):
# ===== OpenAI helper =====
def _openai_chat(model: str, messages: list, timeout: float = None, max_tokens: int | None = None):
    def _do(params):
        cli = client if timeout is None else client.with_options(timeout=timeout)
        return cli.chat.completions.create(**params)
    params = {"model": model, "messages": messages}
    if max_tokens is not None:
        params["max_completion_tokens"] = max_tokens
    try:
        return client.chat.completions.create(**params)
        return _do(params)
    except Exception as e:
        msg = str(e)
        if "max_completion_tokens" in msg or "Unsupported parameter: 'max_completion_tokens'" in msg:
            params.pop("max_completion_tokens", None)
            if max_tokens is not None: params["max_tokens"] = max_tokens
            return client.chat.completions.create(**params)
            return _do(params)
        raise

# ===== Pipelines =====
def answer_with_text_pipeline(pure_text: str, razred: str, history, requested):
def answer_with_text_pipeline(pure_text: str, razred: str, history, requested, timeout_override: float | None = None):
    prompt_za_razred = PROMPTI_PO_RAZREDU.get(razred, PROMPTI_PO_RAZREDU["5"])
    only_clause = ""
    strict_geom_policy = (" Ako problem uključuje geometriju: "
@@ -281,7 +296,7 @@
        messages.append({"role":"user","content": msg["user"]})
        messages.append({"role":"assistant","content": msg["bot"]})
    messages.append({"role":"user","content": pure_text})
    response = _openai_chat(MODEL_TEXT, messages)
    response = _openai_chat(MODEL_TEXT, messages, timeout=timeout_override or OPENAI_TIMEOUT)
    actual_model = getattr(response, "model", MODEL_TEXT)
    raw = response.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
@@ -322,21 +337,21 @@
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"

def route_image_flow_url(image_url: str, razred: str, history, user_text=None):
def route_image_flow_url(image_url: str, razred: str, history, user_text=None, timeout_override: float | None = None):
    only_clause, strict_geom_policy = _vision_clauses()
    messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)
    user_content = []
    if user_text: user_content.append({"type": "text", "text": f"Korisnički tekst: {user_text}"})
    user_content.append({"type": "text", "text": "Na slici je matematički zadatak."})
    user_content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages.append({"role": "user", "content": user_content})
    resp = _openai_chat(MODEL_VISION, messages)
    resp = _openai_chat(MODEL_VISION, messages, timeout=timeout_override or OPENAI_TIMEOUT)
    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
    return f"<p>{latexify_fractions(raw)}</p>", "vision_url", actual_model

def route_image_flow(slika_bytes: bytes, razred: str, history, user_text=None, mime_hint: str | None = None):
def route_image_flow(slika_bytes: bytes, razred: str, history, user_text=None, timeout_override: float | None = None, mime_hint: str | None = None):
    only_clause, strict_geom_policy = _vision_clauses()
    messages = _vision_messages_base(razred, history, only_clause, strict_geom_policy)
    data_url = _bytes_to_data_url(slika_bytes, mime_hint=mime_hint)
@@ -345,7 +360,7 @@
    user_content.append({"type": "text", "text": "Na slici je matematički zadatak."})
    user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    messages.append({"role": "user", "content": user_content})
    resp = _openai_chat(MODEL_VISION, messages)
    resp = _openai_chat(MODEL_VISION, messages, timeout=timeout_override or OPENAI_TIMEOUT)
    actual_model = getattr(resp, "model", MODEL_VISION)
    raw = resp.choices[0].message.content
    raw = strip_ascii_graph_blocks(raw)
@@ -380,6 +395,20 @@
    return fence.sub(repl, text)

# ===================== GCS upload helper (samo u cloud modu) =====================
def gcs_upload_bytes(job_id: str, raw: bytes, filename_hint: str = "image.bin", content_type: str | None = None) -> str | None:
    if not (storage_client and GCS_BUCKET):
        return None
    ext = os.path.splitext(filename_hint or "")[1].lower() or ".bin"
    blob_name = f"uploads/{job_id}/{uuid4().hex}{ext}"
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    try:
        blob.upload_from_string(raw, content_type=content_type or "application/octet-stream")
        return blob_name
    except Exception as e:
        log.error("GCS upload_from_string failed: %s", e)
        return None

def gcs_upload_filestorage(f):
    if not (storage_client and GCS_BUCKET):
        return None
@@ -419,22 +448,22 @@

            if image_url:
                combined_text = pitanje
                odgovor, used_path, used_model = route_image_flow_url(image_url, razred, history, user_text=combined_text)
                odgovor, used_path, used_model = route_image_flow_url(image_url, razred, history, user_text=combined_text, timeout_override=HARD_TIMEOUT_S)
                session["last_image_url"] = image_url
                if (not plot_expression_added) and should_plot(combined_text):
                    expr = extract_plot_expression(combined_text, razred=razred, history=history)
                    if expr: odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True
                display_user = (combined_text + " [slika]") if combined_text else "[slika]"
                history.append({"user": display_user, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sync_job_id = f"sync-{uuid4().hex[:8]}"; log_to_sheet(sync_job_id, razred, combined_text, odgovor, used_path, used_model)
                sync_job_id = f"sync-{uuid4().hex[:8]}"; log_to_sheet(sync_job_id, razred, combined_text, odgovor, "vision_url", used_model)
                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            if slika and slika.filename:
                combined_text = pitanje
                body = slika.read()
                odgovor, used_path, used_model = route_image_flow(body, razred, history, user_text=combined_text, mime_hint=slika.mimetype or None)
                odgovor, used_path, used_model = route_image_flow(body, razred, history, user_text=combined_text, timeout_override=HARD_TIMEOUT_S, mime_hint=slika.mimetype or None)
                try:
                    ext = os.path.splitext(slika.filename or "")[1].lower() or ".img"
                    fname = f"{uuid4().hex}{ext}"
@@ -449,24 +478,24 @@
                display_user = (combined_text + " [SLIKA]") if combined_text else "[SLIKA]"
                history.append({"user": display_user, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sync_job_id = f"sync-{uuid4().hex[:8]}"; log_to_sheet(sync_job_id, razred, combined_text, odgovor, used_path, used_model)
                sync_job_id = f"sync-{uuid4().hex[:8]}"; log_to_sheet(sync_job_id, razred, combined_text, odgovor, "vision_direct", used_model)
                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            requested = extract_requested_tasks(pitanje)
            last_url = session.get("last_image_url")
            if last_url and (requested or (pitanje and FOLLOWUP_TASK_RE.match(pitanje))):
                odgovor, used_path, used_model = route_image_flow_url(last_url, razred, history, user_text=pitanje)
                odgovor, used_path, used_model = route_image_flow_url(last_url, razred, history, user_text=pitanje, timeout_override=HARD_TIMEOUT_S)
                if (not plot_expression_added) and should_plot(pitanje):
                    expr = extract_plot_expression(pitanje, razred=razred, history=history)
                    if expr: odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True
                history.append({"user": pitanje, "bot": odgovor.strip()})
                history = history[-8:]; session["history"] = history
                sync_job_id = f"sync-{uuid4().hex[:8]}"; log_to_sheet(sync_job_id, razred, pitanje, odgovor, used_path, used_model)
                sync_job_id = f"sync-{uuid4().hex[:8]}"; log_to_sheet(sync_job_id, razred, pitanje, odgovor, "vision_url", used_model)
                if is_ajax: return render_template("index.html", history=history, razred=razred)
                return redirect(url_for("index"))

            odgovor, actual_model = answer_with_text_pipeline(pitanje, razred, history, requested)
            odgovor, actual_model = answer_with_text_pipeline(pitanje, razred, history, requested, timeout_override=HARD_TIMEOUT_S)
            if (not plot_expression_added) and should_plot(pitanje):
                expr = extract_plot_expression(pitanje, razred=razred, history=history)
                if expr: odgovor = add_plot_div_once(odgovor, expr); plot_expression_added = True
@@ -569,22 +598,27 @@
             "yes" if bool(image_inline_b64) else "no")

    history = []
    task_ai_timeout = _budgeted_timeout(default=HARD_TIMEOUT_S, margin=5.0)

    # Glavni tok (bez bilo kakvih timeout argumenata)
    # Glavni tok
    if image_path:
        if not storage_client:
            raise RuntimeError("GCS storage client not initialized (image_path zadat).")
        blob = storage_client.bucket(bucket).blob(image_path)
        img_bytes = blob.download_as_bytes()
        mime_hint = blob.content_type or mimetypes.guess_type(image_path)[0] or None
        odgovor_html, used_path, used_model = route_image_flow(img_bytes, razred, history=history, user_text=user_text, mime_hint=mime_hint)
        odgovor_html, used_path, used_model = route_image_flow(img_bytes, razred, history=history, user_text=user_text,
                                                               timeout_override=task_ai_timeout, mime_hint=mime_hint)
    elif image_inline_b64:
        img_bytes = base64.b64decode(image_inline_b64)
        odgovor_html, used_path, used_model = route_image_flow(img_bytes, razred, history=history, user_text=user_text, mime_hint=None)
        odgovor_html, used_path, used_model = route_image_flow(img_bytes, razred, history=history, user_text=user_text,
                                                               timeout_override=task_ai_timeout, mime_hint=None)
    elif image_url:
        odgovor_html, used_path, used_model = route_image_flow_url(image_url, razred, history=history, user_text=user_text)
        odgovor_html, used_path, used_model = route_image_flow_url(image_url, razred, history=history, user_text=user_text,
                                                                   timeout_override=task_ai_timeout)
    else:
        odgovor_html, used_model = answer_with_text_pipeline(user_text, razred, history, requested)
        odgovor_html, used_model = answer_with_text_pipeline(user_text, razred, history, requested,
                                                             timeout_override=task_ai_timeout)
        used_path = "text"

    result = {"html": odgovor_html, "path": used_path, "model": used_model}
@@ -612,80 +646,198 @@
        store_job(job_id, {"status": "done", "result": {"html": err_html, "path": "error", "model": "n/a"},
                           "finished_at": datetime.datetime.utcnow().isoformat() + "Z"}, merge=True)

# --------- SUBMIT (radi lokalno i u cloudu) ----------
# --------- Heuristike i sync helperi (NOVO) ----------
def estimate_tokens(text: str) -> int:
    if not text: return 0
    # gruba procjena: ~4 karaktera ≈ 1 token
    return max(0, len(text) // 4)

def looks_heavy(user_text: str, has_image: bool) -> bool:
    toks = estimate_tokens(user_text or "")
    return has_image or toks > HEAVY_TOKEN_THRESHOLD

def _sync_process_once(razred: str, user_text: str, requested: list, image_url: str | None,
                       file_bytes: bytes | None, file_mime: str | None, timeout_s: float) -> dict:
    """Vraća {'ok': True, 'result': {...}} ili {'ok': False, 'error': '...'}"""
    try:
        history = []
        if image_url:
            html_out, used_path, used_model = route_image_flow_url(image_url, razred, history, user_text=user_text, timeout_override=timeout_s)
            return {"ok": True, "result": {"html": html_out, "path": used_path, "model": used_model}}
        if file_bytes:
            html_out, used_path, used_model = route_image_flow(file_bytes, razred, history, user_text=user_text, timeout_override=timeout_s, mime_hint=file_mime or None)
            return {"ok": True, "result": {"html": html_out, "path": used_path, "model": used_model}}
        # tekst
        html_out, used_model = answer_with_text_pipeline(user_text, razred, history, requested, timeout_override=timeout_s)
        return {"ok": True, "result": {"html": html_out, "path": "text", "model": used_model}}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _prepare_async_payload(job_id: str, razred: str, user_text: str, requested: list,
                           image_url: str | None, file_bytes: bytes | None, file_name: str | None, file_mime: str | None,
                           image_b64_str: str | None) -> dict:
    """Priprema payload za worker; u cloud modu šalje u GCS, inače inline b64."""
    payload = {
        "job_id": job_id, "razred": razred, "user_text": user_text, "requested": requested,
        "bucket": GCS_BUCKET, "image_path": None, "image_url": image_url or None,
        "image_inline_b64": None,
    }
    # 1) file iz forme?
    if file_bytes:
        if not LOCAL_MODE and (storage_client and GCS_BUCKET):
            path = gcs_upload_bytes(job_id, file_bytes, filename_hint=(file_name or "image.bin"), content_type=file_mime or "application/octet-stream")
            if path: payload["image_path"] = path
        else:
            payload["image_inline_b64"] = base64.b64encode(file_bytes).decode()
        return payload

    # 2) image_b64 iz JSON-a?
    if image_b64_str:
        b64_clean = image_b64_str.split(",", 1)[1] if "," in image_b64_str else image_b64_str
        if not LOCAL_MODE and (storage_client and GCS_BUCKET):
            try:
                raw = base64.b64decode(b64_clean)
            except Exception:
                raw = b""
            path = gcs_upload_bytes(job_id, raw, filename_hint="image.bin", content_type="application/octet-stream")
            if path: payload["image_path"] = path
        else:
            payload["image_inline_b64"] = b64_clean
        return payload

    # 3) samo URL ili samo tekst
    return payload

# --------- HYBRID /submit (NOVO) ----------
@app.route("/submit", methods=["POST", "OPTIONS"])
def submit_async():
def submit():
    if request.method == "OPTIONS":
        return ("", 204)

    # --- Ulaz (form + json) ---
    razred = (request.form.get("razred") or request.args.get("razred") or "").strip()
    user_text = (request.form.get("user_text") or request.form.get("pitanje") or "").strip()
    image_url = (request.form.get("image_url") or request.args.get("image_url") or "").strip()
    mode = (request.form.get("mode") or request.args.get("mode") or "auto").strip().lower()

    data = request.get_json(silent=True) or {}
    if data:
        razred    = (data.get("razred")    or razred).strip()
        user_text = (data.get("pitanje")   or data.get("user_text") or user_text).strip()
        image_url = (data.get("image_url") or image_url).strip()
        mode      = (data.get("mode")      or mode).strip().lower()

    if razred not in DOZVOLJENI_RAZREDI:
        # fallback na 5 zbog stabilnosti API-ja (front može validirati)
        razred = "5"

    requested = extract_requested_tasks(user_text)

    job_id = str(uuid4())
    # --- Slikovni ulazi (file/b64) ---
    file_storage = request.files.get("file")
    file_bytes = None
    file_mime = None
    file_name = None
    if file_storage and file_storage.filename:
        file_bytes = file_storage.read()
        file_mime = file_storage.mimetype or "application/octet-stream"
        file_name = file_storage.filename

    image_b64_str = (data.get("image_b64") if data else None)

    has_image = bool(image_url or file_bytes or image_b64_str)

    # --- Režim ---
    if mode not in ("auto", "sync", "async"):
        mode = "auto"

    # --- Ako je force-async: odmah u red ---
    if mode == "async":
        job_id = str(uuid4())
        store_job(job_id, {"status": "pending", "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                           "razred": razred, "user_text": user_text, "requested": requested}, merge=True)
        payload = _prepare_async_payload(job_id, razred, user_text, requested, image_url or None,
                                         file_bytes, file_name, file_mime, image_b64_str)
        try:
            if LOCAL_MODE:
                threading.Thread(target=_local_worker, args=(payload,), daemon=True).start()
            else:
                _create_task_cloud(payload)
            return jsonify({"mode": "async", "job_id": job_id, "status": "queued", "local_mode": LOCAL_MODE}), 202
        except Exception as e:
            log.error("submit async failed: %s\n%s", e, traceback.format_exc())
            store_job(job_id, {"status": "error", "error": str(e)}, merge=True)
            return jsonify({"error": "submit_failed", "detail": str(e), "job_id": job_id}), 500

    # --- AUTO/SYNC odluka ---
    heavy = looks_heavy(user_text, has_image=has_image)

    # Ako auto i teško → async odma'
    if mode == "auto" and heavy:
        job_id = str(uuid4())
        store_job(job_id, {"status": "pending", "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                           "razred": razred, "user_text": user_text, "requested": requested}, merge=True)
        payload = _prepare_async_payload(job_id, razred, user_text, requested, image_url or None,
                                         file_bytes, file_name, file_mime, image_b64_str)
        try:
            if LOCAL_MODE:
                threading.Thread(target=_local_worker, args=(payload,), daemon=True).start()
            else:
                _create_task_cloud(payload)
            return jsonify({"mode": "auto→async", "job_id": job_id, "status": "queued", "local_mode": LOCAL_MODE}), 202
        except Exception as e:
            log.error("submit auto→async failed: %s\n%s", e, traceback.format_exc())
            store_job(job_id, {"status": "error", "error": str(e)}, merge=True)
            return jsonify({"error": "submit_failed", "detail": str(e)}), 500

    # Inače probaj kratki sinhroni pokušaj
    sync_try = _sync_process_once(
        razred=razred,
        user_text=user_text,
        requested=requested,
        image_url=(image_url or None),
        file_bytes=file_bytes,
        file_mime=file_mime,
        timeout_s=SYNC_SOFT_TIMEOUT_S
    )

    if sync_try.get("ok"):
        # (Opcionalno) auto-plot trigger iz teksta
        html_out = sync_try["result"]["html"]
        if should_plot(user_text):
            expr = extract_plot_expression(user_text, razred=razred, history=[])
            if expr: html_out = add_plot_div_once(html_out, expr)

        # log u Sheets
        try: log_to_sheet(f"sync-{uuid4().hex[:8]}", razred, user_text, html_out, sync_try["result"]["path"], sync_try["result"]["model"])
        except Exception as _e: log.warning("Sheets log fail (sync): %s", _e)

        mode_tag = "auto(sync)" if mode == "auto" else "sync"
        return jsonify({
            "mode": mode_tag,
            "result": {"html": html_out, "path": sync_try["result"]["path"], "model": sync_try["result"]["model"]}
        }), 200

    # inicijalno upiši pending
    # Sinhroni nije uspio (timeout/greška) → fallback u red
    job_id = str(uuid4())
    store_job(job_id, {"status": "pending", "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                       "razred": razred, "user_text": user_text, "requested": requested}, merge=True)

    payload = {
        "job_id": job_id, "razred": razred, "user_text": user_text, "requested": requested,
        "bucket": GCS_BUCKET, "image_path": None, "image_url": image_url or None,
        "image_inline_b64": None,
    }

    # FILE -> GCS (cloud) ili inline (local)
    if "file" in request.files:
        f = request.files["file"]
        if not LOCAL_MODE and (storage_client and GCS_BUCKET):
            name = f"uploads/{job_id}/{f.filename or 'image.bin'}"
            bucket = storage_client.bucket(GCS_BUCKET)
            blob = bucket.blob(name)
            blob.upload_from_file(f, content_type=f.mimetype or "application/octet-stream")
            payload["image_path"] = name
        else:
            raw = f.read()
            payload["image_inline_b64"] = base64.b64encode(raw).decode()

    else:
        image_b64 = (data.get("image_b64") if data else None)
        if image_b64:
            if "," in image_b64: image_b64 = image_b64.split(",", 1)[1]
            if not LOCAL_MODE and (storage_client and GCS_BUCKET):
                raw = base64.b64decode(image_b64)
                name = f"uploads/{job_id}/image.bin"
                bucket = storage_client.bucket(GCS_BUCKET)
                blob = bucket.blob(name)
                blob.upload_from_string(raw, content_type="application/octet-stream")
                payload["image_path"] = name
            else:
                payload["image_inline_b64"] = image_b64

    payload = _prepare_async_payload(job_id, razred, user_text, requested, image_url or None,
                                     file_bytes, file_name, file_mime, image_b64_str)
    try:
        if LOCAL_MODE:
            threading.Thread(target=_local_worker, args=(payload,), daemon=True).start()
            return jsonify({"job_id": job_id, "status": "queued", "local_mode": True}), 202
        else:
            _create_task_cloud(payload)
            return jsonify({"job_id": job_id, "status": "queued", "local_mode": False}), 202
        mode_tag = "auto(sync→async)" if mode == "auto" else "sync→async"
        return jsonify({
            "mode": mode_tag, "job_id": job_id, "status": "queued", "local_mode": LOCAL_MODE,
            "reason": sync_try.get("error", "soft-timeout-or-error")
        }), 202
    except Exception as e:
        # Vrati detaljnu grešku umjesto "Internal Server Error"
        log.error("submit_async failed: %s\n%s", e, traceback.format_exc())
        log.error("submit sync→async failed: %s\n%s", e, traceback.format_exc())
        store_job(job_id, {"status": "error", "error": str(e)}, merge=True)
        return jsonify({
            "error": "submit_failed",
            "detail": str(e),
            "hint": "Lokalno koristi LOCAL_MODE=1 (bez Cloud Tasks/Firestore/GCS) ili provjeri GCP kredencijale / queue.",
            "job_id": job_id
        }), 500
        return jsonify({"error": "submit_failed", "detail": str(e), "job_id": job_id}), 500

# --------- STATUS ----------
@app.get("/status/<job_id>")
@@ -694,33 +846,45 @@
    if not data: return jsonify({"status": "pending"}), 200
    return jsonify(data), 200

# (Opcionalno) RESULT endpoint – zgodan za front da dobije samo rezultat
@app.get("/result/<job_id>")
def async_result(job_id):
    data = read_job(job_id)
    if not data:
        return jsonify({"status": "pending"}), 202
    if data.get("status") == "done":
        return jsonify({"job_id": job_id, "result": data.get("result")}), 200
    if data.get("status") == "error":
        return jsonify({"job_id": job_id, "status": "error", "error": data.get("error")}), 500
    return jsonify({"job_id": job_id, "status": data.get("status", "pending")}), 202

# --------- Cloud Tasks endpoint ----------
@app.post("/tasks/process")
def tasks_process():
    if not LOCAL_MODE and request.headers.get("X-Tasks-Secret") != TASKS_SECRET:
        return "Forbidden", 403
    try:
        payload = request.get_json(force=True)
        job_id = payload["job_id"]
        out = _process_job_core(payload)
        store_job(job_id, out, merge=True)
        try:
            log_to_sheet(job_id, out.get("razred"), out.get("user_text"), out["result"]["html"], out["result"]["path"], out["result"]["model"])
        except Exception as _e:
            log.warning("Sheets log fail: %s", _e)
        return "OK", 200
    except Exception as e:
        log.exception("Task processing failed")
        err_html = ("<p><b>Nije uspjela obrada.</b> Pokušaj ponovo ili pošalji jasniji unos.</p>"
                    f"<p><code>{html.escape(str(e))}</code></p>")
        job_id = (request.get_json(silent=True) or {}).get("job_id", f"unknown-{uuid4().hex[:6]}")
        store_job(job_id, {"status": "done", "result": {"html": err_html, "path": "error", "model": "n/a"},
                           "finished_at": datetime.datetime.utcnow().isoformat() + "Z"}, merge=True)
        return "OK", 200  # bez retrija

# ===================== Run =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    log.info("Starting app on port %s, LOCAL_MODE=%s", port, LOCAL_MODE)
    app.run(host="0.0.0.0", port=port, debug=debug)
