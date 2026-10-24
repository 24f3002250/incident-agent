"""GA5 incident agent (profile ga5-incident-agent/v2).

Layers are kept separate:
  * planner.py  - the single LLM planning call (diagnosis + diagnostics + effect plan)
  * this file   - HTTP API, validation, persistence, receipt-driven state machine
  * otlp.py     - trace construction purely from stored state
Replays read stored state only; receipts/GET/replay never call the model.
"""
import asyncio
import copy
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from otlp import build_otlp
from planner import plan_incident, canonical_json

PROFILE = "ga5-incident-agent/v2"
DB_PATH = os.environ.get("STATE_DB", "/tmp/ga5_incident_state.sqlite3")

app = FastAPI(title="GA5 incident agent")

# ---------------------------------------------------------------- persistence
_db_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute(
    "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, body_hash TEXT, "
    "initial_response TEXT, state TEXT)"
)
_conn.execute(
    "CREATE TABLE IF NOT EXISTS receipts (run_id TEXT, receipt_id TEXT, body_hash TEXT, "
    "response TEXT, PRIMARY KEY (run_id, receipt_id))"
)
_conn.execute("CREATE TABLE IF NOT EXISTS plan_cache (k TEXT PRIMARY KEY, plan TEXT)")
_conn.commit()


def db_get_run(run_id):
    with _db_lock:
        row = _conn.execute(
            "SELECT body_hash, initial_response, state FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
    if not row:
        return None
    return {"body_hash": row[0], "initial_response": json.loads(row[1]), "state": json.loads(row[2])}


def db_put_run(run_id, body_hash, initial_response, state):
    with _db_lock:
        _conn.execute(
            "INSERT INTO runs VALUES (?,?,?,?)",
            (run_id, body_hash, json.dumps(initial_response), json.dumps(state)),
        )
        _conn.commit()


def db_get_receipt(run_id, receipt_id):
    with _db_lock:
        row = _conn.execute(
            "SELECT body_hash, response FROM receipts WHERE run_id=? AND receipt_id=?",
            (run_id, receipt_id),
        ).fetchone()
    return None if not row else {"body_hash": row[0], "response": json.loads(row[1])}


def db_apply_receipt(run_id, receipt_id, body_hash, response, state):
    """Atomically store the new run state together with the receipt response."""
    with _db_lock:
        _conn.execute("UPDATE runs SET state=? WHERE run_id=?", (json.dumps(state), run_id))
        _conn.execute(
            "INSERT INTO receipts VALUES (?,?,?,?)",
            (run_id, receipt_id, body_hash, json.dumps(response)),
        )
        _conn.commit()


def cache_get(k):
    with _db_lock:
        row = _conn.execute("SELECT plan FROM plan_cache WHERE k=?", (k,)).fetchone()
    return json.loads(row[0]) if row else None


def cache_put(k, plan):
    with _db_lock:
        _conn.execute("INSERT OR REPLACE INTO plan_cache VALUES (?,?)", (k, json.dumps(plan)))
        _conn.commit()


_run_locks = {}
_run_locks_guard = threading.Lock()


def run_lock(run_id):
    with _run_locks_guard:
        if run_id not in _run_locks:
            _run_locks[run_id] = asyncio.Lock()
        return _run_locks[run_id]


# ---------------------------------------------------------------- helpers
def now_ns():
    return time.time_ns()


def hex_id(nbytes):
    while True:
        v = secrets.token_hex(nbytes)
        if int(v, 16) != 0:
            return v


def body_hash(obj):
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def err(status, msg):
    # Never echo request content in errors.
    return JSONResponse(status_code=status, content={"error": msg})


TP_RE = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


def parse_traceparent(tp):
    if not isinstance(tp, str):
        return None
    m = TP_RE.match(tp.strip())
    if not m:
        return None
    ver, tid, pid, flags = m.groups()
    if ver == "ff" or int(tid, 16) == 0 or int(pid, 16) == 0:
        return None
    if ver == "00" and len(tp.strip()) != 55:
        return None
    return {"traceId": tid, "parentSpanId": pid, "flags": flags}


def collect_secrets(sensitive, do_not_export, raw_body):
    vals = set()

    def walk(v):
        if isinstance(v, str):
            if len(v) >= 4:
                vals.add(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(sensitive)
    # doNotExport may name paths (e.g. "sensitive.accessToken") or literal values.
    for item in do_not_export or []:
        if not isinstance(item, str):
            continue
        cur = raw_body
        ok = True
        for part in item.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not raw_body:
            # A named path: its value is secret (but never treat the whole transcript etc. as a
            # string to scrub from outputs if it is huge; it is never exported anyway).
            if isinstance(cur, str) and len(cur) < 2000:
                walk(cur)
            elif isinstance(cur, (dict, list)) and item.startswith("sensitive"):
                walk(cur)
    return sorted(vals, key=len, reverse=True)


def scrub(obj, secrets_list):
    if not secrets_list:
        return obj
    if isinstance(obj, str):
        for s in secrets_list:
            if s in obj:
                obj = obj.replace(s, "[REDACTED]")
        return obj
    if isinstance(obj, dict):
        return {k: scrub(v, secrets_list) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, secrets_list) for v in obj]
    return obj


def digest_args(args):
    return hashlib.sha256(canonical_json(args).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- validation
def validate_incident_body(b):
    if not isinstance(b, dict):
        return "body must be a JSON object"
    if b.get("profile") != PROFILE:
        return "unsupported profile"
    rid = b.get("runId")
    if not isinstance(rid, str) or not rid.strip() or len(rid) > 256:
        return "runId must be a nonempty string"
    inc = b.get("incident")
    if not isinstance(inc, dict):
        return "incident must be an object"
    if not isinstance(inc.get("transcript"), str) or not inc.get("transcript").strip():
        return "incident.transcript must be a nonempty string"
    arc = inc.get("allowedRootCauses")
    if not isinstance(arc, list) or not arc or not all(isinstance(x, str) and x for x in arc):
        return "incident.allowedRootCauses must be a nonempty string array"
    cat = b.get("toolCatalog")
    if not isinstance(cat, list) or not cat:
        return "toolCatalog must be a nonempty array"
    for t in cat:
        if not isinstance(t, dict) or not isinstance(t.get("name"), str) or not t["name"]:
            return "toolCatalog entries need a name"
    pol = b.get("policy")
    if not isinstance(pol, dict):
        return "policy must be an object"
    if "effectTools" in pol and not isinstance(pol["effectTools"], list):
        return "policy.effectTools must be an array"
    if "approvalRequiredFor" in pol and not isinstance(pol["approvalRequiredFor"], list):
        return "policy.approvalRequiredFor must be an array"
    if "sensitive" in b and not isinstance(b["sensitive"], dict):
        return "sensitive must be an object"
    for k in ("agentName", "publicMarker"):
        if k in b and not isinstance(b[k], str):
            return f"{k} must be a string"
    return None


# ---------------------------------------------------------------- state machine
def new_action(st, phase, tool, arguments, evidence):
    n = len(st["actions"]) + 1
    base = hashlib.sha256(f"{st['runId']}|{n}|{secrets.token_hex(4)}".encode()).hexdigest()
    a = {
        "actionId": f"act-{base[:16]}",
        "callId": f"call-{base[16:32]}",
        "phase": phase,
        "toolName": tool,
        "arguments": arguments,
        "evidence": evidence,
        "logicalSpanId": hex_id(8),
        "attempts": [],
        "state": "pending",
        "approvalId": None,
        "approvalNonce": None,
    }
    st["actions"].append(a)
    return a


def dispatch_attempt(st, a):
    attempt_no = len(a["attempts"]) + 1
    span_id = hex_id(8)
    d = {
        "actionId": a["actionId"],
        "callId": a["callId"],
        "phase": a["phase"],
        "toolName": a["toolName"],
        "arguments": a["arguments"],
        "evidence": a["evidence"],
        "attempt": attempt_no,
        "traceparent": f"00-{st['traceId']}-{span_id}-01",
    }
    if st.get("traceState"):
        d["tracestate"] = st["traceState"]
    if a.get("approvalId"):
        d["approvalId"] = a["approvalId"]
        d["approvalNonce"] = a["approvalNonce"]
    a["attempts"].append(
        {"attempt": attempt_no, "spanId": span_id, "dispatch": d, "dispatchedAt": now_ns(),
         "receipt": None, "receivedAt": None}
    )
    a["state"] = "pending"
    st["actionLog"].append(copy.deepcopy(d))
    st["lastEventAt"] = now_ns()
    return d


def pending_attempts(st):
    out = []
    for a in st["actions"]:
        if a["attempts"] and a["attempts"][-1]["receipt"] is None:
            out.append((a, a["attempts"][-1]))
    return out


def is_success(status):
    return isinstance(status, int) and 200 <= status < 300


NEGATIVE_RC = ("refut", "not_confirm", "unconfirm", "mismatch", "disprov", "rejected", "negative")
LIMIT_RC = ("limit", "max_replica", "maxed", "at_max", "capacity_cap", "ceiling")


def choose_effect(st):
    """Decide the effect tool + args after diagnostics; deterministic, no model call."""
    plan = st["plan"]
    tool = plan["effect"]["toolName"]
    args = plan["effect"]["arguments"]
    reason = None
    diag_classes = [
        (a["attempts"][-1]["receipt"] or {}).get("resultClass", "") or ""
        for a in st["actions"] if a["phase"] == "diagnostic"
    ]
    at_limit = bool(plan.get("atScalingLimit")) or any(
        any(k in c.lower() for k in LIMIT_RC) for c in diag_classes
    )
    page = plan.get("page")
    if at_limit and "scale" in tool.lower() and page and page.get("toolName"):
        tool, args, reason = page["toolName"], page["arguments"], "scaling_limit_reached"
    return tool, args, reason


def finalize(st, status):
    st["status"] = status
    st["lastEventAt"] = now_ns()
    st["finishedAt"] = st["lastEventAt"]


def suppress_effect(st, reason):
    if st.get("effectSuppressed"):
        return
    eff = st["plan"]["effect"]
    rid = hashlib.sha256(f"{st['runId']}|suppressed".encode()).hexdigest()
    st["suppressed"].append(
        {"actionId": f"act-{rid[:16]}", "toolName": eff["toolName"], "phase": "effect",
         "reason": reason}
    )
    st["effectSuppressed"] = True


def advance(st):
    """Move the run forward after receipts. Never calls the model."""
    if st["status"] != "waiting":
        return
    if pending_attempts(st):
        return
    diags = [a for a in st["actions"] if a["phase"] == "diagnostic"]
    failed = [a for a in diags if a["state"] == "failed"]
    if failed:
        suppress_effect(st, "prerequisite_timeout" if any(a.get("failReason") == "timeout" for a in failed)
                        else "prerequisite_failed")
        st["chosenEffect"] = None
        finalize(st, "failed")
        return
    if any(a["state"] == "refuted" for a in diags):
        suppress_effect(st, "diagnosis_not_confirmed")
        st["chosenEffect"] = None
        finalize(st, "failed")
        return
    effects = [a for a in st["actions"] if a["phase"] == "effect"]
    if not effects:
        if st.get("approvalDenied"):
            suppress_effect(st, "approval_denied")
            st["chosenEffect"] = None
            finalize(st, "completed")
            return
        if st.get("effectChoice") is None:
            tool, args, reason = choose_effect(st)
            st["effectChoice"] = {"toolName": tool, "arguments": args, "switchReason": reason}
        tool = st["effectChoice"]["toolName"]
        args = st["effectChoice"]["arguments"]
        needs_approval = tool in st["policy"]["approvalRequiredFor"]
        if needs_approval:
            ap = st["approvals"][0] if st["approvals"] else None
            if ap is None:
                base = hashlib.sha256(f"{st['runId']}|effect|{secrets.token_hex(4)}".encode()).hexdigest()
                st["reservedEffectActionId"] = f"act-{base[:16]}"
                st["reservedEffectCallId"] = f"call-{base[16:32]}"
                st["approvals"].append({
                    "approvalId": f"apr-{base[32:48]}",
                    "actionId": st["reservedEffectActionId"],
                    "toolName": tool,
                    "argumentsDigest": digest_args(args),
                    "requestedAt": now_ns(),
                    "decision": None, "nonce": None, "receiptId": None, "decidedAt": None,
                    "gateSpanId": hex_id(8),
                })
                st["lastEventAt"] = now_ns()
                return
            if ap["decision"] is None:
                return
            if ap["decision"] != "approved":
                suppress_effect(st, "approval_denied")
                st["chosenEffect"] = None
                finalize(st, "completed")
                return
        a = new_action(st, "effect", tool, args, list(st["plan"]["effect"]["evidence"]))
        if needs_approval:
            ap = st["approvals"][0]
            a["actionId"] = st["reservedEffectActionId"]
            a["callId"] = st["reservedEffectCallId"]
            a["approvalId"] = ap["approvalId"]
            a["approvalNonce"] = ap["nonce"]
        st["chosenEffect"] = tool
        dispatch_attempt(st, a)
        return
    eff = effects[0]
    if eff["state"] == "succeeded":
        finalize(st, "completed")
    else:
        finalize(st, "failed")


def apply_outcome(st, o, receipt_id):
    if not isinstance(o, dict):
        raise ValueError("outcome must be an object")
    aid, cid, att = o.get("actionId"), o.get("callId"), o.get("attempt")
    status, nonce = o.get("status"), o.get("nonce")
    if not isinstance(aid, str) or not isinstance(cid, str):
        raise ValueError("outcome ids missing")
    if not isinstance(att, int) or isinstance(att, bool):
        raise ValueError("attempt must be an integer")
    if not isinstance(status, int) or isinstance(status, bool):
        raise ValueError("status must be an integer")
    if not isinstance(nonce, str) or not nonce:
        raise ValueError("nonce required")
    target = None
    for a in st["actions"]:
        if a["actionId"] == aid:
            target = a
    if target is None or target["callId"] != cid:
        raise ValueError("no such pending call")
    last = target["attempts"][-1] if target["attempts"] else None
    if last is None or last["attempt"] != att or last["receipt"] is not None:
        raise ValueError("call is not pending for that attempt")
    rc = o.get("resultClass")
    entry = {"receiptId": receipt_id, "actionId": aid, "callId": cid, "attempt": att,
             "status": status, "resultClass": rc if isinstance(rc, str) else None, "nonce": nonce}
    if isinstance(o.get("errorType"), str):
        entry["errorType"] = o["errorType"]
    last["receipt"] = entry
    last["receivedAt"] = now_ns()
    st["receiptLog"].append(entry)
    st["lastEventAt"] = last["receivedAt"]
    if is_success(status):
        rcl = (rc or "").lower() if isinstance(rc, str) else ""
        if target["phase"] == "diagnostic" and any(k in rcl for k in NEGATIVE_RC):
            target["state"] = "refuted"
        else:
            target["state"] = "succeeded"
        return
    if status == 503 and att == 1:
        target["state"] = "retry"
        return
    target["state"] = "failed"
    target["failReason"] = "timeout" if (status == 0 or o.get("errorType") == "timeout") else "error"


def apply_approval(st, ap_in, receipt_id):
    if not isinstance(ap_in, dict):
        raise ValueError("approval must be an object")
    apid, decision, nonce = ap_in.get("approvalId"), ap_in.get("decision"), ap_in.get("nonce")
    if not isinstance(apid, str) or not isinstance(decision, str) or not decision:
        raise ValueError("approval fields missing")
    if not isinstance(nonce, str) or not nonce:
        raise ValueError("nonce required")
    for ap in st["approvals"]:
        if ap["approvalId"] == apid:
            if ap["decision"] is not None:
                raise ValueError("approval already decided")
            ap["decision"] = decision
            ap["nonce"] = nonce
            ap["receiptId"] = receipt_id
            ap["decidedAt"] = now_ns()
            st["receiptLog"].append(
                {"receiptId": receipt_id, "approvalId": apid, "decision": decision, "nonce": nonce}
            )
            st["lastEventAt"] = ap["decidedAt"]
            if decision != "approved":
                st["approvalDenied"] = True
            return
    raise ValueError("no such pending approval")


def process_receipt(st, body):
    """Returns new state (deep copy) or raises ValueError. Never calls the model."""
    st = copy.deepcopy(st)
    if st["status"] != "waiting":
        raise ValueError("run is already final")
    rid = body["receiptId"]
    outcomes = body.get("outcomes") or []
    approvals = body.get("approvals") or []
    if not isinstance(outcomes, list) or not isinstance(approvals, list):
        raise ValueError("outcomes/approvals must be arrays")
    if not outcomes and not approvals:
        raise ValueError("empty receipt")
    seen = set()
    for o in outcomes:
        key = (o.get("actionId"), o.get("attempt")) if isinstance(o, dict) else None
        if key in seen:
            raise ValueError("duplicate outcome")
        seen.add(key)
    for o in outcomes:
        apply_outcome(st, o, rid)
    for ap in approvals:
        apply_approval(st, ap, rid)
    # retries for 503s (exactly one), unless a sibling diagnostic already failed terminally
    any_failed = any(a["state"] in ("failed", "refuted") and a["phase"] == "diagnostic" for a in st["actions"])
    for a in st["actions"]:
        if a["state"] == "retry":
            if a["phase"] == "diagnostic" and any_failed:
                a["state"] = "failed"
                a["failReason"] = "error"
            else:
                dispatch_attempt(st, a)
    advance(st)
    return st


# ---------------------------------------------------------------- views
def view(st):
    out = {
        "runId": st["runId"],
        "status": st["status"],
        "diagnosis": st["diagnosis"],
    }
    if st["status"] == "waiting":
        out["dispatches"] = [copy.deepcopy(at["dispatch"]) for _, at in pending_attempts(st)]
        out["approvals"] = [
            {k: ap[k] for k in ("approvalId", "actionId", "toolName", "argumentsDigest")}
            for ap in st["approvals"] if ap["decision"] is None
        ]
    else:
        out["dispatches"] = []
        out["approvals"] = []
    out["chosenEffect"] = st.get("chosenEffect")
    out["suppressed"] = copy.deepcopy(st["suppressed"])
    out["actionLog"] = copy.deepcopy(st["actionLog"])
    out["receiptLog"] = copy.deepcopy(st["receiptLog"])
    out["otlp"] = build_otlp(st)
    return scrub(out, st.get("scrubList", []))


# ---------------------------------------------------------------- routes
async def read_json(request):
    raw = await request.body()
    if len(raw) > 8 * 1024 * 1024:
        return None, err(413, "body too large")
    try:
        return json.loads(raw), None
    except Exception:
        return None, err(400, "invalid JSON")


REQ_LOG = []   # (time, method, path-shape, status, ms) - no bodies, no secrets
CAPTURE = []   # recent grader exchanges for offline debugging; `sensitive` is never stored
CAPTURE_PATH = "/debug/capture-7f3k9q2x"


def capture(kind, run_id, headers, body, status, resp):
    try:
        b = copy.deepcopy(body) if isinstance(body, dict) else body
        if isinstance(b, dict) and "sensitive" in b:
            b["sensitive"] = "[omitted]"
        h = {k: v for k, v in headers.items() if k.lower() in (
            "traceparent", "tracestate", "content-type", "user-agent", "baggage")}
        h["_headerNames"] = sorted(headers.keys())
        CAPTURE.append({"at": time.strftime("%H:%M:%S"), "kind": kind, "runId": run_id,
                        "headers": h, "body": b, "status": status, "response": resp})
        del CAPTURE[:-60]
    except Exception:
        pass
RUN_LOG = []   # per-run planning summary - no transcript, no arguments


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t = time.monotonic()
    resp = await call_next(request)
    ms = int((time.monotonic() - t) * 1000)
    path = re.sub(r"/v2/incidents/[^/]+", "/v2/incidents/{runId}", request.url.path)
    REQ_LOG.append(f"{time.strftime('%H:%M:%S')} {request.method} {path} -> {resp.status_code} {ms}ms")
    del REQ_LOG[:-80]
    print("REQ", REQ_LOG[-1], flush=True)
    return resp


@app.get(CAPTURE_PATH)
async def debug_capture():
    return {"captures": CAPTURE}


@app.get("/debug")
async def debug():
    import planner as P
    return {"tokenConfigured": bool(P._api_key()), "model": P.LLM_MODEL,
            "fallbackModels": P.LLM_FALLBACK_MODELS, "baseUrl": P.LLM_BASE_URL,
            "recentRuns": RUN_LOG[-30:], "recentLlmErrors": P.LAST_LLM_ERRORS[-20:],
            "recentRequests": REQ_LOG[-80:]}


@app.get("/debug/ping-llm")
async def ping_llm():
    import planner as P
    t = time.monotonic()
    try:
        text, data = await P.call_llm([{"role": "user", "content": "Reply with the word ok"}],
                                      time.monotonic() + 15, max_tokens=5, json_mode=False)
        return {"ok": True, "model": data.get("model"), "reply": text[:20],
                "ms": int((time.monotonic() - t) * 1000)}
    except Exception as ex:
        return {"ok": False, "error": str(ex), "ms": int((time.monotonic() - t) * 1000),
                "details": P.LAST_LLM_ERRORS[-3:]}


@app.get("/")
async def root():
    return {"ok": True, "profile": PROFILE, "routes": ["POST /v2/incidents",
            "POST /v2/incidents/{runId}/receipts", "GET /v2/incidents/{runId}"]}


@app.post("/v2/incidents")
async def create_incident(request: Request):
    received_at = now_ns()
    body, e = await read_json(request)
    if e:
        return e
    problem = validate_incident_body(body)
    if problem:
        capture("create-invalid", body.get("runId") if isinstance(body, dict) else None,
                request.headers, {"keys": sorted(body) if isinstance(body, dict) else None,
                                  "profile": body.get("profile") if isinstance(body, dict) else None},
                422, {"error": problem})
        return err(422 if isinstance(body, dict) else 400, problem)
    run_id = body["runId"]
    h = body_hash(body)
    async with run_lock(run_id):
        existing = db_get_run(run_id)
        if existing:
            if existing["body_hash"] != h:
                capture("create-conflict", run_id, request.headers, {"keys": sorted(body)}, 409, None)
                return err(409, "runId already exists with different content")
            capture("create-replay", run_id, request.headers, {"keys": sorted(body)}, 200, None)
            return JSONResponse(existing["initial_response"])

        # incoming trace context (header first, then body)
        tp = request.headers.get("traceparent")
        ts = request.headers.get("tracestate")
        if tp is None:
            tc = body.get("traceContext") if isinstance(body.get("traceContext"), dict) else {}
            tp = body.get("traceparent", tc.get("traceparent"))
            ts = body.get("tracestate", tc.get("tracestate"))
        parsed = parse_traceparent(tp)
        if parsed:
            trace_id, parent_span = parsed["traceId"], parsed["parentSpanId"]
            trace_state = ts.strip() if isinstance(ts, str) and ts.strip() else None
        else:
            trace_id, parent_span, trace_state = hex_id(16), None, None

        policy = body["policy"]
        pol = {
            "maximumDiagnostics": policy.get("maximumDiagnostics", 3),
            "effectTools": [t for t in policy.get("effectTools", []) if isinstance(t, str)],
            "approvalRequiredFor": [t for t in policy.get("approvalRequiredFor", []) if isinstance(t, str)],
        }
        scrub_list = collect_secrets(body.get("sensitive") or {}, policy.get("doNotExport"), body)

        # ---- the single model call (sensitive object is never passed)
        chat_start = now_ns()
        plan, model_meta = await plan_incident(
            incident=body["incident"], catalog=body["toolCatalog"], policy=pol,
            cache_get=cache_get, cache_put=cache_put,
        )
        chat_end = now_ns()
        plan = scrub(plan, scrub_list)

        st = {
            "runId": run_id,
            "status": "waiting",
            "agentName": body.get("agentName") or "incident-response",
            "publicMarker": body.get("publicMarker") or "",
            "traceId": trace_id,
            "incomingParentSpanId": parent_span,
            "traceState": trace_state,
            "serverSpanId": hex_id(8),
            "agentSpanId": hex_id(8),
            "chatSpanId": hex_id(8),
            "joinSpanId": hex_id(8),
            "receivedAt": received_at,
            "chat": {"start": chat_start, "end": chat_end, **model_meta},
            "policy": pol,
            "diagnosis": plan["diagnosis"],
            "plan": plan,
            "actions": [], "approvals": [], "actionLog": [], "receiptLog": [],
            "suppressed": [], "chosenEffect": None, "effectChoice": None,
            "scrubList": scrub_list,
        }
        for dg in plan["diagnostics"]:
            a = new_action(st, "diagnostic", dg["toolName"], dg["arguments"], dg["evidence"])
            dispatch_attempt(st, a)
        st["respondedAt"] = now_ns()
        resp = view(st)
        db_put_run(run_id, h, resp, st)
        RUN_LOG.append({
            "at": time.strftime("%H:%M:%S"), "run": run_id[-8:],
            "incident": str(body["incident"].get("incidentId", ""))[:40],
            "source": model_meta.get("source"), "model": model_meta.get("model"),
            "error": model_meta.get("error"), "llmMs": (chat_end - chat_start) // 1_000_000,
            "totalMs": (now_ns() - received_at) // 1_000_000,
            "rootCause": plan["diagnosis"]["rootCause"],
            "diagnosticTools": [d["toolName"] for d in plan["diagnostics"]],
            "effectTool": plan["effect"]["toolName"],
            "transcriptChars": len(body["incident"]["transcript"]),
        })
        del RUN_LOG[:-60]
        print("RUN", json.dumps(RUN_LOG[-1]), flush=True)
        capture("create", run_id, request.headers, body, 200, resp)
        return JSONResponse(resp)


@app.post("/v2/incidents/{run_id}/receipts")
async def post_receipt(run_id: str, request: Request):
    body, e = await read_json(request)
    if e:
        return e
    if not isinstance(body, dict):
        return err(400, "body must be a JSON object")
    rid = body.get("receiptId")
    if not isinstance(rid, str) or not rid.strip():
        return err(422, "receiptId must be a nonempty string")
    h = body_hash(body)
    async with run_lock(run_id):
        run = db_get_run(run_id)
        if not run:
            return err(404, "unknown runId")
        prev = db_get_receipt(run_id, rid)
        if prev:
            if prev["body_hash"] != h:
                return err(409, "receiptId already used with different content")
            return JSONResponse(prev["response"])
        try:
            new_state = process_receipt(run["state"], body)
        except ValueError as ex:
            capture("receipt-invalid", run_id, request.headers, body, 422, {"error": str(ex)})
            return err(422, f"invalid state change: {ex}")
        resp = view(new_state)
        db_apply_receipt(run_id, rid, h, resp, new_state)
        capture("receipt", run_id, request.headers, body, 200, {"status": resp["status"],
                "dispatches": resp["dispatches"], "approvals": resp["approvals"]})
        return JSONResponse(resp)


@app.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str):
    run = db_get_run(run_id)
    if not run:
        return err(404, "unknown runId")
    return JSONResponse(view(run["state"]))
