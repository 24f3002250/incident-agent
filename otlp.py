"""Build the OTLP/JSON trace purely from stored run state (no model, no new IDs)."""

KIND_INTERNAL, KIND_SERVER, KIND_CLIENT = 1, 2, 3


def _attr(k, v):
    if isinstance(v, bool):
        return {"key": k, "value": {"boolValue": v}}
    if isinstance(v, int):
        return {"key": k, "value": {"intValue": v}}
    if isinstance(v, float):
        return {"key": k, "value": {"doubleValue": v}}
    return {"key": k, "value": {"stringValue": str(v)}}


def _span(st, span_id, parent, name, kind, start, end, attrs, status=None, links=None):
    base = {"ga5.run.id": st["runId"], "ga5.public.marker": st["publicMarker"]}
    base.update(attrs)
    s = {
        "traceId": st["traceId"],
        "spanId": span_id,
        "name": name,
        "kind": kind,
        "startTimeUnixNano": str(int(start)),
        "endTimeUnixNano": str(int(max(end, start))),
        "attributes": [_attr(k, v) for k, v in base.items() if v is not None],
        "status": status or {"code": 0},
    }
    if parent:
        s["parentSpanId"] = parent
    if st.get("traceState"):
        s["traceState"] = st["traceState"]
    if links:
        s["links"] = links
    return s


def build_otlp(st):
    spans = []
    last = st.get("lastEventAt") or st.get("respondedAt") or st["receivedAt"]
    agent_end = st.get("finishedAt") or last
    responded = st.get("respondedAt") or st["chat"]["end"]

    spans.append(_span(
        st, st["serverSpanId"], st.get("incomingParentSpanId"), "POST /v2/incidents", KIND_SERVER,
        st["receivedAt"], responded,
        {"http.request.method": "POST", "http.route": "/v2/incidents",
         "url.path": "/v2/incidents", "http.response.status_code": 200},
    ))
    spans.append(_span(
        st, st["agentSpanId"], st["serverSpanId"], f"invoke_agent {st['agentName']}", KIND_INTERNAL,
        st["receivedAt"], agent_end,
        {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": st["agentName"],
         "ga5.run.status": st["status"]},
    ))
    chat = st["chat"]
    chat_attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": chat.get("model") or "unknown-model",
        "gen_ai.provider.name": chat.get("provider") or "openai",
        "gen_ai.response.model": chat.get("responseModel"),
        "gen_ai.usage.input_tokens": chat.get("inputTokens"),
        "gen_ai.usage.output_tokens": chat.get("outputTokens"),
        "ga5.plan.source": chat.get("source"),
    }
    chat_status = {"code": 0}
    if chat.get("error"):
        chat_attrs["error.type"] = chat["error"]
        chat_status = {"code": 2}
    spans.append(_span(st, st["chatSpanId"], st["agentSpanId"], "chat incident-plan", KIND_CLIENT,
                       chat["start"], chat["end"], chat_attrs, chat_status))

    diag_links = []
    diag_end = 0
    for a in st["actions"]:
        if not a["attempts"]:
            continue
        first = a["attempts"][0]
        last_at = a["attempts"][-1]
        a_end = last_at["receivedAt"] or last_at["dispatchedAt"]
        lattrs = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": a["toolName"],
            "gen_ai.tool.call.id": a["callId"],
            "gen_ai.tool.type": "function",
            "ga5.action.id": a["actionId"],
            "ga5.action.phase": a["phase"],
        }
        if a.get("approvalId"):
            lattrs["ga5.approval.id"] = a["approvalId"]
        lstatus = {"code": 0}
        if a["state"] == "failed":
            rec = last_at["receipt"] or {}
            lattrs["error.type"] = "timeout" if a.get("failReason") == "timeout" else str(rec.get("status"))
            lstatus = {"code": 2}
        elif a["state"] == "succeeded":
            lstatus = {"code": 1}
        spans.append(_span(st, a["logicalSpanId"], st["agentSpanId"], f"execute_tool {a['toolName']}",
                           KIND_INTERNAL, first["dispatchedAt"], a_end, lattrs, lstatus))
        if a["phase"] == "diagnostic":
            diag_links.append({"traceId": st["traceId"], "spanId": a["logicalSpanId"],
                               "attributes": [_attr("ga5.action.id", a["actionId"])]})
            diag_end = max(diag_end, a_end)
        for at in a["attempts"]:
            rec = at["receipt"]
            cattrs = {
                "ga5.action.id": a["actionId"],
                "ga5.attempt": at["attempt"],
                "gen_ai.tool.name": a["toolName"],
                "gen_ai.tool.call.id": a["callId"],
                "http.request.method": "POST",
                "http.request.resend_count": at["attempt"] - 1,
                "url.template": f"tool/{a['toolName']}",
            }
            cstatus = {"code": 0}
            if rec is not None:
                cattrs["ga5.receipt.id"] = rec["receiptId"]
                cattrs["ga5.receipt.nonce"] = rec["nonce"]
                code = rec["status"]
                if code and code > 0:
                    cattrs["http.response.status_code"] = code
                if code == 0 or rec.get("errorType") == "timeout":
                    cattrs["error.type"] = "timeout"
                    cstatus = {"code": 2}
                elif not (200 <= code < 300):
                    cattrs["error.type"] = str(code)
                    cstatus = {"code": 2}
                else:
                    cstatus = {"code": 1}
            spans.append(_span(st, at["spanId"], a["logicalSpanId"], f"POST tool/{a['toolName']}",
                               KIND_CLIENT, at["dispatchedAt"], at["receivedAt"] or at["dispatchedAt"],
                               cattrs, cstatus))

    if len(diag_links) > 1:
        spans.append(_span(st, st["joinSpanId"], st["agentSpanId"], "incident.join", KIND_INTERNAL,
                           diag_end or st["chat"]["end"], diag_end or st["chat"]["end"],
                           {"ga5.join.count": len(diag_links)}, None, diag_links))

    for ap in st["approvals"]:
        gattrs = {"ga5.approval.id": ap["approvalId"], "ga5.action.id": ap["actionId"],
                  "gen_ai.tool.name": ap["toolName"]}
        if ap["decision"] is not None:
            gattrs.update({"ga5.approval.decision": ap["decision"],
                           "ga5.approval.nonce": ap["nonce"],
                           "ga5.receipt.id": ap["receiptId"],
                           "ga5.receipt.nonce": ap["nonce"]})
        spans.append(_span(st, ap["gateSpanId"], st["agentSpanId"], "approval_gate", KIND_INTERNAL,
                           ap["requestedAt"], ap["decidedAt"] or ap["requestedAt"], gattrs))

    return {"resourceSpans": [{
        "resource": {"attributes": [_attr("service.name", "ga5-incident-agent"),
                                    _attr("ga5.run.id", st["runId"]),
                                    _attr("ga5.public.marker", st["publicMarker"])]},
        "scopeSpans": [{"scope": {"name": "ga5-incident-agent", "version": "2.0.0"},
                        "spans": spans}],
    }]}
