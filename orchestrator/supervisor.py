"""
CRRA Lab C4 — Orchestration, Human Approval & Audit Trail

The full pipeline as a LangGraph StateGraph:

    analysis → policy_check → [ hitl ] → report

analysis      fetches the contract, then a Claude tool-calling loop over the policy KB
policy_check  pure Python, no model call: decides whether a human is required
hitl          asks a person to approve or reject (only when policy_check says so)
report        writes the final status, whichever path was taken

Every node writes to logs/audit_trail.jsonl through guardrails/audit_logger.py.

Before running, from the project root:
    python data/kb_setup.py               # Lab C1, once
    python mcp_server/contract_shim.py    # Lab C2, leave running in another tab
    ANTHROPIC_API_KEY in .env

Run:
    python orchestrator/supervisor.py                       # the lab portfolio
    python orchestrator/supervisor.py CTR-1010 CTR-1012     # any contracts you name
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, TypedDict

# Project root on the path so `guardrails` and `agents` import when run as
# python orchestrator/supervisor.py
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic
from langgraph.graph import END, START, StateGraph

from agents.renewal_agent import (MAX_ROUNDS, MODEL, TOOLS, check_api, describe_tool_result,
                                  first_text, get_contract, no_decision, search_policy)
from guardrails.audit_logger import AuditLogger

PORTFOLIO = [
    "CTR-1010",  # Figma: band A, healthy, no triggers -> should finish with no prompt
    "CTR-1012",  # PagerDuty: band B, inside notice window -> approve it (y)
    "CTR-1006",  # Lucidchart: owner UNASSIGNED -> reject it (n)
    "CTR-1005",  # New Relic: auto-renews, notice deadline approaching (Step 6 trigger)
]

audit = AuditLogger()
_client: Optional[anthropic.Anthropic] = None


class ContractState(TypedDict, total=False):
    contract_id: str
    contract: dict
    # analysis
    recommendation: str
    confidence: str
    rationale: str
    policy_citation: str
    estimated_annual_impact_inr: int
    # policy_check
    hitl_required: bool
    hitl_reason: str
    # hitl
    hitl_approved: bool
    approver: str
    # report
    final_status: str


def banner(title: str) -> None:
    print(f"\n▶ {title}")


# --------------------------------------------------------------------------- #
# Nodes. Each one receives the whole state and returns the fields it adds;
# LangGraph merges them into the state, so nothing is passed between nodes by hand.
# --------------------------------------------------------------------------- #

# Only these two tools here: analysis_node fetches the contract itself and hands it over
ANALYSIS_TOOLS = [t for t in TOOLS if t["name"] in ("search_policy", "submit_recommendation")]

ANALYSIS_PROMPT = """You are the renewal analyst for Zensar BizOps. For the contract you are given, recommend exactly one of:
- RENEW: healthy contract at a sensible price; keep as is.
- RENEGOTIATE: still needed, but the terms are wrong (typically an uplift above policy benchmarks).
- CONSOLIDATE: the capability is still needed, but another existing vendor can cover it.
- TERMINATE: nobody needs the capability at all. This is a much stronger claim than CONSOLIDATE.

How to work:
1. The contract record is in the first message; it is the only contract data you have.
2. Call search_policy for each factor that drives the decision (uplift, utilisation, notice window, approval band, ownership). Make independent searches together in the same turn.
3. You have at most 5 turns. Call submit_recommendation once. Cite only sections search_policy actually returned, as '<file>.md §<heading>', and make sure the cited text supports the call.

Facts about this environment:
- Today is 2025-04-01. Use notice_state and days_to_notice_deadline from the record; never compute dates yourself.
- An owner of UNASSIGNED means nobody has been asked, not that nobody needs the tool. A termination needs the business owner's written confirmation, so without an owner a TERMINATE can never be HIGH confidence.

LOW confidence is a valid, useful answer. If the data or policy does not support a confident call, submit your best recommendation with LOW confidence and say what a human should check."""


def run_analysis_loop(contract: dict) -> dict:
    """Anthropic tool-calling loop: search_policy + submit_recommendation, capped at MAX_ROUNDS."""
    cid = contract["contract_id"]
    messages = [{"role": "user", "content": (
        f"Analyse this contract and submit a recommendation.\n\n{json.dumps(contract, indent=2)}")}]
    last_text = ""

    for round_no in range(1, MAX_ROUNDS + 1):
        response = _client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=ANALYSIS_PROMPT,
            tools=ANALYSIS_TOOLS,
            messages=messages,
            output_config={"effort": "medium"},  # no temperature: this model rejects it
        )
        # Keep the full content (thinking blocks included) for the next turn
        messages.append({"role": "assistant", "content": response.content})
        last_text = first_text(response) or last_text

        if response.stop_reason == "refusal":
            return no_decision(cid, "The model declined this request.")
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            return no_decision(cid, last_text or f"Analysis ended without a recommendation "
                                                 f"(stop_reason {response.stop_reason}).")

        results = []
        for tu in tool_uses:
            if tu.name == "submit_recommendation":
                return dict(tu.input)
            if tu.name == "search_policy":
                out = search_policy(tu.input["query"])
                print(describe_tool_result(tu.name, tu.input, out))
                content, is_error = json.dumps(out), "error" in out
            else:
                content, is_error = f"Unknown tool {tu.name}", True
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": content, "is_error": is_error})

        if round_no == MAX_ROUNDS - 1:
            # Forced tool_choice can't be combined with thinking, so ask in words instead
            results.append({"type": "text", "text": (
                "This is your final turn. Call submit_recommendation now with what you have; "
                "LOW confidence is acceptable.")})
        messages.append({"role": "user", "content": results})

    return no_decision(cid, f"Round cap ({MAX_ROUNDS}) reached without a submission.")


def analysis_node(state: ContractState) -> dict:
    cid = state["contract_id"]
    print("\n" + "═" * 62 + f"\nANALYSING: {cid}\n" + "═" * 62)
    # 1. Fetch the contract from the Lab C2 API (http://localhost:5001/api/contracts/<id>)
    contract = get_contract(cid)
    if "error" in contract:
        rec = no_decision(cid, contract["error"])
    else:
        print(describe_tool_result("get_contract", {}, contract))
        # 2. Tool-calling loop over the policy KB
        try:
            rec = run_analysis_loop(contract)
        except (anthropic.AuthenticationError, anthropic.NotFoundError):
            raise  # a bad key or model id stops the whole run, see main()
        except anthropic.APIStatusError as e:
            rec = no_decision(cid, f"Anthropic API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError:
            rec = no_decision(cid, "Could not reach the Anthropic API.")

    audit.log(cid, "AnalysisAgent", "recommendation",
              f"{rec['recommendation']} ({rec['confidence']}) — {rec['policy_citation']}",
              model=MODEL, recommendation=rec["recommendation"], confidence=rec["confidence"],
              rationale=rec["rationale"], policy_citation=rec["policy_citation"],
              estimated_annual_impact_inr=rec["estimated_annual_impact_inr"],
              # The model's own view, kept for the record only: policy_check decides
              model_said_human_required=rec["human_approval_required"])

    return {
        "contract": contract,
        "recommendation": rec["recommendation"],
        "confidence": rec["confidence"],
        "rationale": rec["rationale"],
        "policy_citation": rec["policy_citation"],
        "estimated_annual_impact_inr": rec["estimated_annual_impact_inr"],
    }


def policy_check_node(state: ContractState) -> dict:
    """Rules, not judgement: no model call here. Reasons accumulate, never short-circuit."""
    banner("POLICY CHECK")
    c = state["contract"]
    rec = state["recommendation"]
    reasons = []

    if rec == "NONE":
        reasons.append("no recommendation was produced — " + state["rationale"][:200])
    else:
        # 1. Value band (approval_thresholds.md §Human approval is mandatory)
        if c["approval_band"] in ("B", "C"):
            reasons.append(f"approval band {c['approval_band']} requires a named human approver")
        # 2. Notice window (auto_renewal_rules.md §Standard notice windows)
        if c["notice_state"] == "INSIDE_WINDOW":
            reasons.append("inside the notice window — leverage already lost")
        # 3. Exits (termination_procedure.md §Required approvals)
        if rec == "TERMINATE":
            reasons.append("TERMINATE recommended — the business owner must confirm "
                           "in writing that the capability is no longer needed")
        # 4. The agent itself is unsure
        if state["confidence"] == "LOW":
            reasons.append("analysis confidence is LOW — the agent itself is unsure")
        # 5. Nobody has confirmed the facts
        if c.get("owner", "").strip().upper() in ("", "UNASSIGNED"):
            reasons.append("no business owner on record — nobody has confirmed the facts")
        # 6. Step 6: an auto-renewing contract whose notice deadline is coming up is the
        #    one case where doing nothing has a permanent consequence
        #    (auto_renewal_rules.md §Why auto-renewal is a risk, §Standard notice windows)
        if c["auto_renew"] and c["notice_state"] == "APPROACHING":
            reasons.append(
                f"auto-renews and the notice deadline is {c['days_to_notice_deadline']} days "
                f"away ({c['notice_deadline']}) — if nobody acts, it renews for another full "
                f"term at the proposed {c['proposed_uplift_pct']}% uplift and leverage is gone")

    hitl_required = bool(reasons)
    print(f"HITL required: {hitl_required}")
    for r in reasons:
        print(f"  · {r}")

    detail = f"{len(reasons)} trigger(s): " + "; ".join(reasons) if reasons else "no policy trigger"
    audit.log(state["contract_id"], "PolicyCheck", "evaluate_triggers", detail,
              hitl_required=hitl_required, reasons=reasons,
              approval_band=c.get("approval_band"), notice_state=c.get("notice_state"),
              auto_renew=c.get("auto_renew"), owner=c.get("owner"))
    return {"hitl_required": hitl_required, "hitl_reason": "\n".join(reasons)}


def ask(prompt: str, valid=None) -> str:
    while True:
        answer = input(prompt).strip()
        if answer and (valid is None or answer.lower() in valid):
            return answer


def hitl_node(state: ContractState) -> dict:
    banner("HUMAN APPROVAL GATE")
    c = state["contract"]
    print(f"Contract:        {state['contract_id']}  {c['vendor']}  "
          f"(band {c['approval_band']}, INR {c['annual_value_inr']:,}, {c['notice_state']})")
    print(f"Recommendation:  {state['recommendation']}   confidence {state['confidence']}")
    print(f"Policy:          {state['policy_citation']}")
    print(f"Est. impact:     INR {state['estimated_annual_impact_inr']:,} per year")
    print(f"Rationale:       {state['rationale']}")
    print("Why you are being asked:")
    for r in state["hitl_reason"].splitlines():
        print(f"  · {r}")

    try:
        answer = ask("Approve this recommendation? [y/n]: ", {"y", "n", "yes", "no"})
        approved = answer.lower().startswith("y")
        # A rejection is a decision too, so it carries a name as well
        approver = ask("Your name (recorded in the audit trail): ")
    except EOFError:
        # No one at the keyboard: never treat silence as approval
        approved, approver = False, "(no response)"

    audit.log(state["contract_id"], approver, "approval_decision",
              f"{'APPROVED' if approved else 'REJECTED'} {state['recommendation']}",
              recommendation=state["recommendation"], approved=approved,
              reasons=state["hitl_reason"].splitlines())
    return {"hitl_approved": approved, "approver": approver}


def report_node(state: ContractState) -> dict:
    banner("REPORT")
    rec = state["recommendation"]
    if rec == "NONE":
        status = "NO_DECISION"
        note = "No recommendation could be produced. Needs manual analysis."
    elif not state["hitl_required"]:
        status = f"{rec}_AUTO"
        note = "Actioned without human approval (no policy trigger)."
    elif state["hitl_approved"]:
        status = f"{rec}_APPROVED"
        note = f"Approved by {state['approver']}."
    else:
        status = "ON_HOLD_REJECTED"
        note = f"Rejected by {state['approver']}. No commitment made to the vendor."

    print(f"FINAL STATUS: {status}")
    print(note)
    audit.log(state["contract_id"], "Report", "final_status", f"{status} — {note}",
              final_status=status, recommendation=rec,
              hitl_required=state["hitl_required"], approver=state.get("approver"))
    return {"final_status": status}


def route_after_policy_check(state: ContractState) -> str:
    # NONE has nothing to approve, so it goes straight to the report as NO_DECISION
    if state["hitl_required"] and state["recommendation"] != "NONE":
        return "hitl"
    return "report"


def build_graph():
    graph = StateGraph(ContractState)
    graph.add_node("analysis", analysis_node)
    graph.add_node("policy_check", policy_check_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "analysis")
    graph.add_edge("analysis", "policy_check")
    graph.add_conditional_edges("policy_check", route_after_policy_check,
                                {"hitl": "hitl", "report": "report"})
    graph.add_edge("hitl", "report")
    graph.add_edge("report", END)
    return graph.compile()


def print_summary(results: list[ContractState]) -> None:
    print("\n" + "═" * 90 + "\nPORTFOLIO SUMMARY\n" + "═" * 90)
    print(f"{'Contract':<10} {'Vendor':<22} {'Rec':<12} {'Conf':<7} {'Gate':<5} "
          f"{'Decided by':<16} Final status")
    print("─" * 90)
    for s in results:
        decided = s.get("approver") or ("rules" if not s["hitl_required"] else "-")
        print(f"{s['contract_id']:<10} {s['contract'].get('vendor', '-')[:21]:<22} "
              f"{s['recommendation']:<12} {s['confidence']:<7} "
              f"{'yes' if s['hitl_required'] else 'no':<5} {decided[:15]:<16} {s['final_status']}")
    print(f"\nAudit trail: {audit.path}  (run {audit.run_id})")


def main() -> None:
    global _client
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Put it in .env or set it in this terminal.")
    check_api()
    _client = anthropic.Anthropic()
    app = build_graph()

    portfolio = [c.upper() for c in sys.argv[1:]] or PORTFOLIO
    results = []
    for i, cid in enumerate(portfolio):
        if i:
            print("\n── next contract ──")
        try:
            results.append(app.invoke({"contract_id": cid}))
        except anthropic.AuthenticationError:
            sys.exit("The API key was rejected. Check ANTHROPIC_API_KEY.")
        except anthropic.NotFoundError:
            sys.exit(f"Model '{MODEL}' was not found for this API key.")
    print_summary(results)


if __name__ == "__main__":
    main()
