"""
CRRA Lab C3 — Renewal Analysis Agent

Looks at one contract at a time and decides RENEW, RENEGOTIATE, CONSOLIDATE or
TERMINATE, citing the procurement policy section behind the call. Real numbers
come from the Lab C2 API and policy text from the Lab C1 knowledge base, through
tool calls, so the model never has to guess either.

Before running, from the project root:
    python data/kb_setup.py               # Lab C1, once
    python mcp_server/contract_shim.py    # Lab C2, leave running in another tab
    ANTHROPIC_API_KEY in the environment or in .env

Run:
    python agents/renewal_agent.py                 # the five lab contracts
    python agents/renewal_agent.py CTR-1009 CTR-9999   # any contracts you name
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_BASE = "http://localhost:5001"
DB_DIR = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "crra_policy"
MODEL = "claude-opus-5"
MAX_ROUNDS = 5          # hard cap on model calls per contract: no infinite search loops
HTTP_TIMEOUT = 5        # seconds

# Box-drawing characters crash on a cp1252 stream (e.g. PowerShell redirect to a file)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #

class ApiUnreachable(Exception):
    pass


def _api_get(path: str) -> requests.Response:
    try:
        return requests.get(f"{API_BASE}{path}", timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        raise ApiUnreachable(
            f"The contract API at {API_BASE} is unreachable ({type(e).__name__}). "
            "Start it with: python mcp_server/contract_shim.py"
        ) from e


def get_contract(contract_id: str) -> dict:
    try:
        resp = _api_get(f"/api/contracts/{contract_id}")
    except ApiUnreachable as e:
        return {"error": str(e)}
    if resp.status_code == 404:
        return {"error": f"Contract {contract_id} does not exist. Do not invent data for it."}
    if not resp.ok:
        return {"error": f"Contract API returned HTTP {resp.status_code}: {resp.text[:200]}"}
    return resp.json()


_collection = None


def _policy_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def search_policy(query: str) -> dict:
    """Best section per source file, for the top 2 files."""
    try:
        collection = _policy_collection()
    except Exception as e:
        return {"error": f"Policy KB not available at {DB_DIR} ({e}). Run: python data/kb_setup.py"}

    res = collection.query(query_texts=[query], n_results=min(10, collection.count()))
    best_per_file: dict[str, dict] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        confidence = round(1 - dist, 2)  # cosine; ~0.25-0.6 is normal for a correct match
        source = meta["source"]
        if source not in best_per_file or confidence > best_per_file[source]["confidence"]:
            # Strip the heading line that kb_setup.py prepends to each chunk
            text = doc.split("\n", 1)[1] if "\n" in doc else doc
            best_per_file[source] = {"source": source, "section": meta["heading"],
                                     "confidence": confidence, "text": text}
    top = sorted(best_per_file.values(), key=lambda h: -h["confidence"])[:2]
    return {"query": query, "results": top}


def find_category_overlap(category: str) -> dict:
    try:
        resp = _api_get("/api/categories")
    except ApiUnreachable as e:
        return {"error": str(e)}
    if not resp.ok:
        return {"error": f"Contract API returned HTTP {resp.status_code}"}
    for entry in resp.json()["categories"]:
        if entry["category"].lower() == category.lower():
            return entry
    return {"category": category, "vendor_count": 0, "vendors": [],
            "note": "No contracts found in this category."}


TOOL_FUNCS = {
    "get_contract": lambda i: get_contract(i["contract_id"]),
    "search_policy": lambda i: search_policy(i["query"]),
    "find_category_overlap": lambda i: find_category_overlap(i["category"]),
}


# --------------------------------------------------------------------------- #
# Tool definitions shown to the model
# --------------------------------------------------------------------------- #

TOOLS = [
    {
        "name": "get_contract",
        "description": (
            "Fetch one contract from the contract API: vendor, category, business_unit, owner, "
            "annual_value_inr, approval_band (A/B/C), renewal_date, notice_deadline, "
            "notice_state (OPEN/APPROACHING/INSIDE_WINDOW/EXPIRED), days_to_notice_deadline, "
            "auto_renew, seats, utilisation_pct (null for AMC/support plans) and "
            "proposed_uplift_pct. Returns an 'error' field if the contract does not exist "
            "or the API is down."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"contract_id": {"type": "string", "description": "e.g. CTR-1004"}},
            "required": ["contract_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search the BizOps procurement policy knowledge base. Returns the best-matching "
            "section from each of the top 2 policy files, with source file, section heading, "
            "confidence (0-1; 0.3-0.6 is a normal strong match) and the section text. "
            "Phrase the query as the situation, e.g. 'vendor proposing 22 percent increase'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_category_overlap",
        "description": (
            "List every contract in a category (vendor, annual value, utilisation) so you can "
            "see whether another existing vendor could cover the same capability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "description": "e.g. Observability"}},
            "required": ["category"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Record the final recommendation for this contract. Call exactly once.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string"},
                "recommendation": {
                    "type": "string",
                    "enum": ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"],
                },
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "rationale": {
                    "type": "string",
                    "description": "3-5 sentences tying the contract's numbers to the policy rule.",
                },
                "policy_citation": {
                    "type": "string",
                    "description": "Exactly as returned by search_policy: '<file>.md §<section heading>'. "
                                   "Separate several with '; '.",
                },
                "estimated_annual_impact_inr": {
                    "type": "integer",
                    "description": "Estimated annual saving in INR (0 if none; never the full contract "
                                   "value for CONSOLIDATE).",
                },
                "human_approval_required": {"type": "boolean"},
            },
            "required": ["contract_id", "recommendation", "confidence", "rationale",
                         "policy_citation", "estimated_annual_impact_inr",
                         "human_approval_required"],
            "additionalProperties": False,
        },
    },
]

SYSTEM_PROMPT = """You are the renewal analyst for Zensar BizOps. For one contract you recommend exactly one of:
- RENEW: healthy contract at a sensible price; keep as is.
- RENEGOTIATE: still needed, but the terms are wrong (typically an uplift above policy benchmarks).
- CONSOLIDATE: the capability is still needed, but another existing vendor can cover it.
- TERMINATE: nobody needs the capability at all. This is a much stronger claim than CONSOLIDATE.

How to work:
1. Call get_contract first. If it returns an error, do NOT submit a recommendation: explain in one sentence that you cannot analyse a contract you have not seen, and stop.
2. Search the policy for each factor that drives the decision (uplift, utilisation and overlap, notice window, approval band, ownership). Use find_category_overlap before any CONSOLIDATE or TERMINATE call.
3. You have at most 5 turns. Make independent tool calls together in the same turn rather than one per turn.
4. Call submit_recommendation once. Cite only sections that search_policy actually returned, as '<file>.md §<heading>', and make sure the cited text supports the call.

Facts about this environment:
- Today is 2025-04-01. Use notice_state and days_to_notice_deadline from the API; never compute dates yourself.
- An owner of UNASSIGNED means nobody has been asked, not that nobody needs the tool. Read the termination policy carefully: a termination needs the business owner's written confirmation, so without an owner a TERMINATE can never be HIGH confidence or go ahead without a human.
- Set human_approval_required to true whenever the policy you retrieved requires a named human approver for this contract or this action.

LOW confidence is a valid, useful answer. If the data or policy does not support a confident call, submit your best recommendation with LOW confidence and say what a human should check, rather than searching again and again."""


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #

def first_text(response) -> str:
    """First block with a .text attribute; a thinking block may come before it."""
    for block in response.content:
        if getattr(block, "text", None):
            return block.text
    return ""


def describe_tool_result(name: str, tool_input: dict, result: dict) -> str:
    if "error" in result:
        return f"  → {name}: ERROR {result['error']}"
    if name == "get_contract":
        util = result.get("utilisation_pct")
        util_s = f"{util:.0f}%" if util is not None else "n/a"
        return (f"  → contract: band {result['approval_band']} {result['notice_state']} "
                f"util {util_s} uplift {result['proposed_uplift_pct']}% "
                f"INR {result['annual_value_inr']:,}  owner {result['owner']}")
    if name == "search_policy":
        hits = "; ".join(f"{h['source']} §{h['section']} ({h['confidence']:.0%})"
                         for h in result["results"])
        return f'  → policy "{tool_input["query"]}" -> {hits}'
    if name == "find_category_overlap":
        vendors = ", ".join(
            f"{v['vendor']} ({v['utilisation_pct'] if v['utilisation_pct'] is not None else 'n/a'}%)"
            for v in result.get("vendors", []))
        return f"  → overlap {result['category']}: {vendors or 'none'}"
    return f"  → {name}"


def no_decision(contract_id: str, reason: str) -> dict:
    return {"contract_id": contract_id, "recommendation": "NONE", "confidence": "LOW",
            "rationale": reason, "policy_citation": "-", "estimated_annual_impact_inr": 0,
            "human_approval_required": True}


def analyse_contract(client: anthropic.Anthropic, contract_id: str) -> dict:
    print("\n" + "═" * 62 + f"\nANALYSING: {contract_id}\n" + "═" * 62)
    messages = [{"role": "user", "content": f"Analyse contract {contract_id} and submit a recommendation."}]
    recommendation = None
    last_text = ""

    for round_no in range(1, MAX_ROUNDS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "medium"},
        )
        # Keep the full content (thinking blocks included) so the next turn has its reasoning
        messages.append({"role": "assistant", "content": response.content})
        last_text = first_text(response) or last_text

        if response.stop_reason == "refusal":
            return no_decision(contract_id, "The model declined this request.")

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            if response.stop_reason == "max_tokens":
                return no_decision(contract_id, "Response hit max_tokens before a decision.")
            # Model stopped without submitting (e.g. contract not found): respect that
            print(f"  (no recommendation) {last_text[:300]}")
            return no_decision(contract_id, last_text or "Agent ended without a recommendation.")

        results = []
        for tu in tool_uses:
            if tu.name == "submit_recommendation":
                recommendation = dict(tu.input)
                content, is_error = "Recommendation recorded.", False
            elif tu.name in TOOL_FUNCS:
                out = TOOL_FUNCS[tu.name](tu.input)
                print(describe_tool_result(tu.name, tu.input, out))
                content, is_error = json.dumps(out), "error" in out
            else:
                content, is_error = f"Unknown tool {tu.name}", True
            results.append({"type": "tool_result", "tool_use_id": tu.id,
                            "content": content, "is_error": is_error})

        if recommendation is not None:
            return recommendation

        if round_no == MAX_ROUNDS - 1:
            # Forced tool_choice can't be combined with thinking, so ask in words instead
            results.append({"type": "text", "text": (
                "This is your final turn. Call submit_recommendation now with what you have; "
                "LOW confidence is acceptable.")})
        messages.append({"role": "user", "content": results})

    return no_decision(contract_id, f"Round cap ({MAX_ROUNDS}) reached without a submission.")


def print_recommendation(rec: dict) -> None:
    print("  ┌─ RECOMMENDATION " + "─" * 40)
    print(f"  │ {rec['recommendation']}   confidence {rec['confidence']}")
    print(f"  │ Policy: {rec['policy_citation']}")
    print(f"  │ Human approval required: {rec['human_approval_required']}")
    print(f"  │ Est. annual impact: INR {rec['estimated_annual_impact_inr']:,}")
    print("  └" + "─" * 57)
    print(f"  {rec['rationale']}")


def print_summary(results: list[dict]) -> None:
    print("\n" + "═" * 100 + "\nSUMMARY\n" + "═" * 100)
    print(f"{'Contract':<10} {'Recommendation':<14} {'Conf':<7} {'Human':<6} "
          f"{'Impact INR':>12}  Policy citation")
    print("─" * 100)
    for r in results:
        print(f"{r['contract_id']:<10} {r['recommendation']:<14} {r['confidence']:<7} "
              f"{str(r['human_approval_required']):<6} {r['estimated_annual_impact_inr']:>12,}  "
              f"{r['policy_citation'][:50]}")


def check_api() -> None:
    try:
        health = _api_get("/health").json()
    except ApiUnreachable as e:
        sys.exit(f"\nCannot start: {e}")
    print(f"Contract API OK: {health['contracts_loaded']} contracts, "
          f"simulated today {health['simulated_today']}")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Put it in .env or set it in this terminal.")
    check_api()
    client = anthropic.Anthropic()

    test_contracts = sys.argv[1:] or [
        "CTR-1003",  # Microsoft 365: high utilisation, modest uplift
        "CTR-1012",  # PagerDuty: 22% uplift
        "CTR-1005",  # New Relic: 29% utilisation, Datadog in same category
        "CTR-1006",  # Lucidchart: 14% utilisation, owner UNASSIGNED
        "CTR-1004",  # Datadog: band C, inside notice window, 18% uplift
    ]

    results = []
    for cid in test_contracts:
        try:
            rec = analyse_contract(client, cid.upper())
        except anthropic.AuthenticationError:
            sys.exit("The API key was rejected. Check ANTHROPIC_API_KEY.")
        except anthropic.NotFoundError:
            sys.exit(f"Model '{MODEL}' was not found for this API key.")
        except anthropic.APIConnectionError:
            sys.exit("Could not reach the Anthropic API. Check your internet connection or proxy.")
        except anthropic.APIStatusError as e:
            rec = no_decision(cid, f"Anthropic API error {e.status_code}: {e.message}")
        if rec["recommendation"] != "NONE":
            print_recommendation(rec)
        results.append(rec)

    print_summary(results)


if __name__ == "__main__":
    main()
