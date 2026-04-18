import asyncio
import uuid
import json
import pathlib
import threading
from flask import Blueprint, request, jsonify
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.ingestion.ingestor import ingest
from backend.ingestion.graph_builder import build_graph, get_graph_summary
from backend.agents.persona_generator import generate_personas
from backend.agents.debate_engine import run_debate
from backend.report.report_agent import generate_report
from backend.agents.stakeholder_identifier import identify_stakeholders

api = Blueprint("api", __name__)

# ── Tier 1 in-memory store ────────────────────────────────────────
simulations = {}

# ── Tier 2 persistence ───────────────────────────────────────────
# /tmp survives within a worker session but NOT across workers.
# GODMODE FIX: Every GET reloads from disk so any worker can serve any sim.
SIM_STORE = pathlib.Path("/tmp/dtc_simulations.json")
_store_lock = threading.Lock()  # Prevent concurrent write corruption

def _load_dtc_sims():
    try:
        return json.loads(SIM_STORE.read_text())
    except Exception:
        return {}

def _save_dtc_sims():
    with _store_lock:
        try:
            SIM_STORE.write_text(json.dumps(dtc_simulations, default=str))
        except Exception as e:
            print(f"[API] Failed to persist simulations: {e}")

def _fresh_sim(simulation_id: str) -> dict | None:
    """
    GODMODE FIX: Always reload from disk before returning sim.
    Prevents multi-worker 404 — any gunicorn worker can serve any simulation.
    """
    # Check in-memory first (fast path)
    if simulation_id in dtc_simulations:
        return dtc_simulations[simulation_id]
    # Reload from disk (slow path — handles cross-worker requests)
    dtc_simulations.update(_load_dtc_sims())
    return dtc_simulations.get(simulation_id)

# Load on startup
dtc_simulations = _load_dtc_sims()

# ── Harmful topic guard ───────────────────────────────────────────
BLOCKED_TERMS = [
    "rape", "sexual assault", "child abuse", "pedophilia", "child porn",
    "child sexual", "molest", "genocide", "ethnic cleansing",
    "slavery", "human trafficking", "torture", "terrorism how to",
    "mass shooting how", "school shooting", "bomb making",
]

ERROR_MESSAGES = {
    "thin_data": "We couldn't find enough real-world data on this topic. Try a broader or more well-known question.",
    "no_agents": "Assembly couldn't generate stakeholders for this topic. Try rephrasing your question or adding more context.",
    "harmful":   "This topic cannot be simulated. Assembly is designed for policy and social debates.",
    "generic":   "Something went wrong with this simulation. Please try again or rephrase your question.",
}

def is_harmful_topic(topic: str) -> bool:
    topic_lower = topic.lower()
    for term in BLOCKED_TERMS:
        if term in topic_lower:
            return True
    return False

def classify_error(error: Exception, context: dict = None) -> str:
    error_str = str(error).lower()
    if "division by zero" in error_str or "zerodivision" in error_str:
        return "no_agents"
    if "list index out of range" in error_str or "index out of range" in error_str:
        return "no_agents"
    if context:
        if context.get("inst_chunks", 1) < 5 and context.get("pub_chunks", 1) < 5:
            return "thin_data"
        if context.get("total_agents", 1) == 0:
            return "no_agents"
    return "generic"


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ══════════════════════════════════════════════════════════════════
# TIER 1 ROUTES
# ══════════════════════════════════════════════════════════════════

@api.route("/api/simulation/start", methods=["POST"])
def start_simulation():
    try:
        body       = request.get_json()
        topic      = body.get("topic")
        num_agents = body.get("num_agents", 10)
        num_rounds = body.get("num_rounds", 3)
        pdf_paths  = body.get("uploads", [])
        context    = body.get("context", "")

        if not topic:
            return jsonify({"error": "topic is required"}), 400
        if is_harmful_topic(topic):
            return jsonify({"error": ERROR_MESSAGES["harmful"]}), 400

        simulation_id = f"sim_{uuid.uuid4().hex[:8]}"
        simulations[simulation_id] = {
            "simulation_id": simulation_id, "topic": topic, "context": context,
            "status": "running", "agents_created": 0, "graph_summary": {},
            "rounds": [], "final_agents": [], "report": {},
            "error": None, "error_message": None,
        }

        def run_simulation():
            error_context = {}
            try:
                print(f"\n[API] Starting simulation {simulation_id} on: {topic}")
                if context:
                    print(f"[API] Context: {context[:100]}...")

                from backend.agents.topic_classifier import classify_topic
                classification = run_async(classify_topic(topic, context=context))
                inst_count = min(6, max(3, round(num_agents * classification["institutional_ratio"])))
                pub_count  = num_agents - inst_count
                print(f"[API] Agent split: {inst_count} institutional, {pub_count} public")

                inst_chunks, pub_chunks = run_async(ingest(topic, pdf_paths, context=context))
                error_context["inst_chunks"] = len(inst_chunks)
                error_context["pub_chunks"]  = len(pub_chunks)

                if len(inst_chunks) < 5 and len(pub_chunks) < 5:
                    simulations[simulation_id]["status"] = "failed"
                    simulations[simulation_id]["error"] = "thin_data"
                    simulations[simulation_id]["error_message"] = ERROR_MESSAGES["thin_data"]
                    return

                G_inst = run_async(build_graph(inst_chunks, graph_source="institutional"))
                G_pub  = run_async(build_graph(pub_chunks, graph_source="public")) if pub_chunks else G_inst
                graph_summary = get_graph_summary(G_inst)

                from backend.agents.stakeholder_identifier import identify_stakeholders
                stakeholders, keyword_signal = run_async(identify_stakeholders(
                    topic, G_inst, inst_count, G_pub, pub_chunks=pub_chunks, context=context
                ))
                inst_agents = run_async(generate_personas(
                    topic, G_inst, len(stakeholders), stakeholders, context=context
                ))

                from backend.agents.public_agent_generator import generate_public_agents
                pub_agents = run_async(generate_public_agents(
                    topic, G_pub, pub_count, keyword_signal=keyword_signal, context=context
                ))

                for a in inst_agents: a["graph_type"] = "institutional"
                for a in pub_agents:  a["graph_type"] = "public"

                all_agents = inst_agents + pub_agents
                error_context["total_agents"] = len(all_agents)
                print(f"[API] Total agents: {len(all_agents)}")

                if len(all_agents) == 0:
                    simulations[simulation_id]["status"] = "failed"
                    simulations[simulation_id]["error"] = "no_agents"
                    simulations[simulation_id]["error_message"] = ERROR_MESSAGES["no_agents"]
                    return

                debate_result = run_async(run_debate(topic, all_agents, G_inst, num_rounds, G_pub))
                report = run_async(generate_report(
                    topic=topic, simulation_id=simulation_id,
                    rounds=debate_result["rounds"],
                    final_agents=debate_result["final_agents"], G=G_inst
                ))

                simulations[simulation_id].update({
                    "status": "completed", "agents_created": len(all_agents),
                    "graph_summary": graph_summary,
                    "rounds": debate_result["rounds"],
                    "final_agents": debate_result["final_agents"],
                    "report": report,
                    "topic_classification": classification["label"],
                })
                print(f"[API] Simulation {simulation_id} completed")

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_key = classify_error(e, error_context)
                simulations[simulation_id]["status"] = "failed"
                simulations[simulation_id]["error"] = error_key
                simulations[simulation_id]["error_message"] = ERROR_MESSAGES[error_key]

        threading.Thread(target=run_simulation, daemon=True).start()
        return jsonify({
            "simulation_id": simulation_id, "status": "running",
            "message": "Simulation started."
        }), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route("/api/simulation/<simulation_id>/status", methods=["GET"])
def get_simulation_status(simulation_id):
    sim = simulations.get(simulation_id)
    if not sim:
        return jsonify({"error": "simulation not found"}), 404
    return jsonify({
        "simulation_id": simulation_id, "status": sim["status"],
        "agents_created": sim.get("agents_created", 0),
        "error": sim.get("error"), "error_message": sim.get("error_message"),
    }), 200


@api.route("/api/simulation/<simulation_id>/debate", methods=["GET"])
def get_debate(simulation_id):
    sim = simulations.get(simulation_id)
    if not sim:
        return jsonify({"error": "simulation not found"}), 404
    return jsonify({"simulation_id": simulation_id, "rounds": sim["rounds"]}), 200


@api.route("/api/agent/<agent_id>/memory", methods=["GET"])
def get_agent_memory(agent_id):
    memory = []
    for sim_id, sim in simulations.items():
        for agent in sim.get("final_agents", []):
            if agent["id"] == agent_id:
                memory.append({
                    "simulation_id": sim_id, "date": "2026-04-10",
                    "topic": sim["topic"], "final_opinion": agent["opinion"],
                    "final_score": agent["score"], "shifted": agent.get("shifted", False)
                })
    if not memory:
        return jsonify({"error": "agent not found"}), 404
    agent_data = next(
        (a for sim in simulations.values()
         for a in sim.get("final_agents", []) if a["id"] == agent_id), None
    )
    return jsonify({
        "agent_id": agent_id, "name": agent_data["name"],
        "persona": agent_data["persona"], "memory": memory
    }), 200


@api.route("/api/inject", methods=["POST"])
def inject_event():
    try:
        body = request.get_json()
        sim  = simulations.get(body.get("simulation_id"))
        if not sim: return jsonify({"error": "simulation not found"}), 404
        event = body.get("event")
        if not event: return jsonify({"error": "event is required"}), 400
        reactions = [{
            "agent_id": a["id"], "name": a["name"],
            "opinion_before": a["opinion"],
            "opinion_after": f"Considering: {event}. {a['opinion']}",
            "delta": 0.0, "shifted": False
        } for a in sim["final_agents"]]
        return jsonify({"injected_at_tick": len(sim["rounds"]) + 1, "reactions": reactions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route("/api/branch", methods=["POST"])
def branch_simulation():
    try:
        body = request.get_json()
        sim  = simulations.get(body.get("simulation_id"))
        if not sim: return jsonify({"error": "simulation not found"}), 404
        from_tick = body.get("from_tick")
        branch_id = f"branch_{uuid.uuid4().hex[:8]}"
        branched_rounds  = sim["rounds"][:from_tick]
        branched_agents  = branched_rounds[-1]["agents"] if branched_rounds else []
        simulations[branch_id] = {
            "simulation_id": branch_id, "topic": sim["topic"],
            "status": "running", "agents_created": len(branched_agents),
            "rounds": branched_rounds, "final_agents": branched_agents,
            "parent_simulation_id": body.get("simulation_id"),
            "branched_at_tick": from_tick, "report": {}
        }
        return jsonify({
            "branch_id": branch_id, "parent_simulation_id": body.get("simulation_id"),
            "branched_at_tick": from_tick, "status": "running"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route("/api/report/<simulation_id>", methods=["GET"])
def get_report(simulation_id):
    sim = simulations.get(simulation_id)
    if not sim: return jsonify({"error": "simulation not found"}), 404
    return jsonify(sim["report"]), 200


@api.route("/api/sentiment/history/<simulation_id>", methods=["GET"])
def get_sentiment_history(simulation_id):
    sim = simulations.get(simulation_id)
    if not sim: return jsonify({"error": "simulation not found"}), 404
    return jsonify(sim["report"].get("sentiment_history", {
        "simulation_id": simulation_id, "ticks": []
    })), 200


@api.route("/api/correction/store", methods=["POST"])
def store_prediction_correction():
    try:
        from backend.agents.correction_store import store_correction
        body = request.get_json()
        topic, predicted, actual = body.get("topic"), body.get("predicted"), body.get("actual")
        if not all([topic, predicted, actual]):
            return jsonify({"error": "topic, predicted, and actual are required"}), 400
        store_correction(topic, predicted, actual, body.get("root_cause", ""))
        return jsonify({"status": "stored", "topic": topic}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
# TIER 2 DTC ROUTES — GODMODE
# ══════════════════════════════════════════════════════════════════

def _parse_dtc_body(body: dict) -> dict:
    errors = []
    product_name = body.get("product_name", "").strip()
    if not product_name: errors.append("product_name is required")
    description = body.get("product_description", "").strip()
    if not description: errors.append("product_description is required")
    try:
        price = float(body.get("price", 0))
        if price <= 0: errors.append("price must be greater than 0")
    except (TypeError, ValueError):
        errors.append("price must be a number")
        price = 0.0
    if errors:
        return {"errors": errors}
    raw_competitors = body.get("competitors", [])
    competitors = []
    for c in raw_competitors:
        if isinstance(c, str) and c.strip():
            competitors.append({"name": c.strip(), "asin": ""})
        elif isinstance(c, dict) and c.get("name", "").strip():
            competitors.append({"name": c["name"].strip(), "asin": c.get("asin", "").strip()})
    return {
        "product_name": product_name, "description": description,
        "price": price, "category": body.get("category", "general"),
        "demographic": body.get("demographic", ""),
        "competitors": competitors,
        "num_agents": int(body.get("num_agents", 6)),
        "errors": [],
    }


@api.route("/api/dtc/simulation/start", methods=["POST"])
def dtc_start_simulation():
    try:
        body = request.get_json()
        if not body:
            return jsonify({"error": "Request body required"}), 400
        parsed = _parse_dtc_body(body)
        if parsed.get("errors"):
            return jsonify({"error": " | ".join(parsed["errors"])}), 400

        simulation_id = f"sim_dtc_{uuid.uuid4().hex[:8]}"
        dtc_simulations[simulation_id] = {
            "simulation_id": simulation_id, "status": "running",
            "product_name": parsed["product_name"], "price": parsed["price"],
            "agents_created": 0, "rounds": [], "report": {}, "error": None,
        }
        _save_dtc_sims()

        def run_dtc_pipeline():
            try:
                from backend.dtc.dtc_ingestor import run_market_ingestion, ProductBrief
                from backend.dtc.buyer_persona_generator import generate_buyer_personas
                from backend.dtc.market_debate_engine import run_market_debate
                from backend.dtc.market_report_agent import generate_market_report

                product = ProductBrief(
                    name=parsed["product_name"], description=parsed["description"],
                    price=parsed["price"], category=parsed["category"],
                    demographic=parsed["demographic"], competitors=parsed["competitors"],
                )
                num_agents = max(4, min(12, parsed["num_agents"]))
                print(f"\n[API] DTC {simulation_id} starting: {product.name} @ ${product.price}")

                intel = run_async(run_market_ingestion(product, num_agents))
                dtc_simulations[simulation_id]["intel_complete"] = True
                _save_dtc_sims()

                agents = run_async(generate_buyer_personas(intel, num_agents))
                dtc_simulations[simulation_id]["agents_created"] = len(agents)
                _save_dtc_sims()

                debate = run_async(run_market_debate(agents, intel, simulation_id))
                dtc_simulations[simulation_id]["rounds"] = [
                    {"round": r.round, "agents": r.agents} for r in debate.rounds
                ]
                _save_dtc_sims()

                report = run_async(generate_market_report(intel, debate, simulation_id))
                dtc_simulations[simulation_id]["report"]  = report
                dtc_simulations[simulation_id]["status"]  = "complete"
                _save_dtc_sims()

                print(f"[API] DTC {simulation_id} complete ✓")

            except Exception as e:
                import traceback
                traceback.print_exc()
                dtc_simulations[simulation_id]["status"] = "failed"
                dtc_simulations[simulation_id]["error"]  = str(e)
                _save_dtc_sims()

        threading.Thread(target=run_dtc_pipeline, daemon=True).start()
        return jsonify({
            "simulation_id": simulation_id, "status": "running",
            "message": f"DTC simulation started for {parsed['product_name']}",
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api.route("/api/dtc/simulation/<simulation_id>/status", methods=["GET"])
def dtc_get_status(simulation_id):
    # GODMODE FIX: _fresh_sim checks disk if not in memory — beats multi-worker 404
    sim = _fresh_sim(simulation_id)
    if not sim:
        return jsonify({"error": "Simulation not found"}), 404
    return jsonify({
        "simulation_id":  simulation_id,
        "status":         sim.get("status", "running"),
        "agents_created": sim.get("agents_created", 0),
        "error":          sim.get("error"),
        "error_message":  sim.get("error"),
    }), 200


@api.route("/api/dtc/simulation/<simulation_id>/debate", methods=["GET"])
def dtc_get_debate(simulation_id):
    # GODMODE FIX: _fresh_sim checks disk if not in memory
    sim = _fresh_sim(simulation_id)
    if not sim:
        return jsonify({"error": "Simulation not found"}), 404
    return jsonify({
        "simulation_id": simulation_id,
        "rounds":        sim.get("rounds", []),
    }), 200


@api.route("/api/dtc/simulation/<simulation_id>/report", methods=["GET"])
def dtc_get_report(simulation_id):
    # GODMODE FIX: _fresh_sim checks disk if not in memory
    sim = _fresh_sim(simulation_id)
    if not sim:
        return jsonify({"error": "Simulation not found"}), 404
    if sim.get("status") != "complete":
        return jsonify({
            "error":  "Report not ready",
            "status": sim.get("status", "running"),
        }), 404
    report = sim.get("report", {})
    if not report:
        return jsonify({"error": "Report generation failed"}), 500
    return jsonify(report), 200


@api.route("/api/dtc/health", methods=["GET"])
def dtc_health():
    return jsonify({
        "status":      "ok",
        "mode":        "dtc",
        "simulations": len(_load_dtc_sims()),  # always reflects disk state
    }), 200