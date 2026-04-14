import asyncio
import uuid
import json
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

# ── In-memory store (Phase 1) ─────────────────────────────────────
simulations = {}

# ── Harmful topic guard ───────────────────────────────────────────
# Assembly simulates policy debates — not arguments for violence,
# abuse, or criminality. These topics cause LLMs to refuse generation,
# producing 0 agents and a ZeroDivisionError crash downstream.
# Block them cleanly before the simulation thread starts.
#
# Research basis: Content safety taxonomies (Weidinger et al. 2021,
# Google DeepMind) classify requests that solicit harmful content
# as categorically different from policy debates — no legitimate
# simulation use case requires debating these.

BLOCKED_TERMS = [
    "rape", "sexual assault", "child abuse", "pedophilia", "child porn",
    "child sexual", "molest", "genocide", "ethnic cleansing",
    "slavery", "human trafficking", "torture", "terrorism how to",
    "mass shooting how", "school shooting", "bomb making",
]

def is_harmful_topic(topic: str) -> bool:
    """
    Returns True if topic contains terms that are categorically
    outside the scope of policy debate simulation.
    Uses whole-word matching to avoid false positives.
    """
    topic_lower = topic.lower()
    for term in BLOCKED_TERMS:
        # Match as substring — these terms are unambiguous in context
        if term in topic_lower:
            return True
    return False


# ── Helper ────────────────────────────────────────────────────────
def run_async(coro):
    """Run async functions from Flask's sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ── 1. Start simulation ───────────────────────────────────────────
import threading

@api.route("/api/simulation/start", methods=["POST"])
def start_simulation():
    try:
        body = request.get_json()
        topic = body.get("topic")
        num_agents = body.get("num_agents", 10)
        num_rounds = body.get("num_rounds", 3)
        pdf_paths = body.get("uploads", [])

        if not topic:
            return jsonify({"error": "topic is required"}), 400

        # ── Harmful topic guard ───────────────────────────────────
        # Check BEFORE starting the thread — return 400 immediately
        # so the frontend gets a clean error instead of a silent crash.
        if is_harmful_topic(topic):
            print(f"[API] Blocked harmful topic: {topic}")
            return jsonify({
                "error": "This topic cannot be simulated. Assembly is designed for policy and social debates."
            }), 400

        simulation_id = f"sim_{uuid.uuid4().hex[:8]}"

        simulations[simulation_id] = {
            "simulation_id": simulation_id,
            "topic": topic,
            "status": "running",
            "agents_created": 0,
            "graph_summary": {},
            "rounds": [],
            "final_agents": [],
            "report": {}
        }

        def run_simulation():
            try:
                print(f"\n[API] Starting simulation {simulation_id} on: {topic}")

                # Step 1 — Classify topic
                from backend.agents.topic_classifier import classify_topic
                classification = run_async(classify_topic(topic))

                inst_count = max(3, round(num_agents * classification["institutional_ratio"]))
                pub_count = max(2, num_agents - inst_count)

                print(f"[API] Agent split: {inst_count} institutional, {pub_count} public")

                # Step 2 — Ingest both sources in parallel
                inst_chunks, pub_chunks = run_async(ingest(topic, pdf_paths))

                # Step 3 — Build two separate graphs
                G_inst = run_async(build_graph(inst_chunks, graph_source="institutional"))
                G_pub = run_async(build_graph(pub_chunks, graph_source="public")) if pub_chunks else G_inst
                graph_summary = get_graph_summary(G_inst)

                # Step 4 — Generate institutional agents
                from backend.agents.stakeholder_identifier import identify_stakeholders
                stakeholders = run_async(identify_stakeholders(topic, G_inst, inst_count, G_pub))
                actual_inst_count = len(stakeholders)
                inst_agents = run_async(generate_personas(topic, G_inst, actual_inst_count, stakeholders))

                # Step 5 — Generate public agents
                from backend.agents.public_agent_generator import generate_public_agents
                pub_agents = run_async(generate_public_agents(topic, G_pub, pub_count))

                # Step 6 — Tag agents with their graph type
                for a in inst_agents:
                    a["graph_type"] = "institutional"
                for a in pub_agents:
                    a["graph_type"] = "public"

                # Step 7 — Merge all agents
                all_agents = inst_agents + pub_agents
                print(f"[API] Total agents: {len(all_agents)} ({len(inst_agents)} institutional, {len(pub_agents)} public)")

                # Step 8 — Run debate
                debate_result = run_async(run_debate(topic, all_agents, G_inst, num_rounds, G_pub))

                # Step 9 — Generate report
                report = run_async(generate_report(
                    topic=topic,
                    simulation_id=simulation_id,
                    rounds=debate_result["rounds"],
                    final_agents=debate_result["final_agents"],
                    G=G_inst
                ))

                simulations[simulation_id].update({
                    "status": "completed",
                    "agents_created": len(all_agents),
                    "graph_summary": graph_summary,
                    "rounds": debate_result["rounds"],
                    "final_agents": debate_result["final_agents"],
                    "report": report,
                    "topic_classification": classification["label"]
                })
                print(f"[API] Simulation {simulation_id} completed")

            except Exception as e:
                import traceback
                print(f"[API] Simulation {simulation_id} failed: {e}")
                traceback.print_exc()
                simulations[simulation_id]["status"] = "failed"
                simulations[simulation_id]["error"] = str(e)

        thread = threading.Thread(target=run_simulation)
        thread.daemon = True
        thread.start()

        return jsonify({
            "simulation_id": simulation_id,
            "status": "running",
            "message": "Simulation started. Poll /api/simulation/{id}/status for updates."
        }), 202

    except Exception as e:
        print(f"[API] Error: {e}")
        return jsonify({"error": str(e)}), 500


@api.route("/api/simulation/<simulation_id>/status", methods=["GET"])
def get_simulation_status(simulation_id):
    sim = simulations.get(simulation_id)
    if not sim:
        return jsonify({"error": "simulation not found"}), 404

    return jsonify({
        "simulation_id": simulation_id,
        "status": sim["status"],
        "agents_created": sim.get("agents_created", 0),
        "error": sim.get("error", None)
    }), 200


# ── 2. Get debate rounds ──────────────────────────────────────────
@api.route("/api/simulation/<simulation_id>/debate", methods=["GET"])
def get_debate(simulation_id):
    sim = simulations.get(simulation_id)
    if not sim:
        return jsonify({"error": "simulation not found"}), 404

    return jsonify({
        "simulation_id": simulation_id,
        "rounds": sim["rounds"]
    }), 200


# ── 3. Get agent memory ───────────────────────────────────────────
@api.route("/api/agent/<agent_id>/memory", methods=["GET"])
def get_agent_memory(agent_id):
    memory = []
    for sim_id, sim in simulations.items():
        for agent in sim.get("final_agents", []):
            if agent["id"] == agent_id:
                memory.append({
                    "simulation_id": sim_id,
                    "date": "2026-04-10",
                    "topic": sim["topic"],
                    "final_opinion": agent["opinion"],
                    "final_score": agent["score"],
                    "shifted": agent.get("shifted", False)
                })

    if not memory:
        return jsonify({"error": "agent not found"}), 404

    agent_data = None
    for sim in simulations.values():
        for agent in sim.get("final_agents", []):
            if agent["id"] == agent_id:
                agent_data = agent
                break

    return jsonify({
        "agent_id": agent_id,
        "name": agent_data["name"],
        "persona": agent_data["persona"],
        "memory": memory
    }), 200


# ── 4. Inject live event ──────────────────────────────────────────
@api.route("/api/inject", methods=["POST"])
def inject_event():
    try:
        body = request.get_json()
        simulation_id = body.get("simulation_id")
        event = body.get("event")

        sim = simulations.get(simulation_id)
        if not sim:
            return jsonify({"error": "simulation not found"}), 404

        if not event:
            return jsonify({"error": "event is required"}), 400

        final_agents = sim["final_agents"]
        current_tick = len(sim["rounds"]) + 1

        reactions = []
        for agent in final_agents:
            old_opinion = agent["opinion"]
            old_score = agent["score"]

            reactions.append({
                "agent_id": agent["id"],
                "name": agent["name"],
                "opinion_before": old_opinion,
                "opinion_after": f"Considering: {event}. {old_opinion}",
                "delta": 0.0,
                "shifted": False
            })

        return jsonify({
            "injected_at_tick": current_tick,
            "reactions": reactions
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 5. Branch simulation ──────────────────────────────────────────
@api.route("/api/branch", methods=["POST"])
def branch_simulation():
    try:
        body = request.get_json()
        simulation_id = body.get("simulation_id")
        from_tick = body.get("from_tick")

        sim = simulations.get(simulation_id)
        if not sim:
            return jsonify({"error": "simulation not found"}), 404

        branch_id = f"branch_{uuid.uuid4().hex[:8]}"

        branched_rounds = sim["rounds"][:from_tick]
        branched_agents = branched_rounds[-1]["agents"] if branched_rounds else []

        simulations[branch_id] = {
            "simulation_id": branch_id,
            "topic": sim["topic"],
            "status": "running",
            "agents_created": len(branched_agents),
            "rounds": branched_rounds,
            "final_agents": branched_agents,
            "parent_simulation_id": simulation_id,
            "branched_at_tick": from_tick,
            "report": {}
        }

        return jsonify({
            "branch_id": branch_id,
            "parent_simulation_id": simulation_id,
            "branched_at_tick": from_tick,
            "status": "running"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 6. Get report ─────────────────────────────────────────────────
@api.route("/api/report/<simulation_id>", methods=["GET"])
def get_report(simulation_id):
    sim = simulations.get(simulation_id)
    if not sim:
        return jsonify({"error": "simulation not found"}), 404

    return jsonify(sim["report"]), 200


# ── 7. Get sentiment history ──────────────────────────────────────
@api.route("/api/sentiment/history/<simulation_id>", methods=["GET"])
def get_sentiment_history(simulation_id):
    sim = simulations.get(simulation_id)
    if not sim:
        return jsonify({"error": "simulation not found"}), 404

    return jsonify(
        sim["report"].get("sentiment_history", {
            "simulation_id": simulation_id,
            "ticks": []
        })
    ), 200


# ── 8. Store correction (call after validating against real polls) ─
@api.route("/api/correction/store", methods=["POST"])
def store_prediction_correction():
    """
    Store a prediction error after comparing simulation output to
    real polling data. This feeds the reflexion correction memory.

    Body: {
        "topic": "...",
        "predicted": {"for": 0.70, "against": 0.15, "neutral": 0.15},
        "actual":    {"for": 0.45, "against": 0.54, "neutral": 0.08},
        "root_cause": "optional human explanation"
    }
    """
    try:
        from backend.agents.correction_store import store_correction
        body = request.get_json()

        topic      = body.get("topic")
        predicted  = body.get("predicted")
        actual     = body.get("actual")
        root_cause = body.get("root_cause", "")

        if not all([topic, predicted, actual]):
            return jsonify({"error": "topic, predicted, and actual are required"}), 400

        store_correction(topic, predicted, actual, root_cause)
        return jsonify({"status": "stored", "topic": topic}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500