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

simulations = {}

BLOCKED_TERMS = [
    "rape", "sexual assault", "child abuse", "pedophilia", "child porn",
    "child sexual", "molest", "genocide", "ethnic cleansing",
    "slavery", "human trafficking", "torture", "terrorism how to",
    "mass shooting how", "school shooting", "bomb making",
]

# ── Human-readable error messages ────────────────────────────────
# Maps internal failure causes to messages users actually understand.
# Shown in the frontend when status="failed".
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
    """
    Map an exception to a human-readable error category.
    Returns a key from ERROR_MESSAGES.
    """
    error_str = str(error).lower()

    if "division by zero" in error_str or "zerodivision" in error_str:
        return "no_agents"

    if "list index out of range" in error_str or "index out of range" in error_str:
        return "no_agents"

    if context:
        inst_chunks = context.get("inst_chunks", 1)
        pub_chunks  = context.get("pub_chunks", 1)
        if inst_chunks < 5 and pub_chunks < 5:
            return "thin_data"

        total_agents = context.get("total_agents", 1)
        if total_agents == 0:
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


import threading

@api.route("/api/simulation/start", methods=["POST"])
def start_simulation():
    try:
        body = request.get_json()
        topic     = body.get("topic")
        num_agents = body.get("num_agents", 10)
        num_rounds = body.get("num_rounds", 3)
        pdf_paths  = body.get("uploads", [])
        context    = body.get("context", "")

        if not topic:
            return jsonify({"error": "topic is required"}), 400

        if is_harmful_topic(topic):
            print(f"[API] Blocked harmful topic: {topic}")
            return jsonify({"error": ERROR_MESSAGES["harmful"]}), 400

        simulation_id = f"sim_{uuid.uuid4().hex[:8]}"

        simulations[simulation_id] = {
            "simulation_id": simulation_id,
            "topic": topic,
            "context": context,
            "status": "running",
            "agents_created": 0,
            "graph_summary": {},
            "rounds": [],
            "final_agents": [],
            "report": {},
            "error": None,
            "error_message": None,
        }

        def run_simulation():
            error_context = {}

            try:
                print(f"\n[API] Starting simulation {simulation_id} on: {topic}")
                if context:
                    print(f"[API] Context provided: {context[:100]}...")

                # Step 1 — Classify topic
                from backend.agents.topic_classifier import classify_topic
                classification = run_async(classify_topic(topic))

                inst_count = max(3, round(num_agents * classification["institutional_ratio"]))
                pub_count  = max(2, num_agents - inst_count)
                print(f"[API] Agent split: {inst_count} institutional, {pub_count} public")

                # Step 2 — Ingest
                inst_chunks, pub_chunks = run_async(ingest(topic, pdf_paths, context=context))

                # ── Thin data guard ───────────────────────────────
                # If Tavily finds almost nothing, fail early with a
                # clean message instead of crashing downstream.
                error_context["inst_chunks"] = len(inst_chunks)
                error_context["pub_chunks"]  = len(pub_chunks)

                if len(inst_chunks) < 5 and len(pub_chunks) < 5:
                    print(f"[API] Thin data detected — only {len(inst_chunks)} inst + {len(pub_chunks)} pub chunks")
                    simulations[simulation_id]["status"] = "failed"
                    simulations[simulation_id]["error"] = "thin_data"
                    simulations[simulation_id]["error_message"] = ERROR_MESSAGES["thin_data"]
                    return

                # Step 3 — Build graphs
                G_inst = run_async(build_graph(inst_chunks, graph_source="institutional"))
                G_pub  = run_async(build_graph(pub_chunks, graph_source="public")) if pub_chunks else G_inst
                graph_summary = get_graph_summary(G_inst)

                # Step 4 — Identify stakeholders
                from backend.agents.stakeholder_identifier import identify_stakeholders
                stakeholders, keyword_signal = run_async(identify_stakeholders(
                    topic, G_inst, inst_count, G_pub,
                    pub_chunks=pub_chunks,
                    context=context
                ))
                actual_inst_count = len(stakeholders)
                inst_agents = run_async(generate_personas(
                    topic, G_inst, actual_inst_count, stakeholders,
                    context=context
                ))

                # Step 5 — Public agents
                from backend.agents.public_agent_generator import generate_public_agents
                pub_agents = run_async(generate_public_agents(
                    topic, G_pub, pub_count,
                    keyword_signal=keyword_signal,
                    context=context
                ))

                # Step 6 — Tag graph type
                for a in inst_agents:
                    a["graph_type"] = "institutional"
                for a in pub_agents:
                    a["graph_type"] = "public"

                # Step 7 — Merge
                all_agents = inst_agents + pub_agents
                error_context["total_agents"] = len(all_agents)
                print(f"[API] Total agents: {len(all_agents)} ({len(inst_agents)} institutional, {len(pub_agents)} public)")

                # ── Zero agents guard ─────────────────────────────
                # If no agents were generated the debate will crash.
                # Fail cleanly here instead.
                if len(all_agents) == 0:
                    print(f"[API] Zero agents generated — failing cleanly")
                    simulations[simulation_id]["status"] = "failed"
                    simulations[simulation_id]["error"] = "no_agents"
                    simulations[simulation_id]["error_message"] = ERROR_MESSAGES["no_agents"]
                    return

                # Step 8 — Debate
                debate_result = run_async(run_debate(topic, all_agents, G_inst, num_rounds, G_pub))

                # Step 9 — Report
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

                error_key = classify_error(e, error_context)
                simulations[simulation_id]["status"] = "failed"
                simulations[simulation_id]["error"] = error_key
                simulations[simulation_id]["error_message"] = ERROR_MESSAGES[error_key]

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
        "error": sim.get("error", None),
        "error_message": sim.get("error_message", None),
    }), 200


@api.route("/api/simulation/<simulation_id>/debate", methods=["GET"])
def get_debate(simulation_id):
    sim = simulations.get(simulation_id)
    if not sim:
        return jsonify({"error": "simulation not found"}), 404

    return jsonify({
        "simulation_id": simulation_id,
        "rounds": sim["rounds"]
    }), 200


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
            reactions.append({
                "agent_id": agent["id"],
                "name": agent["name"],
                "opinion_before": agent["opinion"],
                "opinion_after": f"Considering: {event}. {agent['opinion']}",
                "delta": 0.0,
                "shifted": False
            })

        return jsonify({
            "injected_at_tick": current_tick,
            "reactions": reactions
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


@api.route("/api/report/<simulation_id>", methods=["GET"])
def get_report(simulation_id):
    sim = simulations.get(simulation_id)
    if not sim:
        return jsonify({"error": "simulation not found"}), 404

    return jsonify(sim["report"]), 200


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


@api.route("/api/correction/store", methods=["POST"])
def store_prediction_correction():
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