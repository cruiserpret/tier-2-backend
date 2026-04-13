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
# Stores active simulations by simulation_id
# In Phase 2 we replace this with a proper database
simulations = {}

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

        simulation_id = f"sim_{uuid.uuid4().hex[:8]}"

        # Store immediately as "running"
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

        # Return immediately
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
                # Pass both graphs so agents query their own
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
    # Search all simulations for this agent
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

        # Inject the event as a new debate round
        from backend.utils.graph_utils import query_graph
        from backend.agents.debate_engine import run_debate_round

        # Temporarily add the event as a node concept
        # In Phase 2 this re-ingests and updates the graph
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

        # Clone state up to from_tick
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