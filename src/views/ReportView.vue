<template>
  <div class="report">
    <div class="report-header">
      <button class="back-btn" @click="$router.push('/')">← New simulation</button>
      <div class="header-badges">
        <span class="badge">God's Eye View</span>
      </div>
    </div>

    <div class="report-hero">
      <h1 class="report-title">{{ topic }}</h1>
      <p class="report-subtitle">Simulation complete · {{ report.agents_shifted }} of {{ totalAgents }} agents shifted opinion</p>
    </div>

    <div class="stats-row">
      <div class="stat-card">
        <span class="stat-num">{{ report.agents_shifted }}</span>
        <span class="stat-label">agents shifted</span>
      </div>
      <div class="stat-card">
        <span class="stat-num">{{ report.agents_held }}</span>
        <span class="stat-label">held position</span>
      </div>
      <div class="stat-card">
        <span class="stat-num">{{ totalAgents }}</span>
        <span class="stat-label">total agents</span>
      </div>
      <div class="stat-card">
        <span class="stat-num">{{ shiftRate }}%</span>
        <span class="stat-label">shift rate</span>
      </div>
    </div>

    <div class="section">
      <p class="section-label">Summary</p>
      <p class="section-text">{{ report.summary }}</p>
    </div>

    <div class="section">
      <p class="section-label">Decisive arguments</p>
      <div class="argument-card" v-for="arg in (report.decisive_arguments ?? [])" :key="arg.agent_id">
        <div class="arg-header">
          <span class="arg-agent">{{ arg.agent_id }}</span>
          <span class="arg-influenced">influenced {{ (arg.influenced_agents ?? []).length }} agents</span>
        </div>
        <p class="arg-text">"{{ arg.argument }}"</p>
      </div>
    </div>

    <div class="section">
      <p class="section-label">Predicted trajectory</p>
      <p class="trajectory">{{ report.predicted_trajectory }}</p>
    </div>

    <div class="section">
      <p class="section-label">Agent summaries</p>
      <div class="agent-summary-grid">
        <div
          class="agent-summary-card"
          v-for="agent in (report.agent_summaries ?? [])"
          :key="agent.agent_id"
          :class="agent.shifted ? 'shifted' : 'held'"
        >
          <div class="summary-top">
            <span class="summary-name">{{ agent.name }}</span>
            <span class="summary-badge" :class="agent.shifted ? 'badge-shifted' : 'badge-held'">
              {{ agent.shifted ? 'shifted' : 'held' }}
            </span>
          </div>
          <p class="summary-stance">Final stance: {{ agent.final_stance }}</p>
          <p class="summary-moment">{{ agent.key_moment }}</p>
        </div>
      </div>
    </div>

    <div class="report-footer">
      <button class="branch-btn" @click="$router.push({ path: '/branch', query: { tick: 8 } })">
  Branch at tick 8 →
</button>
      <button class="new-btn" @click="$router.push('/')">Run another simulation →</button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { getReport } from '../api/mock.js'

const route = useRoute()
const topic = route.query.topic ?? 'Untitled simulation'
const report = getReport()
const totalAgents = computed(() => (report.agents_shifted ?? 0) + (report.agents_held ?? 0))
const shiftRate = computed(() => Math.round((report.agents_shifted / totalAgents.value) * 100))
</script>

<style scoped>
.report { max-width: 860px; margin: 0 auto; padding: 40px 20px; display: flex; flex-direction: column; gap: 32px; }
.report-header { display: flex; justify-content: space-between; align-items: center; }
.back-btn { background: none; border: 1px solid #222; color: #666; padding: 6px 16px; border-radius: 20px; cursor: pointer; font-size: 13px; }
.badge { background: #1a1a1a; color: #555; font-size: 11px; padding: 4px 12px; border-radius: 20px; border: 1px solid #222; }

.report-hero { display: flex; flex-direction: column; gap: 8px; }
.report-title { font-size: 32px; font-weight: 500; color: #fff; line-height: 1.3; }
.report-subtitle { font-size: 14px; color: #555; }

.stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.stat-card { background: #111; border: 1px solid #1e1e1e; border-radius: 12px; padding: 16px; display: flex; flex-direction: column; gap: 4px; }
.stat-num { font-size: 28px; font-weight: 500; color: #fff; }
.stat-label { font-size: 12px; color: #555; }

.section { display: flex; flex-direction: column; gap: 12px; }
.section-label { font-size: 11px; color: #444; text-transform: uppercase; letter-spacing: 0.06em; }
.section-text { font-size: 15px; color: #bbb; line-height: 1.7; }

.argument-card { background: #111; border: 1px solid #1e1e1e; border-radius: 12px; padding: 16px 20px; display: flex; flex-direction: column; gap: 8px; }
.arg-header { display: flex; justify-content: space-between; }
.arg-agent { font-size: 12px; color: #1D9E75; font-weight: 500; }
.arg-influenced { font-size: 12px; color: #444; }
.arg-text { font-size: 14px; color: #bbb; line-height: 1.6; font-style: italic; }

.trajectory { font-size: 15px; color: #bbb; line-height: 1.7; background: #111; border-left: 2px solid #1D9E75; padding: 14px 18px; border-radius: 0 12px 12px 0; }

.agent-summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.agent-summary-card { background: #111; border: 1px solid #1e1e1e; border-radius: 12px; padding: 14px 16px; display: flex; flex-direction: column; gap: 6px; }
.agent-summary-card.shifted { border-left: 2px solid #1D9E75; }
.agent-summary-card.held { border-left: 2px solid #333; }
.summary-top { display: flex; justify-content: space-between; align-items: center; }
.summary-name { font-size: 14px; font-weight: 500; color: #fff; }
.summary-badge { font-size: 11px; padding: 2px 8px; border-radius: 20px; }
.badge-shifted { background: #0a2e1e; color: #1D9E75; }
.badge-held { background: #1a1a1a; color: #555; }
.summary-stance { font-size: 12px; color: #555; }
.summary-moment { font-size: 12px; color: #444; }

.report-footer { padding-top: 20px; border-top: 1px solid #1a1a1a; }
.new-btn { background: #fff; color: #000; border: none; padding: 12px 24px; border-radius: 10px; font-size: 14px; font-weight: 500; cursor: pointer; }
</style>