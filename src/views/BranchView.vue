<template>
  <div class="branch-view">
    <div class="branch-header">
      <button class="back-btn" @click="$router.push('/')">← New simulation</button>
      <h2 class="branch-title">Timeline branches</h2>
      <p class="branch-subtitle">Two parallel simulations forked at tick {{ forkTick }}</p>
    </div>

    <div class="branch-grid">
      <div class="branch-col">
        <div class="branch-label original">Original timeline</div>
        <div class="branch-stats">
          <span class="stat">For: {{ original.for }}</span>
          <span class="stat">Against: {{ original.against }}</span>
          <span class="stat">Neutral: {{ original.neutral }}</span>
        </div>
        <div class="event-list">
          <div class="event" v-for="(e, i) in original.events" :key="i">
            <span class="event-tick">T{{ e.tick }}</span>
            <span class="event-agent" :style="{ color: e.color }">{{ e.agent }}</span>
            <span class="event-text">{{ e.text }}</span>
          </div>
        </div>
      </div>

      <div class="branch-divider">
        <div class="fork-point">
          <span class="fork-label">Fork at T{{ forkTick }}</span>
        </div>
      </div>

      <div class="branch-col">
        <div class="branch-label alternate">Alternate timeline</div>
        <div class="branch-stats">
          <span class="stat">For: {{ alternate.for }}</span>
          <span class="stat">Against: {{ alternate.against }}</span>
          <span class="stat">Neutral: {{ alternate.neutral }}</span>
        </div>
        <div class="event-list">
          <div class="event" v-for="(e, i) in alternate.events" :key="i">
            <span class="event-tick">T{{ e.tick }}</span>
            <span class="event-agent" :style="{ color: e.color }">{{ e.agent }}</span>
            <span class="event-text">{{ e.text }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="branch-footer">
      <button class="new-btn" @click="$router.push('/')">Run new simulation →</button>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const forkTick = route.query.tick ?? 8

const stanceColors = { for: '#1D9E75', against: '#E24B4A', neutral: '#888' }

const agents = ['The Skeptic', 'The Optimist', 'The Realist', 'The Contrarian', 'The Idealist', 'The Pragmatist']

const generateEvents = (seed) => {
  const events = []
  const args = [
    'Regulation protects citizens from harm.',
    'Innovation will be stifled by oversight.',
    'We need more data before deciding.',
    'Both sides have valid points here.',
    'The market will self-correct naturally.',
    'Every powerful technology needs governance.',
  ]
  for (let i = parseInt(forkTick) + 1; i <= parseInt(forkTick) + 7; i++) {
    const agent = agents[Math.floor((i + seed) % agents.length)]
    const stance = ['for', 'against', 'neutral'][Math.floor((i * seed) % 3)]
    events.push({
      tick: i,
      agent,
      text: args[Math.floor((i + seed) % args.length)],
      color: stanceColors[stance]
    })
  }
  return events
}

const original = ref({
  for: 4, against: 5, neutral: 3,
  events: generateEvents(1)
})

const alternate = ref({
  for: 7, against: 2, neutral: 3,
  events: generateEvents(3)
})
</script>

<style scoped>
.branch-view { max-width: 1100px; margin: 0 auto; padding: 40px 20px; display: flex; flex-direction: column; gap: 32px; }
.branch-header { display: flex; flex-direction: column; gap: 6px; }
.back-btn { background: none; border: 1px solid #222; color: #666; padding: 6px 16px; border-radius: 20px; cursor: pointer; font-size: 13px; width: fit-content; }
.branch-title { font-size: 28px; font-weight: 500; color: #fff; }
.branch-subtitle { font-size: 14px; color: #555; }

.branch-grid { display: grid; grid-template-columns: 1fr 60px 1fr; gap: 0; }
.branch-col { display: flex; flex-direction: column; gap: 12px; }
.branch-label { font-size: 12px; font-weight: 500; padding: 4px 12px; border-radius: 20px; width: fit-content; }
.original { background: #0a2e1e; color: #1D9E75; }
.alternate { background: #1e1a0a; color: #BA7517; }

.branch-stats { display: flex; gap: 16px; }
.stat { font-size: 12px; color: #555; }

.event-list { display: flex; flex-direction: column; gap: 6px; }
.event { display: flex; gap: 8px; align-items: baseline; background: #111; border-radius: 8px; padding: 8px 12px; }
.event-tick { font-size: 11px; color: #333; min-width: 24px; }
.event-agent { font-size: 12px; font-weight: 500; min-width: 100px; }
.event-text { font-size: 12px; color: #666; }

.branch-divider { display: flex; flex-direction: column; align-items: center; padding-top: 40px; }
.fork-point { background: #111; border: 1px solid #222; border-radius: 8px; padding: 8px 6px; writing-mode: vertical-rl; }
.fork-label { font-size: 11px; color: #444; }

.branch-footer { padding-top: 20px; border-top: 1px solid #1a1a1a; }
.new-btn { background: #fff; color: #000; border: none; padding: 12px 24px; border-radius: 10px; font-size: 14px; font-weight: 500; cursor: pointer; }
</style>