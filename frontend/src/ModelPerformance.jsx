import { useState } from 'react'
import STATS from './model_stats.json'

const MODELS = [
  { key: 'firesmoke', label: 'Fire / Smoke' },
  { key: 'ppe', label: 'PPE' },
]
const METRICS = ['mAP50', 'mAP50-95', 'Precision', 'Recall']

const fmt = (v) => (v == null ? null : v.toFixed(3))
const barW = (v) => (v == null ? 0 : Math.max(0, Math.min(100, v * 100)))

function tag(m) {
  if (m.status === 'pending') return { cls: 'pending', text: 'Retrain in progress' }
  if (m.fair) return { cls: 'fair', text: 'Same val set' }
  return { cls: 'prov', text: 'Provisional' }
}

export default function ModelPerformance() {
  const [model, setModel] = useState('firesmoke')
  const [metric, setMetric] = useState('mAP50')

  const m = STATS[model]
  if (!m) return null
  const t = tag(m)

  return (
    <section className="panel mperf">
      <div className="phead">
        <span className="num">07</span>
        <h3>Model Performance</h3>
        <span className={'mperf-tag ' + t.cls}>{t.text}</span>
      </div>

      <div className="mperf-body">
        <div className="mperf-tabs">
          <div className="mperf-seg">
            {MODELS.map((x) => (
              <button key={x.key} className={'seg-btn' + (model === x.key ? ' on' : '')} onClick={() => setModel(x.key)}>
                {x.label}
              </button>
            ))}
          </div>
          <div className="mperf-metrics">
            {METRICS.map((x) => (
              <button key={x} className={'met-btn' + (metric === x ? ' on' : '')} onClick={() => setMetric(x)}>
                {x}
              </button>
            ))}
          </div>
        </div>

        <div className="mperf-legend">
          <span><i className="lg old" /> {m.old.label}</span>
          <span><i className="lg new" /> {m.new.label}</span>
        </div>

        <div className="mperf-rows">
          {m.rows.map((row) => {
            const [oldV, newV] = row.metrics[metric] || [null, null]
            const delta = oldV != null && newV != null ? newV - oldV : null
            const newText = newV == null ? (m.status === 'pending' ? 'pending' : 'n/a') : fmt(newV)
            return (
              <div className="mp-row" key={row.name}>
                <div className="mp-name">{row.name}</div>
                <div className="mp-bars">
                  <div className="mp-bar">
                    <div className="mp-track"><div className="mp-fill old" style={{ width: barW(oldV) + '%' }} /></div>
                    <span className="mp-val">{fmt(oldV) ?? 'n/a'}</span>
                  </div>
                  <div className="mp-bar">
                    <div className="mp-track"><div className="mp-fill new" style={{ width: barW(newV) + '%' }} /></div>
                    <span className={'mp-val' + (newV == null ? ' muted' : '')}>{newText}</span>
                  </div>
                </div>
                <div className="mp-delta">
                  {delta == null
                    ? <span className="d-na">—</span>
                    : <span className={'d-badge ' + (delta >= 0 ? 'up' : 'down')}>{(delta >= 0 ? '▲ +' : '▼ ') + delta.toFixed(3)}</span>}
                </div>
              </div>
            )
          })}
        </div>

        {m.footnote && <div className="mperf-foot">{m.footnote}</div>}
      </div>
    </section>
  )
}
