import { useEffect, useRef, useState } from 'react'

const STATUS_TEXT = { present: 'Detected', violation: 'Missing', absent: 'Not observed' }
const LEVELS = {
  critical: ['Critical Hazard', 'Immediate intervention required'],
  warning: ['Caution', 'Conditions need attention'],
  ok: ['Site Clear', 'No active violations detected'],
}
const ZONES = ['head', 'eyes', 'face', 'torso', 'hands']
const EMPTY_PPE = [
  { label: 'Hardhat', zone: 'head', status: 'absent' },
  { label: 'Goggles', zone: 'eyes', status: 'absent' },
  { label: 'Mask', zone: 'face', status: 'absent' },
  { label: 'Safety Vest', zone: 'torso', status: 'absent' },
  { label: 'Gloves', zone: 'hands', status: 'absent' },
]

function fmtClock() {
  const d = new Date(), p = (n) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())} LOCAL`
}

export default function App() {
  const [result, setResult] = useState(null)
  const [file, setFile] = useState(null)
  const [isVideo, setIsVideo] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [over, setOver] = useState(false)
  const [notice, setNotice] = useState(null)
  const [hoveredZone, setHoveredZone] = useState(null)
  const [revealKey, setRevealKey] = useState(0)
  const [clock, setClock] = useState(fmtClock())
  const fileRef = useRef(null)

  useEffect(() => {
    const t = setInterval(() => setClock(fmtClock()), 1000)
    return () => clearInterval(t)
  }, [])

  // --- file intake ---
  const applyFile = (f) => {
    if (!f) return
    setPreviewUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(f) })
    setFile(f)
    setIsVideo((f.type || '').startsWith('video'))
    setNotice(null)
  }
  const clearFile = () => {
    setPreviewUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return null })
    setFile(null); setIsVideo(false); setNotice(null)
  }
  const pickFile = () => fileRef.current && fileRef.current.click()
  const onDropKey = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickFile() } }

  // --- analysis ---
  const runAnalysis = async () => {
    if (!file || processing) return
    const endpoint = isVideo ? '/detect/video' : '/detect/image'
    setProcessing(true); setNotice(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch(endpoint, { method: 'POST', body: fd })
      if (!res.ok) throw new Error('HTTP ' + res.status)
      const data = await res.json()
      if (data.error) throw new Error(data.error)
      setResult(data)
      setRevealKey((k) => k + 1)
    } catch (err) {
      setNotice('Analysis failed: ' + err.message + '. Is the server running?')
    } finally {
      setProcessing(false)
    }
  }

  // --- derived values ---
  const d = result || {}
  const isVid = d.type === 'video'
  const summary = d.ppe_summary && d.ppe_summary.length ? d.ppe_summary : EMPTY_PPE
  const statusMap = {}
  summary.forEach((p) => { statusMap[p.zone] = p.status })
  const zoneClass = (zone) => {
    const st = statusMap[zone] || 'absent'
    const tone = st === 'present' ? 'bz-present' : st === 'violation' ? 'bz-violation' : 'bz-absent'
    return `bz bz-${zone} ${tone}${hoveredZone === zone ? ' glow' : ''}`
  }

  const level = d.level || 'ok'
  const [levelLabel, levelSub] = LEVELS[level] || LEVELS.ok
  const shieldGlyph = level === 'ok' ? 'M8.5 12.5l2.5 2.5 4.5-5' : 'M12 8v4M12 16v.01'

  const sev = isVid ? (d.peak_severity || {}) : (d.severity || {})
  const pct = (v) => Math.max(0, Math.min(100, Math.round(v || 0)))
  const fireP = pct(sev.fire), smokeP = pct(sev.smoke)

  const alerts = d.alerts || []
  const tracked = isVid ? (d.tracking?.tracked_workers || 0) : (d.persons_detected || 0)
  const flagged = isVid
    ? (d.tracking?.workers_with_violations?.length || 0)
    : summary.filter((p) => p.status === 'violation').length
  const fall = !!d.fall_detected

  const tracking = []
  if (isVid && d.tracking?.workers_with_violations) {
    d.tracking.workers_with_violations.forEach((w) => {
      const v = w.violations || {}, keys = Object.keys(v)
      const id = 'W-' + String(w.worker_id).padStart(2, '0')
      if (!keys.length) tracking.push({ id, violation: '—', duration: '—' })
      keys.forEach((k) => tracking.push({
        id,
        violation: k.replace(/^NO-/, 'No '),
        duration: v[k].duration_sec != null ? v[k].duration_sec.toFixed(1) + 's' : '—',
      }))
    })
  }

  // output viewer — prefer the backend's annotated path, else the local upload preview
  let outputUrl = null, outputIsVideo = false
  if (isVid) {
    if (d.video_playable && d.output_video) { outputUrl = d.output_video; outputIsVideo = true }
    else if (d.keyframe) { outputUrl = d.keyframe }
    else if (previewUrl) { outputUrl = previewUrl; outputIsVideo = true }
  } else if (d.output_image) {
    outputUrl = d.output_image
  } else if (previewUrl && !isVideo) {
    outputUrl = previewUrl
  }
  const usingLocal = outputUrl && outputUrl === previewUrl
  const hasOutput = !!outputUrl
  const outputBadge = usingLocal ? 'Source preview' : outputIsVideo ? 'Annotated video' : 'Annotated frame'
  const outputMeta = isVid
    ? `${d.frames_processed || 0}/${d.frames_total || 0} frames · stride ${d.frame_stride || 1}`
    : `${(d.ppe_detections || []).length} PPE boxes · ${(d.firesmoke_detections || []).length} fire/smoke boxes`
  const downloadName = (d.filename || 'output').replace(/\.[^.]+$/, '') + (outputIsVideo ? '_annotated.mp4' : '_annotated.jpg')

  return (
    <div className="app">
      <div className="tape" />
      <div className="wrap">
        <header className="mast">
          <svg className="sigil" viewBox="0 0 48 48" fill="none"><path d="M24 4 L45 42 H3 Z" fill="#F2C200" stroke="#0c0d10" strokeWidth="2.5" strokeLinejoin="round" /><rect x="21.4" y="18" width="5.2" height="13" rx="2.4" fill="#14171C" /><circle cx="24" cy="36" r="2.9" fill="#14171C" /></svg>
          <div><div className="mast-t">Site Safety Monitor</div><div className="mast-s">Computer Vision · PPE · Fire/Smoke · Fall Detection</div></div>
          <div className="mast-r">
            <div className="live"><i /> Online</div>
            <div className="clock">{clock}</div>
          </div>
        </header>

        <div className="lay">
          {/* LEFT: upload + viewer */}
          <section>
            <div className="panel" style={{ padding: 16 }}>
              <div className="eyebrow" style={{ marginBottom: 13 }}>Evidence Intake</div>
              <div
                className={'drop' + (over ? ' over' : '')}
                role="button" tabIndex={0}
                onClick={pickFile} onKeyDown={onDropKey}
                onDragOver={(e) => { e.preventDefault(); if (!over) setOver(true) }}
                onDragLeave={(e) => { e.preventDefault(); setOver(false) }}
                onDrop={(e) => { e.preventDefault(); setOver(false); applyFile(e.dataTransfer.files && e.dataTransfer.files[0]) }}
              >
                {processing && <div className="scanline" />}
                {file ? (
                  <>
                    <div className="filechip"><span className="kind">{isVideo ? 'Video' : 'Image'}</span><b>{file.name}</b></div>
                    <p className="mono">{`${(file.size / 1048576).toFixed(1)} MB · ${file.type || 'unknown'}`}</p>
                    <p style={{ fontSize: '11.5px', color: 'var(--steel2)' }}>Drop another file to replace</p>
                  </>
                ) : (
                  <>
                    <svg className="drop-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="M12 3v13" /><path d="M7 8l5-5 5 5" /></svg>
                    <h4>Drop site photo or video</h4>
                    <p>or click to browse — image and video accepted</p>
                    <p className="mono">JPG · PNG · MP4 · MOV</p>
                  </>
                )}
              </div>
              <input ref={fileRef} type="file" accept="image/*,video/*" style={{ display: 'none' }} onChange={(e) => applyFile(e.target.files && e.target.files[0])} />
              <div className="actions">
                <button className="btn btn-run" onClick={runAnalysis} disabled={!file || processing}>
                  {processing ? (<><span className="spin" /> Analyzing</>) : (file ? 'Run analysis' : 'Select a file')}
                </button>
                {file && <button className="btn btn-ghost" onClick={clearFile}>Clear</button>}
              </div>
              {notice && (
                <div className="notice">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ flex: 'none', marginTop: 1 }}><circle cx="12" cy="12" r="9" /><path d="M12 8v5M12 16.5v.01" /></svg>
                  <span>{notice}</span>
                </div>
              )}
            </div>

            <div className="viewer">
              <div className="eyebrow" style={{ margin: '4px 0 10px' }}>Annotated Output</div>
              <div className="vframe">
                {hasOutput ? (
                  <div style={{ width: '100%' }}>
                    <div className="vbadge">{outputBadge}</div>
                    {outputIsVideo
                      ? <video src={outputUrl} controls playsInline />
                      : <img src={outputUrl} alt="Annotated detection output" />}
                  </div>
                ) : (
                  <div className="vempty">
                    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#5D656F" strokeWidth="1.4"><rect x="3" y="4" width="18" height="14" rx="2" /><circle cx="9" cy="10" r="2.2" /><path d="M3 16l5-4 4 3 3-3 6 5" /></svg>
                    <div className="mono">Annotated detections appear here after analysis. Bounding boxes mark PPE, fire/smoke, and flagged workers.</div>
                  </div>
                )}
              </div>
              {hasOutput && (
                <div className="vbar">
                  <span className="meta">{outputMeta}</span>
                  <a className="dl" href={outputUrl} download={downloadName}>
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3v12M7 11l5 4 5-4M5 21h14" /></svg>
                    {outputIsVideo ? 'Download annotated video' : 'Download annotated image'}
                  </a>
                </div>
              )}
            </div>
          </section>

          {/* RIGHT: status panels */}
          <aside className="col-r">
            <div className={'panel placard ' + level}>
              <svg className="pl-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2.5l8 3.2v5c0 5-3.4 8.6-8 10.8-4.6-2.2-8-5.8-8-10.8v-5z" /><path d={shieldGlyph} /></svg>
              <div><div className="pl-v">{levelLabel}</div><div className="pl-sub">{levelSub}</div></div>
            </div>

            <div className="panel">
              <div className="phead"><span className="num">02</span><h3>PPE Compliance</h3></div>
              <div className="diagwrap">
                <div className="diag reveal" key={revealKey}>
                  <svg viewBox="0 0 240 300" fill="none">
                    <g fill="#20262E"><rect x="96" y="206" width="20" height="84" rx="8" /><rect x="124" y="206" width="20" height="84" rx="8" /><path d="M80 138 H160 L150 214 H90 Z" /><circle cx="120" cy="98" r="33" /><rect x="108" y="120" width="24" height="20" /></g>
                    <g stroke="#20262E" strokeWidth="25" strokeLinecap="round"><path d="M86 152 L56 204" /><path d="M154 152 L184 204" /></g>
                    <g className={zoneClass('hands')}><path d="M44 200 q-6 14 4 22 q12 8 20-2 q4-10-2-20 z" /><path d="M196 200 q6 14 -4 22 q-12 8 -20-2 q-4-10 2-20 z" /></g>
                    <g className={zoneClass('torso')}><path d="M88 142 H152 L145 214 H95 Z" /></g>
                    <rect x="118" y="144" width="4" height="68" fill="#14171C" /><rect x="92" y="166" width="56" height="6" fill="#14171C" opacity=".55" /><rect x="92" y="188" width="56" height="6" fill="#14171C" opacity=".55" />
                    <g className={zoneClass('face')}><path d="M101 108 H139 L133 126 Q120 133 107 126 Z" /></g>
                    <g className={zoneClass('eyes')}><rect x="92" y="89" width="56" height="14" rx="7" /></g>
                    <g className={zoneClass('head')}><path d="M86 92 Q86 50 120 48 Q154 50 154 92 Z" /><rect x="80" y="88" width="80" height="9" rx="4" /><rect x="114" y="42" width="12" height="9" rx="3" /></g>
                  </svg>
                </div>
                <div className="checklist">
                  {summary.map((p) => (
                    <div
                      key={p.zone}
                      className={'ck-row' + (hoveredZone === p.zone ? ' on' : '')}
                      tabIndex={0}
                      onMouseEnter={() => setHoveredZone(p.zone)}
                      onMouseLeave={() => setHoveredZone(null)}
                      onFocus={() => setHoveredZone(p.zone)}
                      onBlur={() => setHoveredZone(null)}
                    >
                      <span className={'ck-dot ' + p.status} />
                      <span className="ck-label">{p.label}</span>
                      <span className={'ck-status ' + p.status}>{STATUS_TEXT[p.status] || p.status}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="panel">
              <div className="phead"><span className="num">03</span><h3>Hazard Spread</h3></div>
              <div className="gauges">
                <div className="gauge"><div className="gauge-h"><span className="gauge-l">Fire</span><span className="gauge-v" style={{ color: 'var(--red)' }}>{fireP}%</span></div><div className="gauge-track"><div className="gauge-fill fire" style={{ width: fireP + '%' }} /></div></div>
                <div className="gauge"><div className="gauge-h"><span className="gauge-l">Smoke</span><span className="gauge-v" style={{ color: '#c3ccd6' }}>{smokeP}%</span></div><div className="gauge-track"><div className="gauge-fill smoke" style={{ width: smokeP + '%' }} /></div></div>
              </div>
            </div>

            <div className="panel">
              <div className="phead"><span className="num">04</span><h3>Alerts</h3></div>
              <div className="alerts">
                {alerts.length ? alerts.map((a, i) => (
                  <div className="alert" key={i}>
                    <svg className="tri" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3 L22 20 H2 Z" /><path d="M12 10v4M12 17v.01" /></svg>
                    <span>{a}</span>
                  </div>
                )) : <div className="alerts-empty">No active alerts.</div>}
              </div>
            </div>

            <div className="panel">
              <div className="phead"><span className="num">05</span><h3>Readout</h3></div>
              <div className="readout">
                <div className="tile"><div className="tile-l">Workers Tracked</div><div className="tile-v">{tracked}</div></div>
                <div className="tile"><div className="tile-l">Flagged Workers</div><div className={'tile-v' + (flagged > 0 ? ' red' : '')}>{flagged}</div></div>
                <div className="tile"><div className="tile-l">Fall Detected</div><div className={'tile-v' + (fall ? ' red' : ' green')}>{fall ? 'YES' : 'NO'}</div></div>
                <div className="tile"><div className="tile-l">Pose Model</div><div className={'tile-v' + (result && d.pose_available ? ' green' : '')}>{result ? (d.pose_available ? 'ACTIVE' : 'OFF') : '—'}</div></div>
              </div>
            </div>

            <div className="panel">
              <div className="phead"><span className="num">06</span><h3>Worker Tracking</h3></div>
              <div className="track">
                {tracking.length ? (
                  <table>
                    <thead><tr><th>Worker</th><th>Violation</th><th>Duration</th></tr></thead>
                    <tbody>
                      {tracking.map((t, i) => (
                        <tr key={i}><td className="wid">{t.id}</td><td className="viol">{t.violation}</td><td className="dur">{t.duration}</td></tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <div className="track-empty">{isVid ? 'No sustained violations tracked across frames.' : 'Worker tracking requires a video upload.'}</div>
                )}
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  )
}
