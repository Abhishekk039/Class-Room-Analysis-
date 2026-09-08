import { useEffect, useRef, useState } from 'react'
import {
  AlertTriangle, ArrowUpRight, Check, CheckCircle2, FileAudio,
  Filter, Headphones, LoaderCircle, Mic2, Play, RotateCcw, Search, Sparkles,
  UploadCloud, Volume2, X,
} from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api'

const formatTime = (seconds) => {
  if (!Number.isFinite(seconds)) return '—'
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  return hours ? `${hours}h ${minutes % 60}m` : `${minutes}m`
}

const formatClock = (seconds) => {
  const mins = Math.floor(seconds / 60)
  return `${String(mins).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}

const formatNumber = (value, digits = 0) => Number(value || 0).toFixed(digits)

// Confidence badge: green ≥ 0.70, amber 0.50–0.69, red < 0.50
function ConfidenceBadge({ value }) {
  if (value == null) return null
  const pct = Math.round(value * 100)
  const cls = value >= 0.70 ? 'conf-high' : value >= 0.50 ? 'conf-med' : 'conf-low'
  return <span className={`conf-badge ${cls}`}>{pct}%</span>
}

function App() {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [job, setJob] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [config, setConfig] = useState(null)
  const [error, setError] = useState('')
  const [isDragging, setIsDragging] = useState(false)

  // Fetch API config on mount
  useEffect(() => {
    fetch(`${API_URL}/config`)
      .then((res) => res.json())
      .then((data) => setConfig(data))
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!job?.session_id || ['completed', 'failed'].includes(job.status)) return undefined
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_URL}/status/${job.session_id}`)
        if (!response.ok) throw new Error('Could not read processing status.')
        const nextJob = await response.json()
        setJob(nextJob)
        if (nextJob.status === 'completed') {
          const result = await fetch(`${API_URL}/analysis/${job.session_id}`)
          if (!result.ok) throw new Error('Analysis finished without a result.')
          setAnalysis(await result.json())
        }
        if (nextJob.status === 'failed') setError(nextJob.error || 'Audio processing failed.')
      } catch (requestError) {
        setError(requestError.message)
      }
    }, 1800)
    return () => window.clearInterval(timer)
  }, [job])

  const chooseFile = (nextFile) => {
    if (!nextFile) return
    setError(''); setAnalysis(null); setJob(null); setFile(nextFile)
  }

  const uploadFile = async () => {
    if (!file) return
    setError(''); setJob({ status: 'uploading' })
    const body = new FormData()
    body.append('file', file)
    try {
      const response = await fetch(`${API_URL}/upload`, { method: 'POST', body })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Upload failed.')
      setJob({ session_id: payload.session_id, status: payload.status || 'queued' })
    } catch (requestError) {
      setJob(null); setError(requestError.message)
    }
  }

  const reset = () => { setFile(null); setJob(null); setAnalysis(null); setError('') }

  const processing = job && !['completed', 'failed'].includes(job.status)

  // Dynamic percentage calculation across all pipeline stages
  const progressPct = (() => {
    if (!job?.status) return 0
    if (job.progress) {
      const m = job.progress.match(/(\d+)\/(\d+)/)
      if (m) {
        const chunkPct = Math.round((parseInt(m[1]) / parseInt(m[2])) * 100)
        return Math.min(90, Math.max(25, chunkPct))
      }
    }
    const stageMap = {
      uploading: 10,
      queued: 15,
      preprocessing: 25,
      analyzing_speech: 35,
      chunking: 40,
      transcribing: 50,
      transcribing_gemini: 55,
      stage2_correction: 75,
      quality_check: 85,
      diarization: 92,
      computing_metrics: 97,
      completed: 100,
    }
    return stageMap[job.status] || 45
  })()

  const statusLabel = {
    uploading: 'Uploading recording…',
    queued: 'Queued for processing…',
    preprocessing: 'Preprocessing audio (16kHz mono)…',
    analyzing_speech: 'Detecting speech regions (VAD)…',
    chunking: 'Splitting into chunks…',
    transcribing: 'Transcribing with local Whisper…',
    transcribing_gemini: 'Stage 1: Gemini 2.5 Flash ASR transcribing…',
    stage2_correction: 'Stage 2: Contextual correction agent running…',
    quality_check: 'Checking transcript quality…',
    diarization: 'Identifying speakers (teacher / student)…',
    computing_metrics: 'Computing engagement metrics…',
  }[job?.status] || 'Processing…'

  const [navNotice, setNavNotice] = useState('')

  const navigateTo = (targetId) => {
    setNavNotice('')
    if (targetId === 'insights' || targetId === 'transcript') {
      if (!analysis) {
        setNavNotice('Please upload and analyze a recording first to view insights and transcript.')
        const recEl = document.getElementById('recording')
        if (recEl) recEl.scrollIntoView({ behavior: 'smooth' })
        return
      }
    }
    const el = document.getElementById(targetId)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" onClick={(e) => { e.preventDefault(); navigateTo('top') }} aria-label="Classroom Voice Analytics home">
          <span className="brand-mark"><Mic2 size={18} /></span>
          <span>classroom<br /><b>voice analytics</b></span>
        </a>
        <nav className="main-nav" aria-label="Primary navigation">
          <a href="#recording" onClick={(e) => { e.preventDefault(); navigateTo('recording') }}>New recording</a>
          <a href="#insights" onClick={(e) => { e.preventDefault(); navigateTo('insights') }}>Insights</a>
          <a href="#transcript" onClick={(e) => { e.preventDefault(); navigateTo('transcript') }}>Transcript</a>
        </nav>
        <div className="topbar-actions">
          {config?.has_gemini_key && (
            <span className="gemini-pill" title="Gemini 2.5 Flash Hybrid Engine Active">
              ✦ Gemini 2.5 Flash
            </span>
          )}
          <a className="outline-button" href="#how-it-works" onClick={(e) => { e.preventDefault(); navigateTo('how-it-works') }}>How it works <ArrowUpRight size={16} /></a>
        </div>
      </header>


      <main id="top">
        <section className="intro-wrap">
          <div className="intro-copy">
            <p className="kicker"><span /> For curious classrooms</p>
            <h1>Listen closer.<br /><em>Teach smarter.</em></h1>
            <p className="intro-text">Turn a classroom recording into a clear story of participation, questions, and quiet moments.</p>
            <div className="intro-notes">
              <span><Check size={14} /> Hindi + English</span>
              <span><Check size={14} /> Local Whisper transcription</span>
            </div>
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="sun-disc" />
            <div className="hero-paper">
              <span className="paper-label">TODAY'S LESSON</span>
              <div className="waveform hero-wave"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
              <div className="paper-line wide" /><div className="paper-line" />
              <span className="paper-caption">voices make<br />the lesson</span>
            </div>
            <div className="doodle-circle"><Headphones size={24} /></div>
            <div className="doodle-star">✦</div>
          </div>
        </section>

        <section className="workspace" id="recording">
          <div className="section-heading">
            <div>
              <p className="kicker"><span /> Start here</p>
              <h2>Bring the room<br /><em>to life.</em></h2>
            </div>
            <p className="section-side-note">Drop in a recording and we'll map the rhythm of the room — from the first hello to the final question.</p>
          </div>

          {navNotice && (
            <div className="nav-notice-strip">
              <Sparkles size={16} />
              <span>{navNotice}</span>
              <button onClick={() => setNavNotice('')}><X size={14} /></button>
            </div>
          )}

          {!analysis && (

            <div
              className={`upload-panel ${isDragging ? 'dragging' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => { e.preventDefault(); setIsDragging(false); chooseFile(e.dataTransfer.files[0]) }}
            >
              <div className="upload-icon"><UploadCloud size={25} /></div>
              <div>
                <h3>{file ? file.name : 'Choose a classroom recording'}</h3>
                <p>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · ready to analyze` : 'WAV, MP3, M4A, FLAC or WebM · up to 2 GB'}</p>
              </div>
              {!file && <button className="soft-button" onClick={() => inputRef.current?.click()}>Browse files <ArrowUpRight size={16} /></button>}
              {file && (
                <div className="upload-actions">
                  <button className="ghost-icon" aria-label="Remove selected file" onClick={reset}><X size={18} /></button>
                  <button className="primary-button" onClick={uploadFile} disabled={processing}><Sparkles size={16} /> Analyze recording</button>
                </div>
              )}
              <input ref={inputRef} type="file" accept="audio/*,video/mp4,video/webm" hidden onChange={(e) => chooseFile(e.target.files[0])} />
            </div>
          )}

          {processing && (
            <div className="processing-strip">
              <LoaderCircle className="spin" size={22} />
              <div>
                <strong>{statusLabel} ({progressPct}%)</strong>
                {job.progress && <span className="chunk-progress">{job.progress}</span>}
                <span>Processing the recording locally with speech recognition. You can leave this tab open.</span>
              </div>
              <div className="progress-right">
                <div className="progress-bar-wrap">
                  <div className="progress-bar-track">
                    <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
                  </div>
                  <span className="progress-pct">{progressPct}%</span>
                </div>
                <span className="status-pill">{job.status}</span>
              </div>
            </div>
          )}

          {error && <div className="error-strip"><X size={18} /> {error}<button onClick={reset}><RotateCcw size={16} /> Try again</button></div>}
        </section>

        {analysis && <AnalysisView analysis={analysis} onReset={reset} />}

        <section className="how-section" id="how-it-works">
          <div className="how-mark"><Volume2 size={28} /></div>
          <div><p className="kicker"><span /> A gentler kind of dashboard</p><h2>Make space for<br /><em>every voice.</em></h2></div>
          <p>Classroom Voice Analytics transcribes locally and applies conservative quality checks to help you notice who is speaking, when questions open up, and where technical terms are spoken.</p>
        </section>
      </main>
      <footer><span>classroom voice analytics</span><span>Made for better conversations <span className="footer-dot">●</span></span></footer>
    </div>
  )
}

function AnalysisView({ analysis, onReset }) {
  const { transcript, metrics } = analysis
  const rawSegments = transcript.segments || []
  
  const [filter, setFilter] = useState('all') // 'all' | 'teacher' | 'student' | 'question' | 'corrected'
  const [searchQuery, setSearchQuery] = useState('')

  const suspiciousCount = rawSegments.filter(s => s.suspicious).length
  const correctedCount  = rawSegments.filter(s => s.correction_applied).length

  // Filtered segments logic
  const filteredSegments = rawSegments.filter(seg => {
    // Role / type filter
    if (filter === 'teacher' && seg.speaker !== 'teacher') return false
    if (filter === 'student' && seg.speaker !== 'student') return false
    if (filter === 'question' && !seg.is_question) return false
    if (filter === 'corrected' && !seg.correction_applied) return false

    // Search query filter
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      const matchText = seg.text.toLowerCase().includes(q)
      const matchOrig = seg.original_text && seg.original_text.toLowerCase().includes(q)
      if (!matchText && !matchOrig) return false
    }
    return true
  })

  return (
    <section className="results" id="insights">
      <div className="results-header">
        <div>
          <p className="kicker"><span /> Your classroom story</p>
          <h2>A room in <em>motion.</em></h2>
        </div>
        <button className="outline-button" onClick={onReset}><RotateCcw size={15} /> New recording</button>
      </div>

      {/* Performance & Provider banner */}
      {analysis.processing_time_seconds > 0 && (
        <div className="perf-banner">
          <span>⚡ Processed in <b>{Math.round(analysis.processing_time_seconds / 60)}m {Math.round(analysis.processing_time_seconds % 60)}s</b></span>
          <span>RTF <b>{analysis.rtf?.toFixed(3)}</b></span>
          <span>Engine <b>{analysis.asr_provider === 'gemini' ? 'Gemini 2.5 Flash' : 'Whisper'}</b></span>
          {suspiciousCount > 0 && <span className="perf-warn"><AlertTriangle size={13} /> {suspiciousCount} flagged</span>}
          {correctedCount > 0 && <span className="perf-ok"><CheckCircle2 size={13} /> {correctedCount} Stage 2 corrected</span>}
        </div>
      )}

      <div className="metrics-grid">
        <Metric label="Teacher talk time" value={formatTime(metrics.teacher_talk_time_seconds)} accent="amber" detail={`${formatNumber(metrics.teacher_dominance_ratio * 100, 0)}% of speech`} />
        <Metric label="Student talk time" value={formatTime(metrics.student_talk_time_seconds)} accent="teal" detail={`${formatNumber(metrics.student_participation_indicator * 100, 0)}% of turns`} />
        <Metric label="Questions asked" value={metrics.teacher_question_count} accent="coral" detail={`${metrics.student_response_count} responses detected`} />
        <Metric label="Quiet moments" value={formatTime(metrics.silence_seconds)} accent="sage" detail={`${formatTime(transcript.duration_seconds)} total recording`} />
      </div>

      <div className="summary-band">
        <div className="summary-icon"><Sparkles size={20} /></div>
        <div>
          <p className="mini-label">A note from the room</p>
          <p>{analysis.summary}</p>
        </div>
        <span className="language-badge">{transcript.language || 'mixed'}</span>
      </div>

      <div className="transcript-section" id="transcript">
        <div className="transcript-head">
          <div>
            <p className="kicker"><span /> The conversation</p>
            <h3>Transcript <small>{filteredSegments.length} of {rawSegments.length} moments</small></h3>
          </div>
          <div className="legend">
            <span><i className="teacher-dot" /> teacher</span>
            <span><i className="student-dot" /> student</span>
            <span className="legend-conf">
              <span className="conf-badge conf-high">high</span>
              <span className="conf-badge conf-med">review</span>
              <span className="conf-badge conf-low">low</span>
            </span>
          </div>
        </div>

        {/* Search & Filter Toolbar */}
        <div className="transcript-toolbar">
          <div className="search-wrap">
            <Search size={15} className="search-icon" />
            <input
              type="text"
              placeholder="Search words or terms in transcript..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button className="clear-search" onClick={() => setSearchQuery('')}><X size={14} /></button>
            )}
          </div>

          <div className="filter-pills">
            <button className={`pill ${filter === 'all' ? 'active' : ''}`} onClick={() => setFilter('all')}>All ({rawSegments.length})</button>
            <button className={`pill ${filter === 'teacher' ? 'active' : ''}`} onClick={() => setFilter('teacher')}>Teacher</button>
            <button className={`pill ${filter === 'student' ? 'active' : ''}`} onClick={() => setFilter('student')}>Student</button>
            <button className={`pill ${filter === 'question' ? 'active' : ''}`} onClick={() => setFilter('question')}>Questions</button>
            {correctedCount > 0 && (
              <button className={`pill ${filter === 'corrected' ? 'active' : ''}`} onClick={() => setFilter('corrected')}>Corrected ({correctedCount})</button>
            )}
          </div>
        </div>

        <div className="transcript-list">
          {filteredSegments.length === 0 ? (
            <div className="empty-transcript">
              <p>No moments match your current search or filter.</p>
              <button className="soft-button" onClick={() => { setFilter('all'); setSearchQuery('') }}>Reset filters</button>
            </div>
          ) : (
            filteredSegments.map((seg, i) => (
              <div className={`transcript-row ${seg.speaker} ${seg.suspicious ? 'flagged' : ''}`} key={`${seg.start}-${i}`}>
                <time>{formatClock(seg.start)}</time>
                <div className="speaker-bar" />
                <div className="seg-body">
                  <div className="seg-meta">
                    <span className="speaker-label">{seg.speaker || 'voice'}</span>
                    <ConfidenceBadge value={seg.confidence} />
                    {seg.suspicious && <span className="flag-icon" title="Flagged for review"><AlertTriangle size={12} /></span>}
                    {seg.correction_applied && <span className="fix-icon" title={`Stage 2 corrected from "${seg.original_text}"`}><CheckCircle2 size={12} /></span>}
                  </div>
                  <p>{seg.text}</p>
                  {seg.correction_applied && seg.original_text && (
                    <details className="orig-text">
                      <summary>Stage 1 ASR raw text</summary>
                      <p>"{seg.original_text}"</p>
                    </details>
                  )}
                </div>
                {seg.is_question && <span className="question-tag">question</span>}
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  )
}

function Metric({ label, value, detail, accent }) {
  return (
    <article className={`metric-card ${accent}`}>
      <div className="metric-top">
        <span>{label}</span>
        <span className="metric-spark">✦</span>
      </div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  )
}

export default App