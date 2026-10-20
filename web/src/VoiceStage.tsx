// The voice view: what a call looks like. One big microphone button starts and ends the call, two live meters show who is
// speaking (Falcon above, the customer below), a status line says whose turn it is, and only a short caption of the last line
// is shown; the full transcript, with its badges and trace ids, stays in the chat view.
import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { CallState } from './Call'
import type { Dict } from './i18n'

let sharedCtx: AudioContext | null = null
const audioContext = () => (sharedCtx ??= new AudioContext())

function Spectrum({ stream, color, active }: { stream: MediaStream | null; color: string; active: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const el = canvas.current
    if (!el) return
    const g = el.getContext('2d')!
    const dpr = window.devicePixelRatio || 1
    const w = el.clientWidth, h = el.clientHeight
    el.width = w * dpr; el.height = h * dpr; g.scale(dpr, dpr)
    const BARS = 33, gap = 4, bw = (w - gap * (BARS - 1)) / BARS
    let analyser: AnalyserNode | null = null, source: MediaStreamAudioSourceNode | null = null
    let data = new Uint8Array(0)
    if (stream && stream.getAudioTracks().length) {
      const ctx = audioContext()
      ctx.resume().catch(() => undefined)
      source = ctx.createMediaStreamSource(stream)
      analyser = ctx.createAnalyser()
      analyser.fftSize = 512; analyser.smoothingTimeConstant = 0.75
      source.connect(analyser)  // not to the destination: the <audio> element plays the voice, this only measures it
      data = new Uint8Array(analyser.frequencyBinCount)
    }
    let raf = 0, t = 0
    const draw = () => {
      t += 1
      g.clearRect(0, 0, w, h)
      if (analyser) analyser.getByteFrequencyData(data)
      const half = Math.ceil(BARS / 2)
      for (let i = 0; i < BARS; i++) {
        // mirrored around the centre: the middle bar is the lowest voice band, the edges the highest
        const k = Math.abs(i - (half - 1))
        const bin = 2 + Math.floor((k / half) * 70)  // ~ 90 Hz to 3 kHz at 48 kHz / 512: where speech lives
        const v = analyser ? data[bin] / 255 : 0
        const idle = active ? 0.06 + 0.03 * Math.sin((t + i * 7) / 18) : 0.03  // a breathing line while the call is open
        const bh = Math.max(3, Math.max(v * v * 1.6, idle) * h)
        g.fillStyle = color
        g.globalAlpha = 0.35 + 0.65 * Math.min(1, v * 1.5)
        const x = i * (bw + gap), y = (h - bh) / 2
        g.beginPath(); g.roundRect(x, y, bw, bh, bw / 2); g.fill()
      }
      raf = requestAnimationFrame(draw)
    }
    draw()
    return () => { cancelAnimationFrame(raf); source?.disconnect() }
  }, [stream, color, active])
  return <canvas ref={canvas} className="w-full h-20" aria-hidden />
}

function useTimer(running: boolean) {
  const [secs, setSecs] = useState(0)
  useEffect(() => {
    if (!running) { setSecs(0); return }
    const started = Date.now()
    const id = setInterval(() => setSecs(Math.floor((Date.now() - started) / 1000)), 500)
    return () => clearInterval(id)
  }, [running])
  return `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')}`
}

export default function VoiceStage({ t, state, error, micStream, botStream, onStart, onStop, caption, card }: {
  t: Dict; state: CallState; error: string | null; micStream: MediaStream | null; botStream: MediaStream | null
  onStart: () => void; onStop: () => void; caption?: { who: 'user' | 'assistant'; text: string } | null; card?: ReactNode
}) {
  const active = state !== 'idle' && state !== 'error'
  const time = useTimer(active && state !== 'connecting')
  const speaking = state === 'speaking', hearing = state === 'hearing'
  return (
    <div className="h-full flex flex-col items-center justify-between bg-gradient-to-b from-navy-900 via-navy-800 to-navy-900 text-white px-6 py-8 overflow-y-auto">
      <div className="text-center space-y-1">
        <div className="mx-auto w-16 h-16 rounded-full bg-white/10 flex items-center justify-center text-3xl">🦅</div>
        <div className="font-semibold text-lg">{t.appName}</div>
        <div className="text-sm text-white/60 tabular-nums" dir="ltr">{active && state !== 'connecting' ? time : t.voiceIdle}</div>
      </div>

      <div className="w-full max-w-md space-y-6">
        <div className={`transition-opacity ${speaking ? 'opacity-100' : 'opacity-50'}`}>
          <div className="text-xs uppercase tracking-wide text-white/60 mb-1 text-center">Falcon</div>
          <Spectrum stream={botStream} color="#2dd4bf" active={active} />
        </div>
        <div className={`transition-opacity ${hearing ? 'opacity-100' : 'opacity-50'}`}>
          <Spectrum stream={micStream} color="#f59e0b" active={active} />
          <div className="text-xs uppercase tracking-wide text-white/60 mt-1 text-center">{t.you}</div>
        </div>
      </div>

      <div className="flex flex-col items-center gap-4 w-full max-w-md">
        <div className="min-h-[1.5rem] text-sm text-white/80 text-center">
          {state === 'error' ? <span className="text-red-300">{t.callFailed}{error ? ` (${error})` : ''}</span> : t.voiceStatus[state]}
        </div>
        <button
          onClick={active ? onStop : onStart}
          title={active ? t.endCall : t.call}
          className={`relative w-28 h-28 rounded-full flex items-center justify-center text-5xl shadow-2xl transition
            ${active ? 'bg-red-600 hover:bg-red-700' : 'bg-teal-500 hover:bg-teal-600'}`}>
          {hearing && <span className="absolute inset-0 rounded-full ring-8 ring-amber-400/40 animate-ping" />}
          {state === 'connecting' && <span className="absolute inset-0 rounded-full ring-4 ring-white/30 animate-pulse" />}
          <span className="relative">{active ? '⏹' : '🎙'}</span>
        </button>
        <div className="text-xs text-white/50">{active ? t.endCall : t.call}</div>
        {card && <div className="w-full text-navy-900">{card}</div>}
        {caption?.text && (
          <div className="w-full text-sm text-white/70 text-center line-clamp-2" dir="auto">
            <span className="text-white/40">{caption.who === 'user' ? t.you : 'Falcon'}: </span>{caption.text}
          </div>
        )}
      </div>
    </div>
  )
}
