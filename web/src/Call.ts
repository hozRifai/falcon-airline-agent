// The voice call (docs/VOICE_PLAN.md phase 2): Pipecat's browser client on the SmallWebRTC transport, talking to the voice bot
// (scripts/up.sh --voice, port 7860). The bot is a front end to the same runtime: it calls /api/chat with this browser's session
// token and the chat's thread, so the call continues the conversation on screen. Everything the bot does is mirrored here as
// server messages ({falcon: 'user' | 'done' | 'spoken' | 'call', ...}) and drawn by Chat.tsx like any other turn.
import { useEffect, useRef, useState } from 'react'
import type { PipecatClient } from '@pipecat-ai/client-js'
import { getToken, type Done } from './api'

export const VOICE_URL = (import.meta.env.VITE_VOICE_URL as string | undefined) || 'http://localhost:7860'

// listening: the line is open and it is the customer's turn; hearing: the customer is speaking; thinking: the turn went to the agent
// and nothing is spoken yet; speaking: the bot is talking (a progress phrase, the reply or the card)
export type CallState = 'idle' | 'connecting' | 'listening' | 'hearing' | 'thinking' | 'speaking' | 'error'
// onInterim: the customer's words as Deepgram hears them (null when the turn is over); onReveal: how much of the latest spoken
// reply has been said, 0..1, paced by the words the bot speaks (the text on screen follows the voice, not the network)
export type CallEvents = { onUser: (text: string) => void; onDone: (d: Done) => void; onInterim?: (text: string | null) => void; onReveal?: (fraction: number) => void }

const words = (s: string) => s.split(/\s+/).filter(Boolean).length
type Line = { kind: string; total: number; said: number }

export function useCall(ev: CallEvents) {
  const [state, setState] = useState<CallState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [micStream, setMicStream] = useState<MediaStream | null>(null)  // the customer's microphone, for the voice view's meter
  const [botStream, setBotStream] = useState<MediaStream | null>(null)  // the bot's voice, played and metered
  const client = useRef<PipecatClient | null>(null)
  const audio = useRef<HTMLAudioElement | null>(null)
  const handlers = useRef(ev)
  handlers.current = ev
  // the lines the bot announced and is speaking, in order; each spoken word (bot-tts-text) is counted against the first one
  const lines = useRef<Line[]>([])
  const said = (n: number) => {
    while (n > 0 && lines.current.length) {
      const line = lines.current[0]
      const take = Math.min(n, line.total - line.said)
      line.said += take; n -= take
      if (line.kind === 'reply') handlers.current.onReveal?.(line.total ? line.said / line.total : 1)
      if (line.said >= line.total) lines.current.shift()
    }
  }
  const flush = () => {  // speech ended (or was interrupted): whatever was announced is shown in full
    if (lines.current.some((l) => l.kind === 'reply')) handlers.current.onReveal?.(1)
    lines.current = []
  }

  useEffect(() => () => { client.current?.disconnect().catch(() => undefined) }, [])

  const start = async (thread: string, lang: string) => {
    if (client.current) return
    setError(null); setState('connecting')
    if (!audio.current) { audio.current = new Audio(); audio.current.autoplay = true }
    // loaded on the first call only: the client and its transport are most of the bundle, and most visitors never call
    const [{ PipecatClient }, { SmallWebRTCTransport }] = await Promise.all([import('@pipecat-ai/client-js'), import('@pipecat-ai/small-webrtc-transport')])
    const c = new PipecatClient({
      transport: new SmallWebRTCTransport(),
      enableMic: true,
      enableCam: false,
      callbacks: {
        onTrackStarted: (track: MediaStreamTrack, participant?: { local: boolean }) => {
          if (track.kind !== 'audio') return
          if (participant?.local) { setMicStream(new MediaStream([track])); return }
          // the bot's voice: a hidden <audio> element plays it (Chrome only feeds a remote WebRTC track to Web Audio while it plays)
          const stream = new MediaStream([track])
          setBotStream(stream)
          if (audio.current) { audio.current.srcObject = stream; audio.current.play().catch(() => undefined) }
        },
        onUserStartedSpeaking: () => setState((s) => (s === 'speaking' ? s : 'hearing')),
        onUserStoppedSpeaking: () => setState((s) => (s === 'hearing' ? 'thinking' : s)),
        onUserTranscript: (d: { text: string; final: boolean }) => { if (d.text) handlers.current.onInterim?.(d.text) },
        onBotStartedSpeaking: () => setState('speaking'),
        onBotStoppedSpeaking: () => { flush(); setState('listening') },
        onBotTtsText: (d: { text: string }) => said(words(d.text || '')),
        onBotReady: () => setState('listening'),
        onServerMessage: (data: any) => {
          if (data?.falcon === 'user' && data.text) { handlers.current.onInterim?.(null); handlers.current.onUser(String(data.text)) }
          else if (data?.falcon === 'done') handlers.current.onDone(data as Done)
          else if (data?.falcon === 'spoken') {
            lines.current.push({ kind: String(data.kind), total: Math.max(1, words(String(data.spoken || ''))), said: 0 })
            if (data.kind === 'reply') handlers.current.onReveal?.(0)
          }
        },
        onDisconnected: () => { client.current = null; flush(); handlers.current.onInterim?.(null); setMicStream(null); setBotStream(null); setState((s) => (s === 'error' ? s : 'idle')) },
        onError: (m: unknown) => { setError(String((m as any)?.data?.message ?? (m as any)?.data?.error ?? 'voice error')); setState('error') },
      },
    })
    client.current = c
    try {
      await c.startBotAndConnect({ endpoint: `${VOICE_URL}/start`, requestData: { transport: 'webrtc', body: { token: getToken(), thread, lang } } })
    } catch (e: any) {
      client.current = null
      setError(String(e?.message ?? e)); setState('error')
    }
  }

  const stop = async () => {
    const c = client.current
    client.current = null
    await c?.disconnect().catch(() => undefined)
    setMicStream(null); setBotStream(null)
    setState('idle')
  }

  const active = state !== 'idle' && state !== 'error'
  return { state, error, start, stop, active, micStream, botStream }
}
