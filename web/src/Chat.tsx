import { useEffect, useRef, useState } from 'react'
import { api, chat, confirmAction, openChat, type Done, type Guardrail, type Pending, type ReviewEvent, type ToolEntry, type Verification } from './api'
import { useCall } from './Call'
import VoiceStage from './VoiceStage'
import { isArabic, type Dict } from './i18n'

type Msg = { role: 'user' | 'assistant'; text: string; checked?: ToolEntry[]; meta?: { latency_ms: number; model: string; usage: Done['usage'] }; verification?: Verification | null; guardrail?: Guardrail | null; turnId?: number | null; traceId?: string | null; voice?: boolean }
type Live = { text: string; checked: ToolEntry[]; review?: ReviewEvent | null }
const WRITES = new Set(['cancel_booking', 'change_booking', 'add_extra_bag', 'request_exception'])
// the first `fraction` of a text's words, whitespace and line breaks kept: a spoken reply appears as it is said
export const revealText = (text: string, fraction: number | null | undefined) => {
  if (fraction == null || fraction >= 1) return text
  const parts = text.split(/(\s+)/)
  const total = parts.filter((p) => p.trim()).length
  let keep = Math.ceil(total * Math.max(0, fraction)), out = ''
  for (const p of parts) { if (p.trim()) { if (keep-- <= 0) break } out += p }
  return out.trimEnd()
}

export default function Chat({ t, role, lang, prefill, onPrefillUsed, openKey, onWrite }: { t: Dict; role: string; lang: string; prefill?: string; onPrefillUsed?: () => void; openKey?: number; onWrite?: () => void }) {
  const [thread, setThread] = useState(() => localStorage.getItem('falcon.thread') || 'main')
  const [messages, setMessages] = useState<Msg[]>([])
  const [pending, setPending] = useState<Pending | null>(null)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [live, setLive] = useState<Live | null>(null)
  const [error, setError] = useState<string | null>(null)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.history(thread).then((h) => {
      setMessages(h.messages.map((m) => ({ role: m.role as Msg['role'], text: m.text })))
      setPending(h.pending)
    }).catch(() => undefined)
  }, [thread])

  useEffect(() => { if (prefill) { setInput(prefill); onPrefillUsed?.() } }, [prefill, onPrefillUsed])
  // a freshly chosen customer: the agent speaks first (greets by name, checks the bookings, flags what needs attention)
  useEffect(() => { if (openKey) { setMessages([]); setPending(null); run((ev) => openChat(thread, ev, lang)) } }, [openKey, lang])  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, live, pending])

  // a finished turn, typed or spoken (voice phase 2: the call's turns are drawn by the same code, with the same badges and trace)
  const applyDone = (d: Done, voice = false) => {
    if (d.message) setMessages((m) => [...m, { role: 'assistant', text: d.message, checked: d.what_i_checked, meta: { latency_ms: d.latency_ms, model: d.model, usage: d.usage }, verification: d.verification, guardrail: d.guardrail, turnId: d.turn_id, traceId: d.trace_id, voice }])
    else if (!d.pending && d.what_i_checked.length) setMessages((m) => [...m, { role: 'assistant', text: '', checked: d.what_i_checked, voice }])
    setPending(d.pending)
    if (d.what_i_checked.some((e) => e.ok && WRITES.has(e.tool))) onWrite?.()  // the trips panel reloads after a cancellation, change or bag
  }
  // two ways to show the same conversation: the chat (transcript, badges, cards) and the voice view (a call: microphone, meters)
  const [mode, setMode] = useState<'chat' | 'voice'>(() => (localStorage.getItem('falcon.mode') === 'voice' ? 'voice' : 'chat'))
  const switchMode = (m: 'chat' | 'voice') => { setMode(m); localStorage.setItem('falcon.mode', m) }
  const [interim, setInterim] = useState<string | null>(null)  // the customer's words while they speak on a call
  const [reveal, setReveal] = useState<number | null>(null)     // how much of the latest spoken reply has been said
  const call = useCall({
    onUser: (text) => { setPending(null); setMessages((m) => [...m, { role: 'user', text, voice: true }]) },
    onDone: (d) => { if (d.message) setReveal(0); applyDone(d, true) },  // shown as it is spoken, from the first word
    onInterim: setInterim,
    onReveal: (f) => setReveal(f >= 1 ? null : f),
  })

  const run = (fn: (ev: Parameters<typeof chat>[2]) => Promise<void>) => {
    setBusy(true); setError(null); setLive({ text: '', checked: [] })
    return fn({
      onToken: (tk) => setLive((l) => ({ text: (l?.text || '') + tk, checked: l?.checked || [], review: l?.review })),
      onTool: (e) => setLive((l) => ({ text: l?.text || '', checked: [...(l?.checked || []), e], review: l?.review })),
      // the reviewer sent the streamed draft back: drop it and say why; the next draft streams in its place
      onReview: (r) => setLive((l) => ({ text: '', checked: l?.checked || [], review: r })),
      onConfirm: (p) => setPending(p),
      onDone: (d) => { setLive(null); setBusy(false); applyDone(d) },
      onError: (e) => { setLive(null); setBusy(false); setError(e) },
    })
  }

  const send = (text?: string) => {
    const msg = (text ?? input).trim()
    if (!msg || busy || call.active) return
    setInput('')
    setMessages((m) => [...m, { role: 'user', text: msg }])
    run((ev) => chat(msg, thread, ev, lang))
  }
  const decide = (approved: boolean) => { setPending(null); run((ev) => confirmAction(approved, approved ? '' : t.declinedNote, thread, ev)) }
  const newChat = () => { const id = 't' + Date.now().toString(36); localStorage.setItem('falcon.thread', id); setThread(id); setMessages([]); setPending(null) }

  const lastIdx = messages.length - 1
  const liveReveal = (i: number) => (i === lastIdx && messages[i]?.voice && messages[i]?.role === 'assistant' ? reveal : null)
  const lastI = messages.map((m) => !!m.text).lastIndexOf(true)
  const last = lastI >= 0 ? messages[lastI] : null
  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-center gap-1 p-1.5 border-b border-sand-200 bg-white">
        {(['chat', 'voice'] as const).map((m) => (
          <button key={m} className={`chip px-3 py-1 text-sm ${mode === m ? 'bg-navy-800 text-white' : 'bg-sand-100 text-navy-800 hover:bg-sand-200'}`} onClick={() => switchMode(m)}>
            {m === 'chat' ? `💬 ${t.modeChat}` : `🎙 ${t.modeVoice}`}
          </button>
        ))}
      </div>
      {mode === 'voice' ? (
        <div className="flex-1 min-h-0">
          <VoiceStage t={t} state={call.state} error={call.error} micStream={call.micStream} botStream={call.botStream}
            onStart={() => call.start(thread, lang)} onStop={() => call.stop()}
            caption={interim ? { who: 'user', text: interim } : last ? { who: last.role, text: revealText(last.text, liveReveal(lastI)) } : null}
            card={pending ? <ConfirmCard p={pending} t={t} onDecide={decide} voice={call.active} /> : null} />
        </div>
      ) : (<>
      <div className="flex-1 overflow-y-auto space-y-3 p-3">
        {messages.map((m, i) => <Bubble key={i} m={m} t={t} thread={thread} reveal={liveReveal(i)} />)}
        {interim && <div className="flex justify-end"><div className="max-w-[85%] rounded-2xl bg-navy-800/60 text-white px-4 py-2 italic" dir={isArabic(interim) ? 'rtl' : 'ltr'}><span className="opacity-60 me-1">🎙</span>{interim}</div></div>}
        {!messages.some((m) => m.role === 'user') && !live && !pending && (
          <div className="flex flex-wrap gap-2 pt-1">
            {(t.quick[role] || []).map((q) => <button key={q} className="chip bg-sand-100 text-navy-800 hover:bg-sand-200 py-1.5 px-3 text-sm" dir={isArabic(q) ? 'rtl' : 'ltr'} onClick={() => send(q)}>{q}</button>)}
          </div>
        )}
        {live && (
          <div className="space-y-2">
            {live.checked.map((e, i) => <ToolRow key={i} e={e} />)}
            {live.review && (
              <div className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-2.5 py-1.5">
                ↩ {t.reviewSentBack} ({t.round} {live.review.round}): {live.review.problems.map((p) => t.problems[p] || p).join(', ')}
              </div>
            )}
            <div className="card inline-block px-4 py-3" dir={isArabic(live.text) ? 'rtl' : 'ltr'}>
              {live.text ? <span className="whitespace-pre-wrap">{live.text}</span> : <span className="text-navy-700"><span className="dot">●</span> <span className="dot">●</span> <span className="dot">●</span></span>}
            </div>
          </div>
        )}
        {pending && <ConfirmCard p={pending} t={t} onDecide={decide} voice={call.active} />}
        {error && <div className="rounded-xl bg-red-50 text-red-700 px-3 py-2 text-sm">{error}</div>}
        {call.state === 'error' && <div className="rounded-xl bg-red-50 text-red-700 px-3 py-2 text-sm">{t.callFailed}{call.error ? ` (${call.error})` : ''}</div>}
        <div ref={bottom} />
      </div>
      <form className="flex gap-2 p-3 border-t border-sand-200 bg-white" onSubmit={(e) => { e.preventDefault(); send() }}>
        <button type="button" className="btn-ghost text-sm" title={t.newChat} onClick={newChat}>⟳</button>
        {/* one channel at a time on a thread: typing is off during a call, so two turns never run on the same conversation */}
        <button type="button" className={call.active ? 'btn-primary px-3 whitespace-nowrap' : 'btn-ghost px-3'} title={call.active ? t.endCall : t.call} disabled={busy && !call.active}
          onClick={() => (call.active ? call.stop() : call.start(thread, lang))}>
          {call.active ? `⏹ ${t.callState[call.state]}` : '📞'}
        </button>
        <input className="input" value={input} onChange={(e) => setInput(e.target.value)} placeholder={call.active ? t.inCall : busy ? t.thinking : t.typeMessage} disabled={call.active} dir={isArabic(input) ? 'rtl' : 'ltr'} autoFocus />
        <button className="btn-primary" disabled={busy || call.active || !input.trim()}>{t.send}</button>
      </form>
      </>)}
    </div>
  )
}

function Bubble({ m, t, thread, reveal }: { m: Msg; t: Dict; thread: string; reveal?: number | null }) {
  const [open, setOpen] = useState(false)
  const [rated, setRated] = useState<'up' | 'down' | null>(null)
  const rtl = isArabic(m.text)
  if (m.role === 'user') return <div className="flex justify-end"><div className="max-w-[85%] rounded-2xl bg-navy-800 text-white px-4 py-2 whitespace-pre-wrap" dir={rtl ? 'rtl' : 'ltr'}>{m.voice && <span className="opacity-60 me-1" title={t.spoken}>🎙</span>}{m.text}</div></div>
  const rate = (rating: 'up' | 'down') => { setRated(rating); api.feedback({ rating, thread, turn_id: m.turnId ?? null, message: m.text }).catch(() => undefined) }
  const v = m.verification
  return (
    <div className="max-w-[92%] space-y-1">
      {m.text && <div className="card px-4 py-3 whitespace-pre-wrap" dir={rtl ? 'rtl' : 'ltr'}>{revealText(m.text, reveal)}{reveal != null && <span className="dot"> ●</span>}</div>}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        {m.voice && <span className="text-gray-500" title={t.spoken}>🔊 {t.spoken}</span>}
        {v && <VerificationBadge v={v} t={t} />}
        {m.guardrail && <GuardrailBadge g={m.guardrail} t={t} />}
        {m.checked && m.checked.length > 0 && (
          <button className="text-navy-700 underline underline-offset-2" onClick={() => setOpen(!open)}>{t.whatIChecked} ({m.checked.length}){m.meta ? ` · ${m.meta.latency_ms} ${t.ms}` : ''}</button>
        )}
        {m.traceId && <TraceLink id={m.traceId} />}
        {m.text && (rated
          ? <span className="text-gray-500">{t.thanks}</span>
          : <span className="inline-flex gap-1">
              <button className="text-gray-500 hover:text-teal-600" title={t.thumbsUp} onClick={() => rate('up')}>👍</button>
              <button className="text-gray-500 hover:text-red-600" title={t.thumbsDown} onClick={() => rate('down')}>👎</button>
            </span>)}
      </div>
      {open && m.checked && <div className="mt-1 space-y-1">{m.checked.map((e, i) => <ToolRow key={i} e={e} />)}</div>}
    </div>
  )
}

function TraceLink({ id }: { id: string }) {
  // the presenter's shortcut into the trace UI (decision #27); the id is also on agent.turns, so it works without a UI too
  const base = (window as any).__falconTraceUi as string | undefined
  const short = id.slice(0, 8)
  const copy = () => navigator.clipboard?.writeText(id).catch(() => undefined)
  return base
    ? <a className="text-gray-400 hover:text-navy-700 font-mono" href={base} target="_blank" rel="noreferrer" title={id} onClick={copy}>trace {short}</a>
    : <button className="text-gray-400 hover:text-navy-700 font-mono" title={`${id} (click to copy)`} onClick={copy}>trace {short}</button>
}

function VerificationBadge({ v, t }: { v: Verification; t: Dict }) {
  if (v.status === 'held') return <span className="rounded-full bg-amber-50 border border-amber-200 text-amber-800 px-2 py-0.5">⚠ {t.reviewHeld}</span>
  if (v.status === 'escalated') return <span className="rounded-full bg-amber-50 border border-amber-200 text-amber-800 px-2 py-0.5">⚠ {t.escalated}</span>
  if (v.status === 'cut_off') return null
  const sentBack = v.rounds.length > 0
  if (!v.evidence && !v.amounts.length && !sentBack) return null
  const title = sentBack ? `${t.reviewSentBack}: ${v.rounds.map((r) => r.problems.map((p) => t.problems[p] || p).join(', ')).join(' | ')}` : undefined
  return (
    <span className="rounded-full bg-teal-500/10 border border-teal-500/30 text-teal-700 px-2 py-0.5" title={title}>
      ✓ {t.verified} {v.evidence} {t.toolResults}{v.amounts.length ? ` · ${v.amounts.length} ${t.amountsChecked}` : ''}{sentBack ? ` · ↩ ${v.rounds.length}` : ''}
    </span>
  )
}

function GuardrailBadge({ g, t }: { g: Guardrail; t: Dict }) {
  // the input guardrail (decision #29): what it flagged, blocked or removed from the customer's message; silent when nothing happened
  const redacted = (g.redactions?.length ?? 0) > 0
  if (g.action === 'allow' && !redacted) return null
  const parts = [g.action === 'allow' ? '' : (t.guardrail[g.label] || g.label), redacted ? t.guardrail.redacted : ''].filter(Boolean).join(' · ')
  const cls = g.action === 'block' ? 'bg-red-50 border-red-200 text-red-800' : 'bg-amber-50 border-amber-200 text-amber-800'
  return <span className={`rounded-full border px-2 py-0.5 ${cls}`} title={g.reason}>🛡 {parts}</span>
}

function ToolRow({ e }: { e: ToolEntry }) {
  const color = e.declined ? 'bg-amber-50 border-amber-200' : e.ok ? 'bg-teal-500/5 border-teal-500/30' : 'bg-red-50 border-red-200'
  return (
    <div className={`rounded-lg border px-2.5 py-1.5 text-xs font-mono ${color}`} dir="ltr">
      <span className="font-semibold">{e.tool}</span>
      <span className="text-gray-500"> {JSON.stringify(e.args)}</span>
      <span className="text-gray-400"> · {e.latency_ms} ms</span>
      <div className="text-gray-600 truncate">{e.summary}</div>
    </div>
  )
}

function ConfirmCard({ p, t, onDecide, voice }: { p: Pending; t: Dict; onDecide: (ok: boolean) => void; voice?: boolean }) {
  const ar = p.lang === 'ar'
  const label = (tool: string) => ({ create_booking: ar ? 'حجز هذه الرحلة' : 'Book this flight', cancel_booking: ar ? 'إلغاء الحجز' : 'Cancel booking', change_booking: ar ? 'تغيير الرحلة' : 'Change flight', add_extra_bag: ar ? 'إضافة أمتعة' : 'Add baggage', request_exception: ar ? 'تسجيل حالة استثناء لفريق خدمة العملاء' : 'Log an exception case for the customer-care team' } as Record<string, string>)[tool] || tool
  return (
    <div className="card border-ember-500/50 border-2 p-4 space-y-3" dir={ar ? 'rtl' : 'ltr'}>
      <div className="font-semibold text-navy-900">{t.confirmTitle}</div>
      {p.actions.map((a, i) => (
        <div key={i} className="space-y-1">
          <div className="font-medium">{label(a.tool)} {a.args.pnr ? <span className="font-mono">({String(a.args.pnr)})</span> : null}</div>
          {a.quote?.flights?.map((f) => (
            <div key={f.seq} className="text-sm text-gray-800" dir="ltr">
              ✈ <span className="font-mono">{f.flight_number}</span> {f.origin} → {f.destination} · {f.departure}
              {a.quote?.kind === 'change' && a.quote.details?.new_flight_instance_id ? <span className="text-gray-500"> → {String(a.quote.details.new_flight_instance_id)}</span> : null}
            </div>
          ))}
          {a.tool === 'request_exception' && a.quote?.exception_if_approved && (
            <div className="text-sm text-gray-700">{t.ifApproved}: <b>{a.quote.exception_if_approved.refund} {a.quote.currency}</b> · {t.fee} {a.quote.exception_if_approved.fee} {a.quote.currency}</div>
          )}
          {a.quote && a.tool !== 'request_exception' && (
            <div className="text-sm text-gray-700">
              {a.quote.kind === 'cancel' && (Number(a.quote.refund) > 0
                ? <span>{t.youGetBack} <b>{a.quote.refund} {a.quote.currency}</b> ({a.quote.refund_method === 'credit' ? (ar ? 'رصيد سفر' : 'travel credit') : (ar ? 'إلى البطاقة' : 'to your card')}) · {t.fee} {a.quote.fee} {a.quote.currency}</span>
                : <span className="text-amber-800">{t.noRefund}</span>)}
              {a.quote.kind !== 'cancel' && <span>{t.amountDue} <b>{a.quote.amount_due} {a.quote.currency}</b>{Number(a.quote.fee) > 0 ? ` · ${t.fee} ${a.quote.fee} ${a.quote.currency}` : ''}</span>}
            </div>
          )}
          {a.args.reason ? <div className="text-sm text-gray-600">{t.reason}: {String(a.args.reason)}</div> : null}
        </div>
      ))}
      {/* on a call the decision is spoken (VOICE_PLAN rule 3): the buttons would race the call's own pending card */}
      {voice
        ? <div className="text-sm text-navy-700">🎙 {t.sayYesNo}</div>
        : <div className="flex gap-2">
            <button className="btn-teal" onClick={() => onDecide(true)}>{t.confirmYes}</button>
            <button className="btn-ghost" onClick={() => onDecide(false)}>{t.confirmNo}</button>
          </div>}
    </div>
  )
}
