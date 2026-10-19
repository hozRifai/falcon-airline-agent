// API client. The token is a signed session token from the demo sign-in; it goes in the Authorization header.
// Chat answers stream as Server-Sent Events: token | tool | review | confirm | done.

export type User = { user_id: string; display_name: string; role: 'customer'; customer_id: string | null; lang: string; session_id: string }
export type Persona = { user_id: string; display_name: string; display_name_ar: string; role: string; customer_id: string | null; lang: string; persona_note_en: string; persona_note_ar: string; sort_order: number }
export type ToolEntry = { tool: string; args: Record<string, unknown>; ok: boolean; latency_ms: number; summary: string; declined?: boolean }
export type Flight = { seq: number; flight_number: string; origin: string; destination: string; departure: string }
export type Quote = Record<string, any> & { flights?: Flight[] }
export type Pending = { type: 'confirm'; actions: { tool: string; args: Record<string, unknown>; quote: Quote | null }[]; lang: string }
export type Verification = { status: 'verified' | 'retried' | 'held' | 'cut_off' | 'escalated'; case?: number; rounds: { round: number; problems: string[]; reviewer_note?: string }[]; amounts: string[]; evidence: number; checked: boolean; reviewer: string }
export type ReviewEvent = { type: 'review'; round: number; problems: string[]; final: boolean }
export type Guardrail = { label: string; action: string; source: string; reason?: string; redactions?: string[]; latency_ms?: number }
export type Done = { message: string; pending: Pending | null; lang: string; what_i_checked: ToolEntry[]; usage: { input_tokens: number; output_tokens: number }; latency_ms: number; model: string; verification?: Verification | null; guardrail?: Guardrail | null; turn_id?: number | null; trace_id?: string | null }

const TOKEN_KEY = 'falcon.token'
export const getToken = () => localStorage.getItem(TOKEN_KEY)
export const setToken = (t: string | null) => (t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY))

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...(init.headers as Record<string, string> | undefined) }
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`
  const r = await fetch(path, { ...init, headers })
  if (r.status === 401) { setToken(null); location.reload() }
  if (!r.ok) {
    let detail = r.statusText
    try { detail = (await r.json()).detail ?? detail } catch { /* ignore */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return r.json() as Promise<T>
}

export const api = {
  personas: () => req<{ personas: Persona[]; frozen_now: string; model: string; trace_ui_url?: string | null; requires_code?: boolean }>('/api/personas'),
  switch: (user_id: string, reset = true) => req<{ token: string; user: User; persona: Persona; reset: boolean }>('/api/demo/switch', { method: 'POST', body: JSON.stringify({ user_id, reset }) }),
  upload: async (file: File) => {
    // a document attached in the chat (decision #33): multipart, the session token, no JSON content type
    const fd = new FormData()
    fd.append('file', file)
    const headers: Record<string, string> = {}
    const token = getToken()
    if (token) headers.Authorization = `Bearer ${token}`
    const r = await fetch('/api/attachments', { method: 'POST', body: fd, headers })
    if (!r.ok) throw new Error((await r.json().catch(() => ({})))?.detail || `upload failed (${r.status})`)
    return r.json() as Promise<{ id: number; filename: string; size_bytes: number }>
  },
  login: (user_id: string, access_code: string) => req<{ token: string; user: User }>('/api/auth/demo-login', { method: 'POST', body: JSON.stringify({ user_id, access_code }) }),
  me: () => req<User>('/api/me'),
  bookings: () => req<{ bookings: Record<string, any>[] }>('/api/bookings'),
  search: (q: Record<string, string>) => req<Record<string, any>>('/api/search?' + new URLSearchParams(q).toString()),
  fareRules: () => req<Record<string, any>>('/api/fare-rules'),
  history: (thread: string) => req<{ messages: { role: string; text: string }[]; pending: Pending | null; lang: string }>(`/api/chat/history?thread=${thread}`),
  feedback: (body: { rating: 'up' | 'down'; thread: string; turn_id?: number | null; message?: string }) => req<{ ok: boolean; feedback_id: number }>('/api/feedback', { method: 'POST', body: JSON.stringify(body) }),
}

export type ChatEvents = { onToken?: (t: string) => void; onTool?: (e: ToolEntry) => void; onConfirm?: (p: Pending) => void; onReview?: (r: ReviewEvent) => void; onDone?: (d: Done) => void; onError?: (e: string) => void }

async function stream(path: string, body: Record<string, unknown>, ev: ChatEvents) {
  let r: Response
  try {
    r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${getToken()}` }, body: JSON.stringify(body) })
  } catch (e) {
    ev.onError?.(`Could not reach the assistant: ${String((e as Error).message)}`)
    return
  }
  if (!r.ok || !r.body) {
    let detail = r.statusText
    try { detail = (await r.json()).detail ?? detail } catch { /* ignore */ }
    ev.onError?.(String(detail))
    return
  }
  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let event = ''
  let gotDone = false
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).replace(/\r$/, '')
        buffer = buffer.slice(idx + 1)
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) {
          const data = JSON.parse(line.slice(5).trim())
          if (event === 'token') ev.onToken?.(data.text)
          else if (event === 'tool') ev.onTool?.(data)
          else if (event === 'confirm') ev.onConfirm?.(data)
          else if (event === 'review') ev.onReview?.(data)
          else if (event === 'done') { gotDone = true; ev.onDone?.(data) }
        }
      }
    }
  } catch (e) {
    ev.onError?.(`The connection to the assistant dropped: ${String((e as Error).message)}`)
    return
  }
  // the server closed the stream without a final event (an error after the reply, a restart): release the UI with a message
  if (!gotDone) ev.onError?.('The assistant did not finish this reply (the server closed the connection). Please try again; if it keeps happening, run scripts/reset.sh: the database schema may be older than the code.')
}

export const chat = (message: string, thread: string, ev: ChatEvents, lang?: string) => stream('/api/chat', { message, thread, lang }, ev)
export const openChat = (thread: string, ev: ChatEvents, lang?: string) => stream('/api/chat/open', { thread, lang }, ev)  // the app's language drives the opening (decision #40)
export const confirmAction = (approved: boolean, note: string, thread: string, ev: ChatEvents) => stream('/api/chat/confirm', { approved, note, thread }, ev)
