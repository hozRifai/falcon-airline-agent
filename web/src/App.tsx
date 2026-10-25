import { useCallback, useEffect, useState } from 'react'
import { api, setToken, type Persona, type User } from './api'
import { dictionary, type Lang } from './i18n'
import Chat from './Chat'
import { Search, Trips } from './Customer'

// The demo console. Left: the customers (a list while choosing, a thin rail of initials once chosen). Middle: the
// conversation. Right (optional): what the agent sees about this customer. Choosing a customer resets the demo data,
// signs in as them and lets the agent open the conversation; "focus" hides everything but the conversation.

type Sidebar = 'open' | 'rail'

export default function App() {
  const [lang, setLang] = useState<Lang>(() => (localStorage.getItem('falcon.lang') as Lang) || 'en')
  const [personas, setPersonas] = useState<Persona[]>([])
  const [meta, setMeta] = useState<{ frozen_now: string; model: string } | null>(null)
  const [user, setUser] = useState<User | null>(null)
  const [persona, setPersona] = useState<Persona | null>(null)
  const [switching, setSwitching] = useState<string | null>(null)
  const [sidebar, setSidebar] = useState<Sidebar>('open')
  const [focus, setFocus] = useState(false)          // full-width conversation: no side panel
  const [drawer, setDrawer] = useState(false)        // small screens: the customer list as an overlay
  const [openKey, setOpenKey] = useState(0)          // bumps when a new conversation should start with the agent's opening line
  const [tripsKey, setTripsKey] = useState(0)        // bumps when the trips panel should reload (after a write)
  const [panel, setPanel] = useState<'trips' | 'search'>('trips')
  const [prefill, setPrefill] = useState<string | undefined>()  // a question the side panel puts in the chat box
  const [error, setError] = useState<string | null>(null)
  const t = dictionary(lang)
  const name = (p: Persona) => p.display_name  // names stay as the booking holds them (Latin script) in both languages, as an airline's systems do (Hozayfa, 22 Oct)
  const note = (p: Persona) => (lang === 'ar' ? p.persona_note_ar : p.persona_note_en)
  const initials = (p: Persona) => name(p).split(' ').map((w) => w[0]).slice(0, 2).join('')

  useEffect(() => {
    document.documentElement.lang = lang
    document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr'
    localStorage.setItem('falcon.lang', lang)
  }, [lang])

  useEffect(() => {
    setToken(null); localStorage.removeItem('falcon.thread')  // every page load starts clean: the demo always begins by picking a customer
    api.personas().then((r) => { setPersonas(r.personas as Persona[]); setMeta(r); (window as any).__falconTraceUi = r.trace_ui_url || undefined }).catch((e) => setError(String(e.message)))
  }, [])

  const choose = useCallback(async (p: Persona) => {
    if (switching) return
    setSwitching(p.user_id); setError(null); setDrawer(false)
    try {
      const r = await api.switch(p.user_id)           // resets the demo data, signs in as this customer, fresh session
      setToken(r.token); localStorage.setItem('falcon.thread', 'main')
      setUser(r.user); setPersona(r.persona as Persona)
      setTripsKey((k) => k + 1); setOpenKey((k) => k + 1)
      setSidebar('rail')                                // the conversation takes the room; the rail keeps switching one click away
    } catch (e) { setError(String((e as Error).message)) } finally { setSwitching(null) }
  }, [switching])

  const onWrite = useCallback(() => setTripsKey((k) => k + 1), [])

  const CustomerList = ({ compact }: { compact: boolean }) => (
    <div className={`flex flex-col ${compact ? 'items-center gap-1.5 py-2' : 'gap-1.5 p-2'}`}>
      {personas.map((p) => {
        const active = user?.user_id === p.user_id
        const busy = switching === p.user_id
        if (compact) {
          return (
            <button key={p.user_id} onClick={() => choose(p)} disabled={!!switching} title={`${name(p)} — ${note(p)}`}
              className={`w-10 h-10 rounded-full grid place-items-center text-xs font-semibold transition ${active ? 'bg-navy-900 text-white ring-2 ring-teal-500 ring-offset-2 ring-offset-sand-50' : 'bg-white border border-sand-200 text-navy-800 hover:border-teal-500'} ${switching && !busy ? 'opacity-60' : ''}`}>
              {busy ? <span className="dot">●</span> : initials(p)}
            </button>
          )
        }
        return (
          <button key={p.user_id} onClick={() => choose(p)} disabled={!!switching}
            className={`w-full text-start rounded-xl border p-2.5 flex gap-3 items-start transition ${active ? 'bg-navy-900 text-white border-navy-900' : 'bg-white border-sand-200 hover:border-teal-500'} ${switching && !busy ? 'opacity-60' : ''}`}>
            <span className={`shrink-0 w-9 h-9 rounded-full grid place-items-center text-xs font-semibold ${active ? 'bg-white/15' : 'bg-sand-100 text-navy-800'}`}>{busy ? <span className="dot">●</span> : initials(p)}</span>
            <span className="min-w-0">
              <span className="block font-medium leading-tight">{name(p)}</span>
              <span className={`block text-xs leading-snug line-clamp-2 ${active ? 'text-white/75' : 'text-gray-600'}`}>{note(p)}</span>
              <span className={`block text-[10px] mt-0.5 ${active ? 'text-white/50' : 'text-gray-400'}`} dir="ltr">{p.lang === 'ar' ? 'العربية' : 'English'} · {p.customer_id}</span>
            </span>
          </button>
        )
      })}
    </div>
  )

  return (
    <div className="h-full flex flex-col bg-sand-50">
      <header className="bg-navy-900 text-white shrink-0">
        <div className="px-3 py-2 flex items-center gap-3">
          <button className="chip bg-white/15 text-white hover:bg-white/25 lg:hidden" onClick={() => setDrawer(!drawer)}>☰ {t.customers}</button>
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-xl">🦅</span>
            <div className="min-w-0"><div className="font-semibold leading-tight truncate">{t.appName}</div><div className="text-[11px] text-white/60 leading-tight truncate">{t.tagline}</div></div>
          </div>
          <div className="flex-1" />
          {user && persona && (
            <div className="hidden sm:flex items-center gap-2 text-sm text-white/85 min-w-0">
              <span className="w-7 h-7 rounded-full bg-white/15 grid place-items-center text-[11px] font-semibold shrink-0">{initials(persona)}</span>
              <span className="truncate">{name(persona)}</span>
            </div>
          )}
          <button className={`chip hover:bg-white/25 hidden lg:inline-flex ${focus ? 'bg-white text-navy-900' : 'bg-white/15 text-white'}`} title={t.focusMode} onClick={() => setFocus(!focus)}>{focus ? '⤡' : '⤢'} {t.focus}</button>
          <button className="chip bg-white/15 text-white hover:bg-white/25" onClick={() => setLang(lang === 'en' ? 'ar' : 'en')}>{lang === 'en' ? 'العربية' : 'English'}</button>
        </div>
      </header>

      <div className="flex-1 min-h-0 flex">
        {/* customers: a list while choosing, a rail afterwards (large screens); an overlay drawer on small screens */}
        <aside className={`hidden lg:flex flex-col shrink-0 border-e border-sand-200 bg-sand-50 transition-all ${sidebar === 'open' ? 'w-72' : 'w-16'}`}>
          <div className={`flex items-center ${sidebar === 'open' ? 'justify-between px-3' : 'justify-center'} py-2 border-b border-sand-200`}>
            {sidebar === 'open' && <span className="flex-1 text-center text-sm font-bold text-navy-900">{t.customers}</span>}
            <button className="text-gray-500 hover:text-navy-900 text-sm px-1" title={sidebar === 'open' ? t.collapse : t.expand} onClick={() => setSidebar(sidebar === 'open' ? 'rail' : 'open')}>{sidebar === 'open' ? '⟨' : '⟩'}</button>
          </div>
          <div className="flex-1 overflow-y-auto"><CustomerList compact={sidebar === 'rail'} /></div>
          {meta && sidebar === 'open' && <div className="text-[10px] text-gray-500 p-2 border-t border-sand-200" dir="ltr">{t.frozenClock}: {meta.frozen_now}<br />{t.model}: {meta.model}</div>}
        </aside>
        {drawer && (
          <div className="lg:hidden fixed inset-0 z-20 flex" onClick={() => setDrawer(false)}>
            <div className="w-80 max-w-[85%] h-full bg-sand-50 shadow-xl overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="px-3 py-2 text-sm font-bold text-navy-900 text-center border-b border-sand-200">{t.customers}</div>
              <CustomerList compact={false} />
            </div>
            <div className="flex-1 bg-black/30" />
          </div>
        )}

        {/* conversation */}
        <main className="flex-1 min-w-0 min-h-0 flex flex-col lg:p-3">
          <section className="flex-1 min-h-0 flex flex-col card rounded-none lg:rounded-2xl border-0 lg:border overflow-hidden">
            {switching && (
              <div className="px-4 py-2 text-sm text-navy-800 bg-sand-100 border-b border-sand-200 flex items-center gap-2">
                <span className="dot">●</span> {t.switching.replace('{name}', personas.find((p) => p.user_id === switching) ? name(personas.find((p) => p.user_id === switching)!) : '')}
              </div>
            )}
            {!user && !switching && (
              <div className="flex-1 grid place-items-center p-8 text-center">
                <div className="max-w-md space-y-3">
                  <div className="text-4xl">🦅</div>
                  <h1 className="text-lg font-semibold text-navy-900">{t.welcomeTitle}</h1>
                  <p className="text-gray-600 text-sm">{t.welcomeBody}</p>
                  <button className="btn-primary lg:hidden" onClick={() => setDrawer(true)}>{t.customers}</button>
                  {error && <div className="text-red-600 text-sm">{error}</div>}
                </div>
              </div>
            )}
            {user && <Chat key={`${user.session_id}-${lang}`} t={t} lang={lang} openKey={openKey} onWrite={onWrite} prefill={prefill} onPrefillUsed={() => setPrefill(undefined)} />}
          </section>
        </main>

        {/* what the agent sees (large screens, unless focus) */}
        {user && !focus && (
          <aside className="hidden lg:flex flex-col w-[26rem] shrink-0 min-h-0 overflow-y-auto p-3 ps-0 space-y-3">
            {persona && (
              <div className="card p-4 space-y-1">
                <div className="text-[11px] uppercase tracking-wide text-gray-500">{t.thisCustomer}</div>
                <div className="font-semibold text-navy-900 text-lg">{name(persona)}</div>
                <div className="text-sm text-gray-600">{note(persona)}</div>
                <div className="text-[11px] text-gray-400 font-mono" dir="ltr">{persona.customer_id}</div>
              </div>
            )}
            <div className="flex gap-1 text-sm">
              {(['trips', 'search'] as const).map((k) => <button key={k} onClick={() => setPanel(k)} className={`chip py-1 px-3 ${panel === k ? 'bg-navy-800 text-white' : 'bg-white text-navy-800 border border-sand-200'}`}>{t[k]}</button>)}
            </div>
            {panel === 'trips' ? <Trips key={tripsKey} t={t} lang={lang} onAsk={setPrefill} /> : <Search t={t} lang={lang} onAsk={setPrefill} />}
          </aside>
        )}
      </div>
      <footer className="px-4 py-1 text-[10px] text-gray-500 bg-sand-100 border-t border-sand-200 text-center shrink-0">{t.footer}</footer>
    </div>
  )
}
