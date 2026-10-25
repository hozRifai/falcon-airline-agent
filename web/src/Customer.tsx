import { useEffect, useState } from 'react'
import { api } from './api'
import type { Dict, Lang } from './i18n'

// The side panel of the demo console: what the agent sees about this customer (trips with live status) and a flight search.

const STATUS_COLOR: Record<string, string> = { on_time: 'bg-teal-500/10 text-teal-600', scheduled: 'bg-sand-100 text-navy-800', boarding: 'bg-teal-500/10 text-teal-600', delayed: 'bg-amber-100 text-amber-800', cancelled: 'bg-red-100 text-red-700', departed: 'bg-gray-100 text-gray-600', landed: 'bg-gray-100 text-gray-600', unknown: 'bg-gray-100 text-gray-600' }

export function Trips({ t, lang, onAsk }: { t: Dict; lang: Lang; onAsk: (q: string) => void }) {
  const [data, setData] = useState<Record<string, any>[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { api.bookings().then((r) => setData(r.bookings)).catch((e) => setErr(String(e.message))) }, [])
  if (err) return <div className="text-red-600">{err}</div>
  if (!data) return <div className="text-gray-500">…</div>
  if (data.length === 0) return <div className="text-gray-500">—</div>
  return (
    <div className="space-y-4">
      {data.map((b) => (
        <div key={b.pnr} className="card pass-notch overflow-hidden">
          <div className="bg-navy-900 text-white px-4 py-2 flex items-center justify-between">
            <span className="font-mono tracking-widest">{b.pnr}</span>
            <span className={`chip ${b.status === 'confirmed' ? 'bg-teal-500 text-white' : 'bg-white/20 text-white'}`}>{t.bookingStatus[b.status] || b.status}</span>
          </div>
          <div className="p-4 space-y-3">
            {b.segments.map((s: Record<string, any>) => (
              <div key={s.seq} className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-2xl font-semibold text-navy-900">{s.origin} <span className="text-gray-400">→</span> {s.destination}</div>
                  <div className="text-sm text-gray-600" dir="ltr">{s.flight_number} · {s.departure.replace(' (Gulf time)', '')}</div>
                </div>
                <span className={`chip ${STATUS_COLOR[s.live_status] || STATUS_COLOR.unknown}`}>{t.status[s.live_status] || s.live_status}{s.delay_minutes ? ` +${s.delay_minutes}m` : ''}</span>
              </div>
            ))}
            <div className="grid grid-cols-3 gap-2 text-sm text-gray-700 border-t border-dashed border-sand-200 pt-3">
              <div><div className="text-xs text-gray-400">{t.fare}</div>{t[String(b.fare_family).toLowerCase() as 'special'] || b.fare_name}</div>
              <div><div className="text-xs text-gray-400">{t.passengers}</div>{b.passengers.length}</div>
              <div><div className="text-xs text-gray-400">{t.paid}</div>{b.total_paid} {b.currency}</div>
            </div>
            {b.cancellation_deadline && b.status === 'confirmed' && <div className="text-xs text-gray-500" dir="ltr">{t.deadline}: {b.cancellation_deadline.replace(' (Gulf time)', '')}</div>}
            <button className="btn-ghost text-sm w-full justify-start px-0" onClick={() => onAsk(lang === 'ar' ? `أريد الاستفسار عن حجزي ${b.pnr}` : `I have a question about my booking ${b.pnr}`)}>{t.askTrip} →</button>
          </div>
        </div>
      ))}
    </div>
  )
}

const DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
const DAYS_AR: Record<string, string> = { monday: 'الاثنين', tuesday: 'الثلاثاء', wednesday: 'الأربعاء', thursday: 'الخميس', friday: 'الجمعة', saturday: 'السبت', sunday: 'الأحد' }

export function Search({ t, lang, onAsk }: { t: Dict; lang: Lang; onAsk: (q: string) => void }) {
  const [form, setForm] = useState({ origin: 'Dubai', destination: 'Cairo', date_from: '2026-11-03', date_to: '2026-11-05', passengers: '1' })
  const [skip, setSkip] = useState<string[]>([])
  const [res, setRes] = useState<Record<string, any> | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const go = async (e?: React.FormEvent) => {
    e?.preventDefault(); setBusy(true); setErr(null)
    try { setRes(await api.search({ ...form, exclude_weekdays: skip.join(',') })) } catch (ex) { setErr(String((ex as Error).message)) } finally { setBusy(false) }
  }
  const fareName = (code: string) => t[code.toLowerCase() as 'special']
  const fareNote: Record<string, string> = { SPECIAL: `20 kg · 1 ${t.changes} · ${t.refund}: ${t.creditOnly}`, SAVER: `30 kg · 2 ${t.changes} · ${t.refund}: ${t.cashRefund}`, FLEX: `35 kg · 3 ${t.changes} · ${t.refund}: ${t.cashRefund}` }
  return (
    <div className="space-y-4">
      <form className="card p-4 grid grid-cols-2 gap-3" onSubmit={go}>
        <label className="text-sm"><span className="text-gray-500">{t.from}</span><input className="input" value={form.origin} onChange={(e) => setForm({ ...form, origin: e.target.value })} /></label>
        <label className="text-sm"><span className="text-gray-500">{t.to}</span><input className="input" value={form.destination} onChange={(e) => setForm({ ...form, destination: e.target.value })} /></label>
        <label className="text-sm"><span className="text-gray-500">{t.dateFrom}</span><input className="input" type="date" value={form.date_from} onChange={(e) => setForm({ ...form, date_from: e.target.value })} /></label>
        <label className="text-sm"><span className="text-gray-500">{t.dateTo}</span><input className="input" type="date" value={form.date_to} onChange={(e) => setForm({ ...form, date_to: e.target.value })} /></label>
        <label className="text-sm"><span className="text-gray-500">{t.passengers}</span><input className="input" type="number" min={1} max={9} value={form.passengers} onChange={(e) => setForm({ ...form, passengers: e.target.value })} /></label>
        <div className="text-sm col-span-2"><span className="text-gray-500">{t.skip}</span>
          <div className="flex flex-wrap gap-1 mt-1">{DAYS.map((d) => <button type="button" key={d} onClick={() => setSkip(skip.includes(d) ? skip.filter((x) => x !== d) : [...skip, d])} className={`chip ${skip.includes(d) ? 'bg-navy-800 text-white' : 'bg-sand-100 text-navy-800'}`}>{lang === 'ar' ? DAYS_AR[d] : d.slice(0, 3)}</button>)}</div>
        </div>
        <button className="btn-primary col-span-2" disabled={busy}>{t.findFlights}</button>
      </form>
      {err && <div className="text-red-600 text-sm">{err}</div>}
      {res && res.alternative_airport && (
        <div className="rounded-2xl bg-teal-500/10 border border-teal-500/30 p-3 text-sm">
          <b>{t.cheaperFrom} {lang === 'ar' ? res.alternative_airport.city_ar : res.alternative_airport.city} ({res.alternative_airport.airport})</b>: {res.alternative_airport.best_option.flight_number} {res.alternative_airport.best_option.date} · {res.alternative_airport.best_option.price_per_passenger_aed.SPECIAL} AED · {t.saving} {res.alternative_airport.saving_per_passenger_aed_vs_best_main_option} AED {t.perPassenger} · {res.alternative_airport.drive_minutes} min {t.drive}
        </div>
      )}
      {res && res.options_cheapest_first.length === 0 && <div className="text-gray-500">{t.noResults}</div>}
      {res && res.options_cheapest_first.map((o: Record<string, any>) => (
        <div key={o.flight_instance_id} className="card p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div><div className="text-lg font-semibold text-navy-900">{o.origin} → {o.destination} <span className="text-sm font-normal text-gray-500">{o.flight_number}</span></div>
              <div className="text-sm text-gray-600" dir="ltr">{o.weekday} {o.departure_local} → {o.arrival_local.slice(11)} · {Math.floor(o.duration_minutes / 60)}h{o.duration_minutes % 60 ? ` ${o.duration_minutes % 60}m` : ''}</div></div>
            <div className="text-xs text-gray-500">{o.seats_available} {t.seatsLeft}</div>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {['SPECIAL', 'SAVER', 'FLEX'].map((f) => (
              <button key={f} className="rounded-xl border border-sand-200 hover:border-teal-500 p-2 text-start" title={t.askAboutFare}
                onClick={() => onAsk(lang === 'ar' ? `ما الذي تشمله درجة ${fareName(f)} على الرحلة ${o.flight_number} بتاريخ ${o.date}؟` : `What does the ${fareName(f)} fare include on ${o.flight_number} on ${o.date}?`)}>
                <div className="text-xs text-gray-500">{fareName(f)}</div>
                <div className="font-semibold text-navy-900">{o.price_per_passenger_aed[f]} <span className="text-xs font-normal">AED</span></div>
                <div className="text-[11px] text-gray-500 leading-tight">{fareNote[f]}</div>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
