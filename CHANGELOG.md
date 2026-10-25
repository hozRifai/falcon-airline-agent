# Changelog

What changed in the code, newest first, grouped by day. Each entry says what was wrong or what was added and how it behaves now;
the tests that cover a fix are named with it.

## 2026-10-22

### Changed (the frozen clock is Fri 23 Oct 2026)
- Every seeded date moved three days (`FALCON_NOW`, the schedule, the bookings, the disruptions, the promo window, the tasks, the live
  tests). The weekly price pattern now follows the day's position in the schedule rather than the calendar weekday, so every price and
  pinned amount is unchanged. `scripts/reset.sh` needed after pulling.

### Added (five use cases)
- The app lists five use cases, each on a hard day (a cancellation, a 5½ h delay, a 2½ h delay
  inside the deadline, a pregnancy at 37 weeks, a cancelled leg with a separate connection to Colombo, a new route); the twelve earlier personas stay in the
  seed, hidden (`agent.users.shown`). The sidebar says Use cases and no longer shows a language preference.
- Cancelling some of a booking's passengers: `quote_cancellation(passenger_ids)` and `cancel_booking` keep the booking for the others,
  mark the named passengers cancelled (`reservations.passengers.cancelled_at`), release their seats and refund their fares minus one fee
  each. Tests: `test_cancelling_some_of_the_passengers_refunds_their_fares_and_keeps_the_booking`; tasks: `eval/tasks/use_cases.yaml`.

### Changed
- Voice: the opening progress phrase ("Give me a minute...") is spoken on the first turn of a call only; later slow turns get just
  the 10-second "still checking" phrase (`voice/call.py`, test added).
- A change or a new booking carries the bag tools and the baggage prompt section: a change and a bag in one
  message no longer end in "I can't add baggage". Test: `test_the_topic_picks_the_prompt_section_and_the_tool_menu`.
- Tried and reverted: clause ids stripped from the customer's reply with the clauses shown as a source line under it;
  reverted the same day on the demo's behaviour, the reply cites the clause inline as before.

## 2026-10-21

### Removed (the code carries only what the demo runs)
- **Entra ID auth mode** and the `azure-identity` dependency; every token is HS256 with one shared secret, and a service token's
  audience is the server's name (`reservations`, `flightops`, `policy`). The direct **Anthropic provider**: the model is OpenRouter or
  the scripted stand-in. Four model settings that named the same model collapse into `LLM_MODEL`.
- The optional **model reviewer** (`REVIEW_LLM`): the reviewer is the deterministic checks plus the citation check.
- The **role** dimension (one persona): no `role` on the user context, in the prompt, in the graph state or in the tool
  profile; the supervisor `Override` model is two flags on the exception quote (`ignore_deadline`, `waive_fee`).
- `/api/auth/demo-login` and the access code, the per-IP rate limit, the per-session turn cap, `/api/me`, `/api/fare-rules`,
  `/api/admin/reset`: the app, the harness and the scripts all sign in through `/api/demo/switch`; the daily token budget stays.
- The legacy flight API's failure injection and rate limit, and its client's cache; the unused `ScriptedChatModel`; `eval/stats.py`
  (the scorecard's percentile is the one); the `--draft-labels` scaffolding of the generation evaluation; the voice STT comparison
  tooling (`wer.py`, the shadow STT, the audio probe, `scripts/voice_stt_compare.py`, `scripts/voice_latency.py`).

### Added
- **`/api/ready`**: a readiness check for a load balancer or the platform's probe. It checks the database, the conversation store and
  each tool server (with a real service token, so a broken credential shows) and answers 200, or 503 naming what is down
  (`{"ready": false, "checks": {"flightops": "down (...)", ...}}`). `/api/health` stays the liveness check: it said "ok" with a system
  down, which a monitor cannot use. `tests/test_resilience.py`.

### Fixed
- **The side panel's "ask the assistant about this trip / fare" buttons did nothing.** They now put the question in the chat box.
- **A database restart or failover stopped the agent until it was restarted.** The conversation history went through one long-lived
  connection; once the database dropped it, every chat failed. It now goes through a connection pool, every pool tests a connection
  before using it, and every query has a time limit (15 s by default, `DB_STATEMENT_TIMEOUT_MS`), so a slow query becomes an error
  instead of a hung conversation. `tests/test_resilience.py`.
- **One unreachable system took the whole agent down.** If any one of the three tool servers (bookings, flight operations, policy)
  was down or refused the service token, every conversation got "I cannot reach the booking systems". Each server is now discovered
  on its own: the agent works with the others, is told which system is down, and says so instead of guessing (for example: "I can't
  check live flight status at the moment", with an offer to retry or hand over). A degraded list is retried after 30 s.
  `tests/test_resilience.py`.
- **Two confirmations of the same quote executed twice.** A double click, a network retry or a spoken "yes" together with the
  button could confirm one quote twice at the same moment: the booking was cancelled once, but the refund and fee were written
  twice. Every write (cancel, change, new booking, extra bag) now claims its quote as the first step of its database transaction;
  the second request waits, finds the quote used and returns the first result. `tests/test_concurrency_live.py`.
- **A question without a question mark was not recognised while the classifier was down.** The pattern that keeps policy questions
  on the document when the input classifier is unavailable ended in a stray control character instead of a word boundary, so
  only messages with "?" or "؟" counted. `tests/test_guardrails.py::test_a_question_without_a_question_mark_is_a_question`.
- **The citation check could hold a reply for minutes.** Its 25-second deadline was configured but never applied; it now uses that
  deadline and a single attempt (a check that does not answer in time passes the draft, as designed).
  `tests/test_review_gate.py::test_the_citation_check_has_its_own_deadline`.

### Changed
- One definition each for helpers that existed in several copies: the Arabic-digit table (the reviewer's amount check now also reads
  Persian digits, like the personal-data filter and the scorer), message text, the set of write tools, the topic list, month names,
  the evaluation report helpers. The personal-data filter's `find()` and `redact()` share one matching routine.
- A linter configuration (`[tool.ruff]` in `pyproject.toml`) and the codebase passes it: exceptions narrowed to the errors that can
  actually occur, deliberate catch-alls labelled with their reason, ambiguous variable names renamed.
- Comments and docstrings no longer refer to internal design-note numbers; the explanations stay.
- **Exception cases for a personal reason ask for the document by email.** When a customer asks for an exception because of a
  medical emergency, a bereavement or an accident, the agent logs the case once they confirm, and the customer-care team emails them
  for the document it needs (a death certificate, or a doctor's note or official letter) with the case number in the subject. A reason
  the airline can already see (a delayed flight) asks for nothing. `tests/test_reservations_live.py`.

### Removed
- Code nothing called (two clock helpers, a synchronous connection helper, an unused flight lookup, a staff-only request listing,
  an evaluation helper), the `sqlglot` dependency of a removed SQL agent, and an unread configuration setting.
- Document attachments: the upload endpoint (`POST /api/attachments`), the two attachment tools, the attachments table (dropped on
  the next `scripts/up.sh`) and the `python-multipart` dependency.

## 2026-10-20

### Added
- **Voice view in the app**: a Chat | Voice switch on the conversation. The voice view is a call screen: one microphone button, live
  meters for the customer and the agent, whose-turn status, a call timer, a short caption, the confirmation card in compact form.
  The customer's words appear as they are heard and the reply appears word by word as it is spoken; every spoken turn is also in
  the chat with its badges and trace id.
- **Call button** (`web/src/Call.ts`): a call uses the app's session, conversation and language, so it continues the conversation on
  screen; typing is off during a call; a confirmation card already on screen is answered by voice.
- **Timed progress phrases on a call**: one fixed phrase if no reply came within 1.2 s, one more after 10 s, never more, none for a
  fast reply.
- Speech recognition hints (the customer's booking codes and common airline words), the recogniser's confidence logged on every
  transcript, an optional second recogniser on the same audio for comparison (`VOICE_STT`, `VOICE_STT_SHADOW`), and the per-turn
  latency of every call in the log.
- Measurement tools: `python -m falcon.eval.voice_tasks` (five Arabic tasks through the voice path, graded like the chat tasks),
  `scripts/voice_latency.py`, `scripts/voice_stt_compare.py` with `eval/voice/stt_script.yaml` (word and character error rate).

### Fixed
- **Follow-ups were judged without their conversation.** The input classifier saw the latest message alone, so a fragment or "and
  the other one?" could be refused as off-topic or routed to the wrong tools; it now reads the assistant's previous reply as context.
- **A bare "yes" to a two-option question picked the second option.** After the agent offered two refunds (travel credit or the
  card), "yes" raised the confirmation card for the last one quoted; with two different offers on the table no shortcut chooses, and
  the agent asks which.
- **The input classifier could return no label** on muddled speech: the model's hidden reasoning used up its token budget before the
  answer. The budget now leaves room for both.
- **"Could not reach my language model" after citing a parent clause.** A reply citing clause 7.4 when only 7.4.x had been retrieved
  crashed the reviewer; the parent is now checked against its retrieved children.
- **A rebuilt app did not show without a hard refresh**: the page is sent with `Cache-Control: no-cache`.

### Removed
- The attachment button in the app (the upload endpoint stays for the evaluation's medical-exception task).

## 2026-10-19

### Added
- **Voice channel** (`src/falcon/services/voice`, `scripts/voice.sh`, `scripts/up.sh --voice`): a Pipecat pipeline per call
  (WebRTC, Silero voice detection, Deepgram Nova-3 in Arabic or English, ElevenLabs Flash v2.5) in front of the same runtime. It runs
  no model of its own: each turn goes to `/api/chat` with the customer's session, so the guard, tools, reviewer, confirmation gate
  and traces are the chat's. Only the reviewed reply is spoken; the confirmation card is read aloud and answered with a bare
  "yes" / «نعم» or "no"; codes, clause ids and amounts are normalised for speech in both languages.
- The bare yes / bare no vocabulary in one module (`falcon/core/confirmation.py`), shared by the chat and the voice channel.

### Fixed
- **One spoken request answered several times**: the customer's turn ended at every short pause; it now ends after 1.5 s of silence
  (`VOICE_TURN_PAUSE`), and whatever is said while a turn is running goes to the agent as one message.

First public snapshot: 2026-10-19.
