# Sister Vuo

Sister Vuo is an AI nursing patient simulator built with Python 3.11+ and FastAPI. A student nurse takes a history from a fictional patient played by a Llama model, manages the patient's deterioration with clinical actions, then receives a structured debrief from a separate evaluator model.

This is an educational simulation. It is not a clinical decision-support tool, and it must not contain real patient data.

## Features

- Three fictional scenarios:
  1. Abdominal pain in a woman of childbearing age
  2. Paediatric dehydration, with the mother speaking for the child
  3. Chest pain in an older adult
- In-character patient replies in one to three sentences
- Hidden diagnosis, risk factors and ground truth withheld from the patient prompt
- Server-side history disclosure: a topic is revealed only when the student asks about it
- Beginner, Intermediate and Advanced learning modes
- A patient monitor with vitals that deteriorate as simulated and real time pass
- Clinical actions including observations, ECG, oxygen, aspirin, IV access and escalation
- Supervisor hints in Beginner (automatic) and Intermediate (on request) modes, with a score penalty
- Optional local guideline mode: links-only (0 steps) or full (5-8 steps) per scenario, with question-driven supervisor hints and protocol scoring
- Automatic evaluation after `END` or 12 student messages
- Six rubric domains: red flag recognition, history completeness, prioritisation, safety, communication and empathy, and time to escalation
- A debrief with timeline, missed findings, critical errors, model answer, ideal approach, reflection questions, guideline card and protocol-step status
- Browser chat UI with local saved debriefs and a responsive monitor layout
- WhatsApp Cloud API webhook with per-phone sessions and text clinical commands
- Local JSONL transcript logs
- API and webhook tests with no external model calls

## How it works

The application uses two LLM roles sharing one OpenAI-compatible Llama endpoint plus a deterministic clinical engine:

- **Patient model**: one chat completion per student message at temperature `0.7`. It receives only the patient details and the history topics the student has unlocked. Hidden diagnosis, risk factors and ground truth are deliberately excluded.
- **Clinical engine**: advances a simulated clock, deteriorates or improves vitals, records actions and determines escalation timing. The monitor also responds to real time while the student is thinking.
- **Evaluator model**: one completion per session at temperature `0.2`, after `END` or after the 12th student message. It receives the ground truth, transcript, monitor, timeline and actions, then returns a single JSON object.

The server computes red flag recognition, history completeness, safety and time to escalation itself, and takes prioritisation and communication/empathy from the evaluator. This keeps objective scores reproducible. The overall score is the mean of the six domains minus any hint penalty. If the evaluator returns invalid JSON, the app retries once with a stricter instruction. Sessions are stored in memory and are lost when the process restarts.

Guideline content is loaded from a local JSON file into a repository at startup. A missing or invalid file disables the feature with one clear log message while the rest of the app keeps working. Each scenario can be links-only (0 steps) or full (5-8 steps). Full mode augments the evaluator prompt with the supplied step IDs and text, normalises the returned `protocol_steps` itself, and applies a dose guard so the debrief cannot introduce a dose that was not supplied. Supervisor questions are matched to stored steps with simple keyword overlap; no guideline text is sent to the patient model or added to the patient transcript.

## Project layout

```text
app/
  config.py          Environment and path settings
  models.py          Pydantic request, response, and evaluation models
  clinical.py        Vitals, deterioration, actions, difficulty profiles
  guidelines.py      Guideline schema, validation, modes, hint matching
  scoring.py         Deterministic assessment and evaluator merge
  llm.py             OpenAI-compatible client, patient prompt, evaluator parsing
  sessions.py        Scenario loading, sessions, actions, hints, debriefs, logs
  main.py            FastAPI app and REST routes
  whatsapp.py        Meta webhook, command handling, Graph API sender
  static/index.html  Browser chat UI
prompts/
  patient.txt        Patient role-play system prompt
  evaluator.txt      Evaluator rubric and JSON contract
scenarios/
  abdominal_pain.json
  paediatric_dehydration.json
  chest_pain.json
guidelines.example.json  Template for the local, gitignored guidelines.json
tests/
  test_api.py
  test_guidelines.py
  test_scoring.py
  test_llm.py
  test_whatsapp.py
```

## Setup

Requirements: Python 3.11 or newer, and access to an OpenAI-compatible Llama endpoint.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`:

```dotenv
LLM_BASE_URL=https://your-llama-endpoint/v1
LLM_API_KEY=replace-me
LLM_MODEL=your-llama-model
```

Set `APP_ENV=production` on a public host. In production the app raises a clear `RuntimeError` at startup when `APP_API_KEY` is empty, so it cannot run with an open API by accident. `APP_ENV` defaults to `development`. Other operational settings are documented in `.env.example`: `RATE_LIMIT_START_PER_MINUTE`, `RATE_LIMIT_TURN_PER_MINUTE`, `RATE_LIMIT_PROBE_PER_MINUTE`, `RATE_LIMIT_WHATSAPP_PER_MINUTE` and `TRANSCRIPT_RETENTION_DAYS`.

`LLM_BASE_URL` should be the API root that ends in `/v1` or similar. The app calls `POST {LLM_BASE_URL}/chat/completions`. If the provider supports OpenAI JSON mode, set `LLM_JSON_MODE=true`; otherwise the evaluator prompt itself requests JSON and the parser recovers fenced or surrounding text.

Guidelines are optional. Copy `guidelines.example.json` to `guidelines.json` (gitignored), fill in document metadata, and write the steps in your own short words. Do not reproduce guideline text without permission, and do not commit `guidelines.json`. Point `GUIDELINES_PATH` at another local file if you prefer. With no file, the app remains fully usable and falls back to scenario hints.

Start the server:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

The app starts even without LLM configuration so the UI and `/health` can be inspected. Session messages will return HTTP 503 until the LLM variables are set.

## REST API

### Start a session

```bash
curl -X POST http://127.0.0.1:8000/session/start \
  -H "Content-Type: application/json" \
  -d "{\"scenario_id\":\"abdominal_pain\",\"student_name\":\"Alex\",\"difficulty\":\"Intermediate\"}"
```

Scenario IDs are `abdominal_pain`, `paediatric_dehydration`, and `chest_pain`. `difficulty` is `Beginner`, `Intermediate` or `Advanced`; when omitted, the scenario's own level is used.

The response includes `session_id`, a random `session_token`, a safe scenario summary, the patient's opening line, the 12-turn limit, the current `monitor`, available `actions`, `difficulty`, `hints_enabled`, and a `guideline` object (`name`, `edition_year`, `section_text`, optional `url`, `mode`, and `further_reading`) or `null` when guidelines are off.

Every `/session/{id}/...` route also needs the session token in the `X-Session-Token` header. A missing or wrong token returns HTTP 404 with the same `Session not found.` body as an unknown id, so session ids cannot be used on their own. Keep the token in memory; do not put it in a URL, log or browser storage. WhatsApp sessions stay bound to the sender's phone number and do not use this token.

### Send a student message

```bash
curl -X POST http://127.0.0.1:8000/session/SESSION_ID/message \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: SESSION_TOKEN" \
  -d "{\"message\":\"When did the pain start?\"}"
```

The response contains the patient reply, turn counts, status, the updated `monitor`, an automatic `hint` when one applies, and `evaluation` if the 12-turn limit was reached. If evaluation fails at the limit, the response includes `evaluation_error`; call the end endpoint to retry.

### Perform a clinical action

```bash
curl -X POST http://127.0.0.1:8000/session/SESSION_ID/action \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: SESSION_TOKEN" \
  -d "{\"action_id\":\"vitals\"}"
```

Action IDs depend on the scenario, for example `vitals`, `ecg`, `oxygen`, `aspirin`, `iv_access`, `pregnancy_test`, `call_ambulance` and `escalate_doctor`. The response includes the result, any safety warning, the updated monitor, actions taken and remaining turns. Repeating a completed non-repeatable action returns HTTP 409; an unknown or unavailable action returns HTTP 404 or 409.

### Ask for a hint

```bash
curl -X POST http://127.0.0.1:8000/session/SESSION_ID/hint \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: SESSION_TOKEN" \
  -d "{\"question\":\"Which observations should I record?\"}"
```

The body is optional for backward compatibility. Hints are disabled in Advanced mode (HTTP 409) and each used hint adds a `0.3` penalty, capped at `1.2`. In full guideline mode the server returns at most three stored steps per session, matched by simple keyword overlap; a fourth request returns HTTP 409. A links-only scenario returns HTTP 404 for hints. If no stored step matches, the response says `Check the guideline: <section_text>.` A step whose text contains one of the scenario's `forbidden_patient_terms` is never returned as a hint (it is still used for scoring); if that leaves no match, the fallback line is returned. The question and matched step IDs are logged, but hint text is never added to the patient transcript or patient prompt. Beginner sessions also receive automatic hints at turns 3, 6 and 9.

### Read the monitor

```bash
curl http://127.0.0.1:8000/session/SESSION_ID/monitor \
  -H "X-Session-Token: SESSION_TOKEN"
```

### End and evaluate

```bash
curl -X POST http://127.0.0.1:8000/session/SESSION_ID/end \
  -H "X-Session-Token: SESSION_TOKEN"
```

Calling `/end` again returns the stored evaluation without another model call.

Other routes:

- `GET /scenarios` - available scenario summaries
- `GET /session/{id}` - status, transcript, timeline, monitor and debrief snapshot
- `GET /health` - configuration, active-session status, and guideline mode (`off`, `links_only` or `full`)
- `GET /health?probe=true` - also makes one tiny model call (`status` becomes `degraded` on failure)
- `GET /health/llm` - dedicated model-only deep check (requires `APP_API_KEY` when set)

All scenario and session routes require the `X-API-Key` header when `APP_API_KEY` is set. The health probes also require it when probing the model. Every `/session/{id}/...` route additionally requires that session's `X-Session-Token`; a missing or wrong token returns HTTP 404.

### Rate limits

A small in-memory sliding-window limiter tracks each client IP (`request.client.host`) and returns HTTP 429 with a `Retry-After` header when a limit is exceeded. The limits are `RATE_LIMIT_START_PER_MINUTE` (default 10), `RATE_LIMIT_TURN_PER_MINUTE` (default 60, shared by message, action and hint), `RATE_LIMIT_PROBE_PER_MINUTE` (default 6 for `/health?probe=true` and `/health/llm`) and `RATE_LIMIT_WHATSAPP_PER_MINUTE` (default 30 per sender phone; over the limit the sender gets one short "please slow down" reply and the message is not processed). Old entries and keys are dropped, so memory stays bounded.

When the app runs behind Render's proxy, start Uvicorn with `--proxy-headers` so `request.client.host` is the real client and not the proxy. `render.yaml` already includes it.

## If the model cannot be reached

The server logs the real failure cause, not only a friendly message: request URL, model, attempt, HTTP status, duration, provider error message, and a truncated response body. The API key is never logged.

The student sees a specific message:

| Failure | What the app says |
| --- | --- |
| Missing `LLM_BASE_URL`, `LLM_API_KEY`, or `LLM_MODEL` | Names the missing variable and says to restart after fixing `.env` |
| Timeout | Names the provider host and the timeout value |
| DNS, TLS, proxy, or connection failure | Says the endpoint was unreachable and points at `LLM_BASE_URL`, proxy, VPN, or firewall |
| HTTP 401 or 403 | Says the provider rejected the API key and points at `LLM_API_KEY` |
| HTTP 402 | Says the provider account has no credit or quota |
| HTTP 404 | Points at `LLM_BASE_URL` (usually ends in `/v1`) and `LLM_MODEL` |
| HTTP 429 or 5xx | Retries once, then reports rate limiting or a provider server error |
| HTTP 400 or 422 | Points at `LLM_MODEL` and `LLM_JSON_MODE`, and includes the provider message |
| Redirect or non-JSON body | Says the base URL is probably wrong or points at a landing page |

Every request has a timeout (`LLM_TIMEOUT_SECONDS`, default 60) and one automatic retry for timeouts, connection failures, HTTP 429, and HTTP 5xx. Click **Test model connection** in the UI, or call `/health?probe=true`, to verify the full path without starting a session.

## Evaluation output

The API exposes a merged evaluation and debrief:

```json
{
  "overall_score": 3.4,
  "domain_scores": {
    "red_flag_recognition": 4,
    "history_completeness": 3,
    "prioritisation": 4,
    "safety": 4,
    "communication_empathy": 3,
    "time_to_escalation": 3
  },
  "strengths": ["..."],
  "missed_red_flags": ["..."],
  "unsafe_actions": [],
  "critical_errors": [],
  "recommended_learning": ["..."],
  "feedback_to_student": "...",
  "hint_penalty": 0.3,
  "protocol_steps": [
    {"id": "step-1", "status": "met", "evidence_turn": 2}
  ],
  "debrief": {
    "what_was_missed": ["..."],
    "ideal_approach": ["..."],
    "model_answer": "...",
    "reflection_questions": ["..."],
    "guideline_references": ["..."],
    "timeline": ["..."],
    "time_to_escalation_minutes": 8,
    "escalation_target_minutes": 10,
    "actions_taken": ["vitals", "ecg", "call_ambulance"]
  }
}
```

`overall_score` is recomputed by the server as the mean of the six whole-number domains minus the hint penalty, rounded to one decimal place. Red flag recognition, history completeness, safety and time to escalation come from the deterministic assessor; prioritisation and communication/empathy come from the evaluator. This prevents inconsistent arithmetic and keeps objective scores reproducible.

In full guideline mode, `protocol_steps` is `null` only when the feature is off. The server keeps configured order, drops unknown or duplicate IDs, treats missing IDs as `missed`, and appends a deterministic count sentence to `feedback_to_student`. The dose guard removes any sentence in `strengths`, `recommended_learning` or `feedback_to_student` that names a dose not present in the supplied steps.

## WhatsApp Cloud API

1. Create a Meta developer app with the WhatsApp product and a test phone number.
2. Expose the local server over public HTTPS, for example with ngrok:

   ```powershell
   ngrok http 8000
   ```

3. Set the callback URL to `https://YOUR-PUBLIC-HOST/webhook/whatsapp` and the verify token to `WHATSAPP_VERIFY_TOKEN`.
4. Subscribe the app to the `messages` webhook field.
5. Fill in the WhatsApp variables in `.env`:

   ```dotenv
   WHATSAPP_VERIFY_TOKEN=your-verify-token
   WHATSAPP_ACCESS_TOKEN=your-meta-access-token
   WHATSAPP_PHONE_NUMBER_ID=your-phone-number-id
   WHATSAPP_APP_SECRET=your-meta-app-secret
   WHATSAPP_API_VERSION=v23.0
   ```

`WHATSAPP_APP_SECRET` is required. The webhook validates `X-Hub-Signature-256` on every POST, and stays disabled (HTTP 503) until all four WhatsApp variables are set, so forged payloads cannot make the app message arbitrary numbers.

A tunnel such as `ngrok http 8000` exposes the whole app, not only the webhook. Set `APP_API_KEY` before starting one (see below).

### WhatsApp commands

- `START` - begins scenario 1 (abdominal pain)
- `START 1`, `START 2`, `START 3` - begins a specific scenario
- `HELP` or `MENU` - shows the scenario menu
- `VITALS`, `ECG`, `OXYGEN`, `ASPIRIN`, `IV`, `AMBULANCE`, `DOCTOR`, `PREGNANCY TEST` - perform clinical actions
- `HINT` or `HINT <question>` - requests a supervisor hint when the mode allows it; full guideline mode matches the question to stored steps
- `STATUS` - returns the current monitor and clinical time
- `END` - evaluates the active session
- Any other text - sent to the active patient; if there is no active session, the menu is returned

The session ends automatically after 12 student messages and the feedback is sent to the same phone number. Starting a new session while one is active is blocked; reply `END` first.

WhatsApp free-form replies are appropriate inside the 24-hour customer service window. Production use also requires Meta app review, business verification, messaging limits, and compliance with WhatsApp policy.

## Logs and privacy

- Each session writes `logs/transcripts/{session_id}.jsonl` with start, student messages, patient messages, history disclosure, clinical actions, hints, escalation, evaluation, and evaluation failures. Guideline hints log the student's question and matched step IDs, not the stored hint text.
- Transcripts contain fictional patient dialogue and student messages. Do not enter real patient information.
- Transcript files older than `TRANSCRIPT_RETENTION_DAYS` (default 30) are deleted at startup, whenever a session starts, on an hourly background sweep, and on shutdown. Set it to `0` to keep them until the operator removes them. The operator can also delete a session's transcript on request.
- Completed debriefs are stored in the browser's `localStorage` so the student can review them later. They are not sent anywhere else; use **Clear saved sessions** to remove them.
- WhatsApp phone numbers are kept only in memory for routing. Logs contain a masked value such as `***1234`, never the full number.
- `.env`, `logs/` and `guidelines.json` are gitignored. Only `guidelines.example.json` belongs in version control.
- API keys are never logged or returned to clients.

Access control:

- With `APP_API_KEY` set, `/scenarios` and every `/session/...` endpoint require an `X-API-Key` header. The chat UI asks for the key once and keeps it in the browser. Without it the API is open, which is fine on localhost only.
- Each web session gets a random `session_token`; `/session/{id}/...` routes require it in `X-Session-Token`. The chat UI keeps it in memory only (never `localStorage`) and sends it with every session request.
- Sessions idle for `SESSION_TTL_MINUTES` (default 120) are removed from memory, and at most `MAX_ACTIVE_SESSIONS` (default 200) can be active at once; beyond that, starting returns HTTP 503.
- The evaluator receives the transcript as a JSON array, so a student cannot forge patient turns by typing line breaks.

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check app tests
python -m compileall -q app
```

Tests substitute fake LLM and WhatsApp clients, so they do not need model credentials or network access. They cover the full session flow, automatic evaluation, evaluator failure and retry, hidden-history disclosure, clinical actions, monitor updates, difficulty and hints, guideline modes and limits, protocol normalisation, dose guarding, prompt/history isolation, debriefs, transcript logging, scenario privacy, webhook verification, signatures, deduplication, and WhatsApp commands.

## Limitations

- Sessions and phone mappings are in memory; a restart clears them. Run a single Uvicorn worker, because multiple workers would have separate session stores.
- There is no database, per-user accounts, or streaming. Rate limits are in memory and per process; the single `APP_API_KEY` is shared by everyone who uses the UI.
- The browser history is per-browser and per-origin; clearing site data removes it. It is not a formal portfolio or assessment record.
- The monitor and deterioration curves are simplified educational models, not physiological simulations. Vitals and timings are approximations for learning.
- The patient and evaluator are simulated and can be wrong. Evaluations are not formal assessments and must not be used for clinical decisions.
- Guideline steps are locally supplied summaries, not a licensed copy of the NDoH text. Verify every step and link against the current official Knowledge Hub publication before teaching use. A scenario with an unsupported step count is disabled rather than scored approximately.
- The WhatsApp webhook replies after returning HTTP 200 using FastAPI background tasks; in-process background tasks are not durable if the process stops mid-message.
