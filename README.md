# Sister Vuo

Sister Vuo is an AI nursing patient simulator built with Python 3.11+ and FastAPI. A student nurse takes a history from a fictional patient played by a Llama model, then receives structured feedback from a separate evaluator model.

This is an educational simulation. It is not a clinical decision-support tool, and it must not contain real patient data.

## Features

- Three fictional scenarios:
  1. Abdominal pain in a woman of childbearing age
  2. Paediatric dehydration, with the mother speaking for the child
  3. Chest pain in an older adult
- In-character patient replies in one to three sentences
- Hidden diagnosis and red flags withheld from the patient prompt
- Automatic evaluation after `END` or 12 student messages
- Scores for history taking, clinical reasoning, communication and empathy, language accessibility, and safety and referral
- Browser chat UI served by FastAPI
- WhatsApp Cloud API webhook with per-phone sessions
- Local JSONL transcript logs
- API and webhook tests with no external model calls

## How it works

The application has two LLM roles sharing one OpenAI-compatible Llama endpoint:

- **Patient model**: one chat completion per student message at temperature `0.7`. It receives only the patient's subjective scenario information. Hidden diagnosis and ground-truth data are deliberately excluded.
- **Evaluator model**: one completion per session at temperature `0.2`, after `END` or after the 12th student message. It receives the scenario ground truth and transcript and must return a single JSON object.

If the evaluator returns invalid JSON, the app retries once with a stricter instruction. Sessions are stored in memory and are lost when the process restarts.

## Project layout

```text
app/
  config.py          Environment and path settings
  models.py          Pydantic request, response, and evaluation models
  llm.py             OpenAI-compatible client, patient prompt, evaluator parsing
  sessions.py        Scenario loading, sessions, turn handling, transcript logs
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
tests/
  test_api.py
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

`LLM_BASE_URL` should be the API root that ends in `/v1` or similar. The app calls `POST {LLM_BASE_URL}/chat/completions`. If the provider supports OpenAI JSON mode, set `LLM_JSON_MODE=true`; otherwise the evaluator prompt itself requests JSON and the parser recovers fenced or surrounding text.

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
  -d "{\"scenario_id\":\"abdominal_pain\",\"student_name\":\"Alex\"}"
```

Scenario IDs are `abdominal_pain`, `paediatric_dehydration`, and `chest_pain`.

The response includes `session_id`, a safe scenario summary, the patient's opening line, and the 12-turn limit.

### Send a student message

```bash
curl -X POST http://127.0.0.1:8000/session/SESSION_ID/message \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"When did the pain start?\"}"
```

The response contains the patient reply, turn counts, status, and `evaluation` if the 12-turn limit was reached. If evaluation fails at the limit, the response includes `evaluation_error`; call the end endpoint to retry.

### End and evaluate

```bash
curl -X POST http://127.0.0.1:8000/session/SESSION_ID/end
```

Calling `/end` again returns the stored evaluation without another model call.

Other routes:

- `GET /scenarios` - public scenario summaries
- `GET /session/{id}` - status and transcript snapshot
- `GET /health` - configuration and active-session status

## Evaluation output

The evaluator returns and the API exposes:

```json
{
  "overall_score": 3.4,
  "domain_scores": {
    "history_taking": 4,
    "clinical_reasoning": 3,
    "communication_empathy": 4,
    "language_accessibility": 3,
    "safety_referral": 3
  },
  "strengths": ["..."],
  "missed_red_flags": ["..."],
  "unsafe_actions": [],
  "recommended_learning": ["..."],
  "feedback_to_student": "..."
}
```

`overall_score` is recomputed by the server as the mean of the five whole-number domain scores, rounded to one decimal place. This prevents inconsistent arithmetic from the model.

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
- `END` - evaluates the active session
- Any other text - sent to the active patient; if there is no active session, the menu is returned

The session ends automatically after 12 student messages and the feedback is sent to the same phone number. Starting a new session while one is active is blocked; reply `END` first.

WhatsApp free-form replies are appropriate inside the 24-hour customer service window. Production use also requires Meta app review, business verification, messaging limits, and compliance with WhatsApp policy.

## Logs and privacy

- Each session writes `logs/transcripts/{session_id}.jsonl` with start, student messages, patient messages, evaluation, and evaluation failures.
- Transcripts contain fictional patient dialogue and student messages. Do not enter real patient information.
- WhatsApp phone numbers are kept only in memory for routing. Logs contain a masked value such as `***1234`, never the full number.
- `.env` and `logs/` are gitignored.
- API keys are never logged or returned to clients.

Access control:

- With `APP_API_KEY` set, `/scenarios` and every `/session/...` endpoint require an `X-API-Key` header. The chat UI asks for the key once and keeps it in the browser. Without it the API is open, which is fine on localhost only.
- Sessions idle for `SESSION_TTL_MINUTES` (default 120) are removed from memory, and at most `MAX_ACTIVE_SESSIONS` (default 200) can be active at once; beyond that, starting returns HTTP 503.
- The evaluator receives the transcript as a JSON array, so a student cannot forge patient turns by typing line breaks.

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check app tests
python -m compileall -q app
```

Tests substitute fake LLM and WhatsApp clients, so they do not need model credentials or network access. They cover the full session flow, automatic evaluation, evaluator failure and retry, transcript logging, scenario privacy, webhook verification, signatures, deduplication, and WhatsApp commands.

## Limitations

- Sessions and phone mappings are in memory; a restart clears them. Run a single Uvicorn worker, because multiple workers would have separate session stores.
- There is no database, per-user accounts, per-client rate limiting, or streaming. The single `APP_API_KEY` is shared by everyone who uses the UI.
- The patient and evaluator are simulated and can be wrong. Evaluations are not formal assessments and must not be used for clinical decisions.
- The WhatsApp webhook replies after returning HTTP 200 using FastAPI background tasks; in-process background tasks are not durable if the process stops mid-message.
