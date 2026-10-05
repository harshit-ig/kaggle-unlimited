# kaggle-rotate

Runs your Ollama notebook on Kaggle, one account at a time, and **switches to the next
account without your client noticing**. Clients talk to a single local URL; the pool
boots the next GPU session *before* retiring the current one and flips traffic across
when the replacement is warm.

Built around `qwen38_27b_kaggle_ollama_mtp_clean_64k.ipynb`: that notebook's setup
cells are reused verbatim, so the model, context length, and MTP settings stay yours.

```
your client (Cline / aider / Continue / OpenAI SDK)
    │  http://127.0.0.1:8317/v1          ← one stable URL, forever
    ▼
┌──────────────────────────── local ────────────────────────────┐
│  proxy      :8317   forwards to whichever session is active  │
│  relay      :8318   ← published via a Cloudflare quick tunnel│
│  ledger             per-account 12h/30h accounting            │
└───────────────────────────▲───────────────────────────────────┘
                            │ register / heartbeat / shutdown
        ┌───────────────────┴────────────────────┐
        ▼                                        ▼
  Kaggle kernel "acct-a"                  Kaggle kernel "acct-b"
  ollama + cloudflared tunnel             ollama + cloudflared tunnel
  (warming up, model not loaded yet)       (model resident, serving)
```

## Setup (once)

You need the `kaggle` package (it comes with this project) and one `kaggle.json` per
account, from <https://www.kaggle.com/settings/api> → **Create Legacy API Key**.
Each Kaggle account also needs phone verification before it gets a GPU.

```bash
uv sync                                   # or: pip install -e .
uv run kaggle-rotate doctor               # checks ports, CLI, source notebook
uv run kaggle-rotate init                 # walks through each kaggle.json
```

`init` copies each credential into `~/.config/kaggle-rotate/accounts/<slug>/` with
`0600` permissions and verifies it against the real API before accepting it:

```
Path to a kaggle.json (blank when done, or 'token' to paste an API token): ~/Downloads/kaggle-alice.json
  account slug (blank to derive from username): alice
  ok alice: authenticated as alice
```

Non-interactive equivalent: `uv run kaggle-rotate add-account bob --credential ~/Downloads/kaggle-bob.json`

## Run

```bash
uv run kaggle-rotate run            # foreground; Ctrl-C stops and cleans up
uv run kaggle-rotate status         # sessions, budgets, remaining session time
uv run kaggle-rotate accounts       # weekly quota per account
uv run kaggle-rotate stop           # stop a backgrounded pool
```

First boot downloads Ollama's runtime plus ~17 GB of weights and waits for a GPU, so
expect 10–25 minutes before the endpoint answers. The pool prints a progress line for
each stage; `run/kaggle-rotate.log` has the full history.

## Point your client at it

Base URL `http://127.0.0.1:8317/v1`, any API key, model `qwen3.8-27b-uncensored-mtp`.

```bash
# OpenAI SDK / aider / Continue / Cline — all take the same three values
export OPENAI_API_BASE=http://127.0.0.1:8317/v1
export OPENAI_API_KEY=anything
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8317/v1", api_key="anything")
client.chat.completions.create(
    model="qwen3.8-27b-uncensored-mtp",
    messages=[{"role": "user", "content": "hello"}],
)
```

Raw Ollama endpoints work too: `http://127.0.0.1:8317/api/generate`.

## Using a different model

Set `model` in `config.toml`. Two kinds:

```toml
# 1. Any Ollama library model. Nothing to build - the notebook just pulls it.
[kernel]
model = "qwen3:30b-a3b"
# also fine: "llama3.1:70b-instruct-q4_K_M", "deepseek-r1:14b", "gpt-oss:20b"

# 2. A Hugging Face GGUF, derived into a local variant (what the shipped notebook
#    does, so speculative decoding stays available).
[kernel]
model        = "qwen3.8-27b-uncensored-mtp"
source_model = "hf.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF:Q4_K_M"
```

`derive_model` is inferred: a colon tag with no `hf.co/` prefix is treated as a
library model. Set `derive_model = true` or `false` to override.

Two things to know when switching:

- **Library models get no speculative decoding.** `draft_num_predict` only applies to
  a derived variant, so it is ignored on the library path.
- **Your source notebook's own model assignment is rewritten**, not just this tool's
  cells. Otherwise it would keep creating and loading the old model while the
  prewarm cell waited for the new one that never arrives.

Preview what would be pushed before spending quota:

```bash
uv run kaggle-rotate render --out /tmp/s.ipynb
```

## How the handoff works

Rotation is **prewarm-then-cutover**, never kill-then-restart:

1. The active session is `prewarm_lead_minutes` (default 120) from its 12h cap, or its
   account is near the 30h weekly cap. The pool picks whichever other account has the
   most budget left and pushes the kernel.
2. That kernel installs Ollama, pulls the weights, opens its own quick tunnel, and
   **registers its tunnel URL with the relay** — that callback is how the pool learns
   an address it never had to scrape. It then warms the model into VRAM.
3. Once the pool health-checks the new endpoint *and* confirms the model is resident,
   the proxy flips its active upstream. New requests go to the new session.
4. The old session keeps answering for `drain_grace_seconds` so in-flight streams
   finish, then its kernel is **deleted**. Kaggle stops billing it.

The proxy binds the upstream per request, so a request in flight at step 3 completes on
the session it started on. A request that fails to connect is retried once against the
warm standby.

Sessions also self-terminate: the notebook exits at `max_runtime_minutes` (default 11h,
under Kaggle's 12h cap) and shuts itself down if the relay goes unreachable for 15
minutes, so an orphaned kernel never burns quota.

## Configuration

Copy `config.example.toml` to `config.toml`. Unknown keys are a hard error. The knobs
you are most likely to want:

| Key | Default | Why you would change it |
| --- | --- | --- |
| `rotation.prewarm_lead_minutes` | `120` | Boot is slow; raise it if a rotation misses the cap |
| `rotation.max_active_sessions` | `1` | Raise to keep a permanently warm spare |
| `rotation.idle_stop_minutes` | `0` | Set `30` to release the GPU when nothing is calling |
| `rotation.weekly_limit_hours` | `30` | Kaggle's quota "is 30 hours or sometimes higher" — set what your account actually shows |
| `kernel.accelerator` | `NvidiaTeslaT4` | `NvidiaTeslaT4` is 2× T4; also available: `NvidiaTeslaA100`, `NvidiaL4`, `NvidiaH100` |
| `kernel.model` | *(from notebook)* | Serve a different model; library refs like `qwen3:30b-a3b` just work |
| `kernel.derive_model` | *(inferred)* | `false` to pull a library model instead of deriving a variant |
| `kernel.num_ctx` | `65536` | Lower it if the model will not fit alongside the KV cache |
| `relay.public_url` | *(unset)* | Use a named Cloudflare tunnel instead of a quick tunnel |

Per-account weekly caps live in `~/.config/kaggle-rotate/accounts.json` (`weekly_limit_hours`).

## Inspecting what gets pushed

```bash
uv run kaggle-rotate render --out /tmp/server.ipynb
```

Each launch writes `run/launches/<account>/` containing `server.ipynb`,
`kernel-metadata.json`, and `launch.json` (the per-launch relay URL and token).

## Security notes

- The **relay** is reachable from the internet because Kaggle has to call it. Every
  route except `/healthz` requires a per-launch bearer token, so it cannot be used to
  register a rogue endpoint. Verified: unauthenticated registration returns 401.
- The **model endpoint** behind each kernel's quick tunnel is *not* authenticated —
  that is a property of Cloudflare quick tunnels, not of this tool. Point clients at
  `127.0.0.1` and never share a `trycloudflare.com` URL.
- Both `/tmp` credential copies are `0600`; `launch.json` holds a live token and is
  `0600` too.

## Stopping a session

Deletion *is* the stop. There is no graceful path that can be trusted, because the
notebook can only be told to exit through the relay tunnel — and if the relay is down,
wedged, or the pool was killed outright, the command never arrives. So:

- **Ctrl-C / `kaggle-rotate stop`** deletes every managed kernel. Immediate.
- **`kaggle-rotate cleanup`** deletes them again if a previous session was killed
  (SIGKILL, closed terminal, crashed machine). Run it whenever `status` shows a
  session that should not exist.
- **Backstop:** a kernel that cannot reach the relay for `kernel.orphan_grace_seconds`
  (default 5 min) stops itself, and hard-stops at `kernel.max_runtime_minutes`.

```bash
uv run kaggle-rotate cleanup -n     # show what would be deleted
uv run kaggle-rotate cleanup        # delete it
```

## Known limits

- **The first request after a cutover is not instant** for clients with a cold prefix
  cache; the *connection* never drops, but the new session's KV cache is empty.
- **If the pool dies but its `cloudflared` child survives, the relay keeps answering**
  and a kernel would never notice it had been abandoned. The pool now pulses the relay
  every tick, the relay reports `driver_alive`, and a kernel that stops seeing pulses
  shuts itself down after `kernel.orphan_grace_seconds`.
- **Notebook boot time is the hard floor on lead time.** If a session fills its 12h
  and the next one needs 25 minutes to become resident, a short `prewarm_lead_minutes`
  guarantees a gap. 120 minutes is comfortable.
- **Weekly quota is estimated from wall time**, since Kaggle only reports real GPU
  seconds after a session ends. Slight over-counting is deliberate.
- **Ollama keeps models on the Kaggle disk**, so every session re-pulls ~17 GB. Using a
  published dataset of the weights would cut boot time substantially.
- `cleanup` deletes the kernel this tool created (`kaggle-rotate-ollama` under each
  account). It cannot distinguish that kernel from your own runs, so point
  `kernel.kernel_slug` at something you do not use for anything else.

## Before you point real accounts at this

Kaggle's Terms of Service require one account per person. Rotating several accounts to
multiply free GPU hours is circumvention, and the platform does detect and disable
accounts for it — that would cost you every account you have on the platform, not just
the spares. This tool is built for accounts you are entitled to use (a team, an org
that bought extra quota) and for keeping one session alive across the weekly reset. If
you want sustained GPU capacity rather than a demo, Kaggle's own paid GPU options are
the supported route.

## Development

```bash
uv run pytest -q          # 60 tests
uv run ruff check .
```

`tests/test_integration.py` is the one that matters: it runs the real supervisor
against a stubbed `kaggle` CLI and stand-in kernels, then asserts that client traffic
actually moves from one account to the other and that the retired kernel is told to
stand down. `tests/test_supervisor.py` executes the generated notebook's control cells
verbatim against a live relay, so a mismatch between the two sides fails the suite.