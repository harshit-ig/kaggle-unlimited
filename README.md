# kaggle-rotate

Runs a self-hosted LLM notebook on Kaggle, one account at a time, and **switches to the
next account without your client noticing**. Clients talk to a single local URL; the pool
boots the next GPU session *before* retiring the current one and flips traffic across
when the replacement is warm.

Built around `bonsai2_27b_pq2_0_kaggle_llamacpp.ipynb`: that notebook's setup cells are
reused verbatim, so the model, context length and server flags stay yours.

The shipped notebook serves **Ternary Bonsai 2 27B (PQ2_0)** through PrismML's
`llama.cpp` fork on `127.0.0.1:8080`, exposing an OpenAI-compatible API. There is no
Ollama anywhere in the stack — the ternary `PQ2_0`/`PTQ1_0` tensor types are new ggml
types that neither upstream `llama.cpp` nor Ollama can parse yet.

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
  llama-server + cloudflared tunnel       llama-server + cloudflared tunnel
  (warming up, weights still downloading)  (model resident, serving)
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

First boot clones `PrismML-Eng/Bonsai-demo`, compiles it with CUDA, and pulls the
7.2 GB `PQ2_0` GGUF, so expect **20–40 minutes** before the endpoint answers — this is a
full compiler build, not a package install. The pool prints a progress line for each
stage; `run/kaggle-rotate.log` has the full history. Subsequent sessions reuse the same
notebook, so treat the first one as the slow one.

## Point your client at it

Base URL `http://127.0.0.1:8317/v1`, any API key, model `ternary-bonsai-2-27b-pq2`
(declared by the source notebook and reported by `kaggle-rotate status`).

```bash
# OpenAI SDK / aider / Continue / Cline — all take the same three values
export OPENAI_API_BASE=http://127.0.0.1:8317/v1
export OPENAI_API_KEY=anything
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8317/v1", api_key="anything")
client.chat.completions.create(
    model="ternary-bonsai-2-27b-pq2",
    messages=[{"role": "user", "content": "hello"}],
)
```

Any `llama-server`-compatible path under `/v1` is proxied transparently, e.g.
`http://127.0.0.1:8317/v1/models`.

## Using a different model

There is no `model` config key. **The notebook is the single source of truth** for the
model, the weights and the context width, because it is the thing that actually loads
them: it downloads the GGUF, launches `llama-server`, and declares `MTP_MODEL` /
`NUM_CTX`. Editing config to disagree with the notebook would only produce a kernel that
advertises one model and serves another.

To serve something else, point `source_notebook` at a notebook that loads it:

```toml
[kernel]
source_notebook = "your_notebook.ipynb"
```

Two requirements, both structural:

- **The notebook must end up serving an OpenAI-compatible API on `127.0.0.1:8080`.**
  The tunnel and the pool's readiness probe both target that port; a server on another
  port comes up fine and then 502s every request.
- **It must define `MTP_MODEL`** (the name clients send) and `NUM_CTX`. Both are read
  straight out of the notebook and used for the generated header and the model name the
  pool advertises. Missing values fall back to the shipped defaults rather than failing,
  so check `kaggle-rotate render` if the name looks wrong.

Benchmark, tunnel and diagnostics sections at the end of a notebook are dropped from the
rendered kernel, so they cost nothing per session.

Preview what would be pushed before spending quota:

```bash
uv run kaggle-rotate render --out /tmp/s.ipynb
```

## How the handoff works

Rotation is **prewarm-then-cutover**, never kill-then-restart:

1. The active session is `prewarm_lead_minutes` (default 120) from its 12h cap, or its
   account is near the 30h weekly cap. The pool picks whichever other account has the
   most budget left and pushes the kernel.
2. That kernel builds PrismML's `llama.cpp`, downloads the PQ2_0 GGUF, starts
   `llama-server`, opens its own quick tunnel, and **registers its tunnel URL with the
   relay** — that callback is how the pool learns an address it never had to scrape.
3. Once `llama-server` answers `/health` — which it only does once the model is
   resident — the proxy flips its active upstream. New requests go to the new session.
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
| `kernel.source_notebook` | *(Bonsai notebook)* | Serve a different model; the notebook owns the model, weights and context |
| `kernel.max_runtime_minutes` | `660` | Lower it to rotate sooner, so a slow boot has slack |
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
- **Notebook boot time is the hard floor on lead time**, and it is much larger with this
  runtime than it was under Ollama. Budget **20–40 minutes** for the first boot: a git
  clone, a CUDA compile of `llama.cpp`, a 7.2 GB download, then the model load at
  262,144 context. `rotation.prewarm_lead_minutes` (default 120) must comfortably exceed
  it or a rotation will open a gap where no session is serving. Verify it on your first
  boot and raise the lead if the log shows it taking longer.
  `rotation.boot_timeout_seconds` (default 2400) also has to cover that build — raise it
  if the pool gives up mid-boot.
- **Weekly quota is estimated from wall time**, since Kaggle only reports real GPU
  seconds after a session ends. Slight over-counting is deliberate.
- **Boot is a compiler build, not a package install.** Each session clones
  `Bonsai-demo` and builds it with CUDA, then pulls 7.2 GB. A published Kaggle dataset
  holding the compiled runtime and the GGUF would cut this substantially — it is the
  single biggest win available.
- `cleanup` deletes the kernel this tool created (`kaggle-rotate-llamacpp` under each
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