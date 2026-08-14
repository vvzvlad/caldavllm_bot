"""Smoke gate for the caldavllm_bot image, run on the CI runner against an already-built image.

It is split in two, and the split is the thing to understand before changing anything here.

The OUTER half is this file: a plain `python3 ci/smoke.py` on the runner that drives `docker`
against the tag it is given in $SMOKE_IMAGE. It answers the questions that are about the
container as an object — what `docker inspect` says its CMD, WORKDIR and environment are, how the
image behaves when the variables it needs are missing or wrong, whether a container is still alive
when everything else is done, and what the image writes to its log when started with its REAL
command. None of those can be answered from a process already running inside it.

The INNER half is the PROBE below: a program that lives here as a string constant and is fed to
`docker exec -i <name> python -u -`. It answers everything that is about the code INSIDE the image
— that every module imports, that the third-party symbols the code actually calls are present,
that `get_settings()` really returns the settings the rest of the app indexes into, that
`UserManager` really writes and reads credentials on a real filesystem, and that `CalendarBot` can
be constructed with its full handler set. It runs inside for two reasons. The obvious one is that
those objects only exist in there. The other is the one that shapes this whole file: **this job
and the docker daemon are not in the same network namespace.** Gitea's act_runner executes the job
inside its own job container while the `docker` CLI it provides drives a daemon that lives outside
it, so nothing here may assume it can reach a container by address.

NO PORT IS PUBLISHED by anything in this file, and for this service that is not even a trade-off:
the bot is a long-polling Telegram client and has no listening socket at all. There is nothing to
publish and nothing to curl, so the whole gate is built out of `docker inspect`, `docker logs` and
one `docker exec`. Publishing one would not have worked anyway — the port would land in the host
daemon's network namespace, which this job cannot reach.

`docker exec -i <name> python -u -` is the mechanism the fleet already uses: `-i` attaches stdin
without asking for a tty, the image's own interpreter runs the program, and `docker exec`
propagates the command's exit status — which is what lets the outer half tell "the probe reported
failures" from "the probe died before it could report anything".

The runner's job image (`docker.gitea.com/runner-images:ubuntu-latest`) ships /usr/bin/python3;
note that it ships `python3` and NOT a bare `python`, which is why both workflows invoke this as
`python3 ci/smoke.py`. Inside the container it is the other way round — the image's own interpreter
is `python` — and this file invokes it accordingly. Nothing beyond the standard library is imported
on the outer side; the probe imports this repo's own modules and the six pinned wheels in
requirements.txt, which is exactly the point of running it in there.

Why this gate carries the whole load
------------------------------------
There is no test job in either workflow, and that is deliberate rather than an oversight. The one
test module in this repo, tests/test_llm.py, has an autouse fixture that fails without an API_KEY
in the environment and then calls the real `parse_calendar_event()` against a live LLM — it is a
paid, non-deterministic, integration-grade check of parsing QUALITY, not a unit suite. Running it
on every push would spend money to produce a red build whenever a model felt differently about a
sentence. The full reasoning is at the top of both files in .gitea/workflows/.

So everything that stands between a commit and production is below, and every check is written
against a way this image has a realistic chance of shipping broken:

* (a) the image's declared contract: CMD, WORKDIR, no ENTRYPOINT — and PYTHONUNBUFFERED. WORKDIR
  is not cosmetic: `UserManager.data_dir` is the RELATIVE string "data", so a moved working
  directory silently puts every user's CalDAV credentials somewhere that is not the mounted
  volume. PYTHONUNBUFFERED is what makes check (b) possible at all; see its note below. Outer half.
* (b) the startup guard still fires AND still says why, in all three of its branches. Outer half —
  it needs `docker run` with a deliberately broken environment.
* (c) the code inside the image imports, and the third-party symbols its call sites use exist.
  Probe.
* (d) `get_settings()` returns a real settings dict with the documented defaults, and does NOT
  kill the process on a valid environment. Probe.
* (e) `UserManager` round-trips CalDAV credentials through a real file. Probe.
* (f) `CalendarBot` constructs with all eight handlers registered. Probe.
* (g) the image's real CMD gets through import and construction. Outer half — it needs
  `docker run` with the image's own command and `docker logs`.
* (h) the container the probe ran in is still alive afterwards. Outer half.

Nothing here holds a credential or reaches an external service. Every value is invented by this
file: a syntactically valid but fake bot token, fake API keys, and a loopback address on a port
nothing listens on for the Telegram API server. A gate that needed the deployment's real token
could not run on a pull request at all, and one that reddened on somebody else's outage is a gate
everybody learns to ignore.

Two properties matter and are easy to lose, so they are stated where they can be checked:

* Failures leave through SystemExit, never `assert` — on both sides of the split. Asserts vanish
  under PYTHONOPTIMIZE=1, which would silently turn this gate permanently green.
* Every check runs before the run is judged, so one run shows the full extent of the breakage
  instead of only the first broken thing. A check that CANNOT run reports itself as FAILED; it is
  never quietly skipped, which is the classic way a gate keeps reporting success while proving
  less and less. The probe's own report lines are parsed back into this gate's report, so a
  failure inside the container is one row in the same list as a failure outside it.
"""

import json
import os
import subprocess
import time

# The tag to test. Required rather than defaulted: a default would let a mistyped `env:` block in
# a workflow silently gate some other image that happens to be on the daemon.
IMAGE_ENV = "SMOKE_IMAGE"
# Base name for every container this gate starts. The workflows put the run id in it, because the
# runner has a single docker daemon shared by every repository in the fleet and two concurrent runs
# must not collide on a name.
# Required, exactly like the tag above, and for two reasons that both end in silence rather than in
# an error — see the message in main() where it is enforced.
NAME_ENV = "SMOKE_NAME"

# The five containers this gate starts, by suffix on $SMOKE_NAME:
#   ""          the long-lived one the probe runs inside. It is started with a sleeping command
#               rather than the image's own, because the image's own command talks to Telegram.
#   -noenv      guard case 1 for check (b): no environment at all.
#   -notoken    guard case 2: an LLM key but no BOT_TOKEN.
#   -provider   guard case 3: a complete environment with an unknown LLM_PROVIDER.
#   -cmd        the one started with the image's REAL command for check (g).
# All five are NAMED rather than left to docker's random name generator, and the reason is the one
# case that matters: `subprocess` hitting its timeout kills the docker CLIENT on the runner, not
# the container on the daemon. With no name nobody could ever remove the survivor — not
# remove_container() here, not the workflow's `if: always()` step — and it would go on pinning the
# image, so the `docker rmi` at the end of the job would fail too, be swallowed by its `|| true`,
# and leave both a container and a few hundred MB of image on the daemon the whole fleet shares.
# Kept in step with the suffix list in both workflows' cleanup steps: if you move one, move the
# others.
CMD_SUFFIX = "-cmd"

# The Dockerfile's WORKDIR and CMD, and the data directory that resolves against that WORKDIR.
# Hardcoded rather than read back out of the image: these are the contract between this repo and
# its image, so a Dockerfile that quietly moves them has to go red here and be looked at, not be
# politely followed.
APP_DIR = "/app"
EXPECTED_CMD = ["python", "main.py"]
DATA_DIR = "/app/data"

# The one environment variable the image itself must declare. It is checked here — rather than
# only being set in the Dockerfile and hoped for — because check (b) below is built entirely on
# matching the TEXT the startup guard prints, and that text reaches `docker logs` only if it was
# flushed before `os._exit(1)` threw the process away without running interpreter shutdown. The
# full mechanism is written out in the Dockerfile next to the ENV line. Dropping the variable would
# not fail loudly on its own — it would make an operator's diagnosis of "why is this container
# exiting 1" depend on the internals of whatever logging library is installed that month — so it
# is asserted here, in the same file as the checks that consume it.
EXPECTED_ENV = "PYTHONUNBUFFERED=1"

# The three ways a real deployment gets its environment wrong, each with the fragment of THIS
# REPO'S OWN wording that `get_settings()` prints in reply. They are separate containers rather
# than one, because src/config.py checks them in this order and stops at the first failure: with no
# environment at all only the API-key branch is ever reached, so a BOT_TOKEN guard that had
# silently stopped working would be invisible behind it. One case per branch is what makes each
# branch individually provable.
#
# `env` is what gets passed with `-e`; everything absent from it is genuinely absent in that
# container. Note the progression: case 2 satisfies the API-key branch so it can reach the token
# branch, and case 3 satisfies both so it can reach the provider branch.
#
# On case 3's fragment, and it is worth reading before "improving" it: src/config.py writes that
# message as
#     logger.error("Unsupported LLM provider '%s'. Allowed providers: %s", llm_provider, ...)
# which is stdlib-logging style. loguru does not do %-interpolation — it formats with
# `str.format(*args)` — and the message contains no `{}`, so the arguments are silently dropped and
# the operator sees a literal `'%s'` where the rejected value should be. That is a real (small)
# bug in config.py, not in this gate, which is why the fragment matched below is only the fixed
# prefix. When config.py is fixed to name the value, tighten this entry to require the value
# itself: an operator staring at `Unsupported LLM provider '%s'` learns that SOMETHING is wrong
# with LLM_PROVIDER but not what was rejected, which on a stack with several similar variables is
# most of the diagnosis.
GUARD_CASES = [
    {
        "suffix": "-noenv",
        "label": "with no environment at all",
        "env": [],
        "fragment": "At least one API key must be set",
        "branch": (
            "the API-key branch of get_settings(). This is the case a stack redeployed with the "
            "keys dropped out of it hits, and `restart: unless-stopped` means it is restarted "
            "into the same failure forever — the exit code plus this line are the only thing "
            "telling an operator that a variable went missing rather than that the LLM is down"),
    },
    {
        "suffix": "-notoken",
        "label": "with an LLM key but no BOT_TOKEN",
        "env": ["GROQ_API_KEY=smoke-not-a-real-key"],
        "fragment": "BOT_TOKEN must be set",
        "branch": (
            "the BOT_TOKEN branch. It sits AFTER the API-key branch in src/config.py, so it is "
            "unreachable without a key present — which is exactly why it gets a container of its "
            "own rather than sharing case 1's"),
    },
    {
        "suffix": "-provider",
        "label": "with a complete environment but LLM_PROVIDER=nonsense",
        "env": [
            "BOT_TOKEN=123456789:AAFakeSmokeTokenNotReal_0123456789abc",
            "GROQ_API_KEY=smoke-not-a-real-key",
            "LLM_PROVIDER=nonsense",
        ],
        "fragment": "Unsupported LLM provider",
        "branch": (
            "the provider branch, the last of the three and the only one that can be reached with "
            "every required variable present. Without it a typo in LLM_PROVIDER would be caught "
            "only if config.py's default table still happened to contain the typo's nearest "
            "neighbour"),
    },
]

# The environment both the probe container and the real-CMD container run with. Every value is
# invented here, and each one is chosen so that nothing can reach anything:
#   * the token is fake but SYNTACTICALLY valid. aiogram validates the shape of the token in
#     `Bot.__init__` (digits, a colon, then a tail of legal characters) and raises before any
#     network call, so a token like "smoke" would fail `CalendarBot()` construction for a reason
#     that has nothing to do with the image;
#   * both LLM keys are set, as production sets them, so `get_settings()` takes the branch
#     production takes. Neither is ever used to make a request: nothing in this gate calls the LLM;
#   * LLM_PROVIDER=groq matches the deployment, so `get_llm()` builds the provider production
#     builds rather than the default one nobody runs;
#   * TELEGRAM_BOT_API_SERVER points at 127.0.0.1:1 — inside the container, on a port nothing
#     binds. It is set rather than left out on purpose: production sets it, so this makes
#     `CalendarBot.__init__` take the custom-server branch instead of the api.telegram.org default
#     that no deployment uses. Loopback rather than a hostname is deliberate for check (g): a name
#     would put the RUNNER's resolver in the loop, and a resolver that answers everything (a
#     captive portal, a hijacking upstream) would turn an instant refusal into a minute of connect
#     timeout;
#   * TZ matches the deployment. It only ever reaches the `caldav.timezone` settings key.
SMOKE_ENV = [
    "BOT_TOKEN=123456789:AAFakeSmokeTokenNotReal_0123456789abc",
    "DEEPSEEK_API_KEY=smoke-not-a-real-key",
    "GROQ_API_KEY=smoke-not-a-real-key",
    "LLM_PROVIDER=groq",
    "TELEGRAM_BOT_API_SERVER=http://127.0.0.1:1",
    "TZ=Europe/Moscow",
]

# The command the probe container runs INSTEAD of the image's own. The image's CMD reaches Telegram
# a few lines into startup (see check (g) below), so a probe container started with it would be
# racing a process that is on its way out. A sleeping container is a stable place to `docker exec`
# into, and check (g) covers the real command separately, in its own container, which is where that
# question belongs.
# 900 s is far beyond this gate's own worst case (see the timeout arithmetic below); the container
# is removed in a `finally` regardless, and the workflow removes it again under `if: always()`.
IDLE_COMMAND = ["python", "-c", "import time; time.sleep(900)"]

# The first line the bot logs, from `CalendarBot.start()`. Everything that can go wrong with the
# IMAGE happens before it: main.py constructs `CalendarBot()` first, and that constructor runs
# `get_settings()` (the guard), builds the aiogram Bot (token validation), builds the LLM provider,
# the CalendarManager and the UserManager (which creates data/ under the real WORKDIR), and
# registers all eight handlers. So the marker is a receipt for the whole of startup — and the
# absence of it is the only signal check (g) needs.
#
# It is also the LAST thing this gate can assert on, which is worth stating so nobody adds more:
# `start()` continues straight into `_advertise_commands()` -> `bot.set_my_commands(...)`, which is
# a REQUEST to Telegram. Against 127.0.0.1:1 that is refused instantly and the process dies with a
# traceback. That traceback is EXPECTED, and check (g) is careful to look for tracebacks only in
# the part of the log BEFORE this marker.
STARTUP_MARKER = "Starting bot..."

# Printed by the probe on its last line and nowhere else. DECLARED TWICE — here and again inside
# the PROBE text — and the two have to keep agreeing. The duplication is deliberate: interpolating
# it into the probe would mean `.format()` on a body that contains braces of its own, which would
# quietly turn the probe into a template that no longer runs as-is — and being able to paste that
# block into a container by hand and have it work is most of why it is a plain string. A drift
# costs a false RED (the outer half reports a probe that never finished), never a false green.
PROBE_MARKER = "caldavllm_bot smoke probe ok"

# The prefix put in front of every row the probe reported, so the merged report says where each
# verdict was decided. Without it a reader of thirty result lines cannot tell which of them were
# taken from inside the container and which from the runner.
PROBE_ROW_PREFIX = "[in-container] "

# Bounds. Every docker call gets one, because a gate that hangs is worse than a gate that fails: it
# holds a slot on a runner the whole fleet queues for until the step timeout kills it, and a killed
# step never runs its cleanup.
# The numbers are generous against the real cost — this image starts in a second or two — and are
# sized for a shared runner under load right after a build. Their worst-case SUM is what the smoke
# step's `timeout-minutes` in both workflows has to exceed, so it is spelled out here, in the order
# the calls actually happen:
#     30 (inspect config)
#   + 3 x (30 rm + 90 run) = 360   (the three guard cases)
#   + 30 (rm probe)     + 60 (probe run -d)  + 120 (docker exec: the probe)
#   + 30 (inspect probe state)
#   + 30 (rm cmd)       + 60 (cmd run -d)    + 45 (startup poll: 30 budget + one 15 s `logs`)
#   + 30 (inspect cmd state)
#   + 30 (rm cmd, finally) + 30 (rm probe, finally)
#   = 855 s, a little over 14 minutes. Both workflows allow 16.
# PROBE_TIMEOUT is the one number here that is not arbitrary: the probe touches no network at all,
# so its worst case is imports plus a handful of file operations — a couple of seconds — and 120
# leaves it two full minutes of headroom for a slow `docker exec` on a loaded daemon while still
# fitting the sum above inside the step timeout.
# Five of these `rm`s are pre-run cleanups: every container this gate starts is removed by name
# before it is started, so a re-run from the UI — which keeps the same run id, hence the same
# $SMOKE_NAME — cannot die on "name already in use".
INSPECT_TIMEOUT = 30
REMOVE_TIMEOUT = 30
GUARD_TIMEOUT = 90
START_TIMEOUT = 60
PROBE_TIMEOUT = 120
LOGS_TIMEOUT = 15

# The startup-marker poll, bounded in WALL CLOCK rather than in attempts: each attempt shells out
# to `docker logs`, whose own timeout is 15 s, so an attempt-counted bound would multiply into
# minutes the moment the daemon got slow. The marker is logged as soon as the constructor returns,
# so it arrives within a second or two of the container starting; 30 s is roughly fifteen times
# that.
STARTUP_BUDGET = 30
STARTUP_PAUSE = 0.5

# How much of an unexpected output reaches the log. Container logs can run to thousands of lines
# when something loops, and an unbounded dump would bury the verdict.
EXCERPT_CHARS = 4000


# The program fed to `python -u -` inside the container under test. It lives here as a string
# rather than as a second file so that this gate stays one artifact: the workflows already pass it
# nothing but an image tag and a container name, and a second file is a second thing to forget to
# copy when this repo's CI is cloned into the next one.
#
# `-u` is not decoration: this stdout is a pipe, and CPython block-buffers those. A normal exit
# flushes (SystemExit included, so a self-reported failure is never lost), but a probe killed by
# PROBE_TIMEOUT would otherwise take every line it had already printed down with it — in exactly
# the run whose output is the only thing left to diagnose from.
#
# It takes NO parameters. Everything it needs is already in the container's environment, because
# `docker exec` runs with the environment the container was created with — the very values
# SMOKE_ENV above passed to `docker run`. The only thing declared on both sides is PROBE_MARKER,
# and the note on it above explains why that one is copied rather than interpolated.
PROBE = r'''
"""Runs INSIDE a container of the caldavllm_bot image, fed to `python -u -` over stdin.

It is HERMETIC: not one line of it opens a socket to anything outside the container, and it never
calls the LLM. That is a requirement rather than a nicety, because this gate has to be runnable on
a pull request, where no credential for Telegram, for Groq, for DeepSeek or for anybody's calendar
exists and none should. What it therefore checks is everything about this image that can be decided
without a peer: that the code imports, that the third-party symbols it calls are really there, that
the settings function returns what the rest of the app indexes into, that the credential store
round-trips through a real file, and that the bot object can be built with its full handler set.

REPORTING IS STREAMED — each verdict is printed the moment it is decided, rather than collected and
printed at the end. That differs from the sibling gate in asakusa-tg-print, and the reason is
specific to this repo: `get_settings()` reports a bad environment by calling `os._exit(1)`, which
terminates the interpreter on the spot. A batched report would be thrown away in its entirety by
any check that tripped that path, in the one run where knowing which check tripped it is the whole
point. Streaming costs some interleaving with loguru's own output; the outer half only parses lines
that start with `ok   ` or `FAIL `, and loguru's start with a timestamp, so the report survives it.

Everything is checked before the run is judged, and a check that cannot run reports itself as
FAILED rather than being skipped. Failures leave through SystemExit, never `assert`.
"""

import asyncio
import json
import os
import shutil
import tempfile

# Printed on the last line and nowhere else. This is the outer half's PROBE_MARKER, written out
# again because this text is handed to another interpreter and cannot import anything from the file
# it lives in. If you move one, move the other; the note there explains why a drift costs a false
# red rather than a false green.
PROBE_MARKER = "caldavllm_bot smoke probe ok"

# The WORKDIR the image declares, checked here as well as in the outer half's `docker inspect` —
# and they are not the same statement. The outer half reads what the image DECLARES; this reads
# where a process actually LANDS, which is what makes `UserManager`'s relative "data" resolve.
# Production depends on the same thing for its CMD.
APP_DIR = "/app"
DATA_DIR = "/app/data"

# Every module in src/. Imported by name rather than by walking the directory, so a module that is
# silently missing from the image is a failure here instead of a shorter loop that passes. The
# Dockerfile copies them with a glob (`COPY src/*.py src/`), which is exactly the kind of line that
# can quietly copy fewer files than intended.
SRC_MODULES = [
    "src.config",
    "src.users",
    "src.calendar",
    "src.llm_base",
    "src.llm_deepseek",
    "src.llm_groq",
    "src.llm",
    "src.bot",
]

# The third-party symbols this repo actually reaches for, module by module. An import of the package
# alone proves much less than it looks: a wheel can import and still be missing the attribute a call
# site uses, which is what a major-version bump of aiogram or httpx looks like from in here. Every
# name below is taken from a real call site in src/ — grep for it before deleting one.
THIRD_PARTY_SYMBOLS = [
    # src/bot.py: the Bot/Dispatcher pair plus the `types` namespace every handler annotates with.
    ("aiogram", ["Bot", "Dispatcher", "types"]),
    ("aiogram.client.session.aiohttp", ["AiohttpSession"]),
    ("aiogram.client.telegram", ["TelegramAPIServer"]),
    ("aiogram.filters", ["Command"]),
    # Reached as `types.X` in src/bot.py: the five constructed or annotated there.
    ("aiogram.types", [
        "BotCommand", "CallbackQuery", "InlineKeyboardButton", "InlineKeyboardMarkup", "Message",
    ]),
    # src/calendar.py builds a DAVClient for every calendar operation.
    ("caldav", ["DAVClient"]),
    # src/llm_groq.py and src/llm_deepseek.py both build an AsyncClient and catch these two.
    ("httpx", ["AsyncClient", "TimeoutException", "RequestError"]),
    # Every module logs through the singleton.
    ("loguru", ["logger"]),
    # src/config.py calls it at the top of get_settings().
    ("dotenv", ["load_dotenv"]),
    # NOTHING under src/ imports pytz today — it is pinned in requirements.txt because production
    # carries it. This row is therefore a weaker statement than the others on purpose: it says the
    # wheel installs and imports in this image, not that a call site works. It is kept because a
    # pinned dependency that stopped installing (a yanked release, a platform with no wheel) should
    # be a red build here rather than a surprise the next time somebody starts using it.
    ("pytz", ["timezone", "utc"]),
]

# src/llm.py is a facade: it re-exports both provider classes so that `from src.llm import
# DeepSeekLLM` keeps working (tests/test_llm.py does exactly that), and src/llm_base.py declares the
# Protocol both providers are checked against. Neither is exercised by importing src.llm alone — a
# facade that lost one of its re-exports still imports perfectly well — so they get their own row.
FACADE_SYMBOLS = [
    ("src.llm", ["DeepSeekLLM", "GroqLLM", "get_llm"]),
    ("src.llm_base", ["LLMProvider"]),
]

# What `get_settings()` must return. These are the keys the rest of the app indexes into with
# `settings["..."]` — an index, not a `.get()`, in most call sites — so a key that disappeared would
# be a KeyError at the moment a user sends a message, not at startup.
EXPECTED_SETTINGS_KEYS = [
    "batch_timeout",
    "caldav",
    "daily_token_limit",
    "deepseek_api_key",
    "groq_api_key",
    "llm_provider",
    "max_batch_size",
    "model",
    "telegram_bot_api_server",
    "telegram_token",
]

# The values `get_settings()` must produce when only the two REQUIRED variables are set. These are
# the numbers the service runs on in production, because the deployment sets none of the optional
# variables that would override them — so a default that drifted would change live behaviour
# without anybody editing the stack. `model` is the interesting one: it comes from config.py's
# per-provider default table, so this row also proves the table still maps "groq" to the model the
# deployment expects to be billed for.
EXPECTED_DEFAULTS = [
    ("llm_provider", "groq"),
    ("model", "openai/gpt-oss-120b"),
    ("daily_token_limit", 30000),
    ("batch_timeout", 0.8),
    ("max_batch_size", 30),
]

# The two variables `get_settings()` refuses to start without, and the value used for each. The
# token has to satisfy aiogram's format check further down.
REQUIRED_ENV = {
    "BOT_TOKEN": "123456789:AAFakeSmokeTokenNotReal_0123456789abc",
    "GROQ_API_KEY": "smoke-not-a-real-key",
}

# Every variable get_settings() reads that has a default. They are removed from the environment for
# the defaults check, so that what is measured is config.py's own table and not whatever the
# container happens to have been started with.
OPTIONAL_ENV = [
    "DAILY_TOKEN_LIMIT",
    "DEEPSEEK_API_KEY",
    "LLM_PROVIDER",
    "MAX_BATCH_SIZE",
    "MESSAGE_BATCH_TIMEOUT",
    "MODEL",
    "TELEGRAM_BOT_API_SERVER",
    "TZ",
]

# The default timezone, checked separately because it is nested one level down under "caldav" and
# is the value written into every VEVENT this bot creates. A drift here does not fail anything — it
# silently books everyone's appointments in the wrong zone.
EXPECTED_TIMEZONE = "Europe/Moscow"

# The credentials the UserManager check writes and reads back. The password deliberately carries
# non-ASCII characters, a double quote and punctuation: src/users.py stores it with `json.dump` into
# a utf-8 file and reads it with `json.load`, and a round trip that mangles it would hand a user's
# calendar server a password that is almost right.
STORE_USER_ID = 424242424
STORE_ABSENT_USER_ID = 999999999
STORE_CREDENTIALS = {
    "username": "smoke@example.test",
    "password": 'смоук "p@ss" :/#1',
    "url": "https://caldav.example.test/dav/",
    "calendar_name": "Смоук календарь",
}

# The handler set src/bot.py registers: seven message handlers (/start, /google, /fastmail,
# /caldav, /stats, the photo filter and the catch-all) and one callback_query handler. Counted from
# the real aiogram structures rather than by reading the source, so a decorator that silently
# stopped registering — a filter that raises during registration, a handler commented out — shows
# up here. Per event type rather than as one total, so a drift says WHICH half moved.
# This is an exact match, not a lower bound: adding a handler is supposed to fail this check once,
# loudly, and be answered by updating this constant.
EXPECTED_HANDLERS = {"message": 7, "callback_query": 1}

# aiogram's Dispatcher registers ONE handler of its own on its `update` observer in its constructor
# (`_listen_update`), and this bot registers nothing there, so the observer is skipped when counting.
# Note that src/bot.py hangs its handlers straight off the Dispatcher rather than off a sub-router,
# which is why the count below walks the dispatcher itself and not only `dp.sub_routers` — a
# sub-router-only walk would report an empty dict here.
IGNORED_DISPATCHER_OBSERVERS = {"update"}


def describe(error):
    return "{}: {}".format(type(error).__name__, error)


def count_handlers(dispatcher):
    """Handlers registered under a Dispatcher, per event type, as aiogram really stores them.

    Walks the dispatcher AND every router it includes, counting each observer's `handlers` list.
    The dispatcher's own `update` observer is excluded: aiogram registers a built-in handler there
    in its constructor, so counting it would yield nine where eight were registered and the extra
    one would belong to the framework rather than to this repo.
    """
    routers = [dispatcher]
    pending = list(dispatcher.sub_routers)
    while pending:
        router = pending.pop()
        routers.append(router)
        pending.extend(router.sub_routers)

    counts = {}
    for router in routers:
        for name, observer in router.observers.items():
            if router is dispatcher and name in IGNORED_DISPATCHER_OBSERVERS:
                continue
            handlers = getattr(observer, "handlers", None)
            if handlers:
                counts[name] = counts.get(name, 0) + len(handlers)
    return counts


def check_working_directory():
    """A process in this container must land in /app, which is what makes relative paths work."""
    target = "the container's working directory is {}".format(APP_DIR)
    cwd = os.getcwd()
    if cwd == APP_DIR:
        return [(target, None)]
    return [(target, (
        "it is {!r}. UserManager.data_dir is the RELATIVE string 'data', so a process that starts "
        "anywhere else writes every user's CalDAV credentials outside the mounted volume — where "
        "they are lost on the next redeploy and every user has to run /caldav again".format(cwd)))]


def check_no_dotenv():
    """No .env may be baked into the image.

    `get_settings()` opens with `load_dotenv()`, which reads a `.env` from the working directory. An
    image carrying one would hand BOT_TOKEN and the LLM keys to every container started from it —
    somebody's own credentials, published — and would silently disable the startup guard that the
    outer half's check (b) is built on. .dockerignore lists `.env` for exactly this; the check is
    here because a Dockerfile edit could undo that without touching .dockerignore.
    """
    target = "no .env file is baked into {}".format(APP_DIR)
    path = os.path.join(APP_DIR, ".env")
    if not os.path.exists(path):
        return [(target, None)]
    return [(target, (
        "{!r} exists in the image. get_settings() calls load_dotenv(), so this file is read at "
        "startup: it can ship a developer's own bot token and LLM keys inside a published image, "
        "and it makes the missing-variable guard unreachable".format(path)))]


def check_data_dir():
    """The state directory exists and is writable.

    It is the mount point of the production volume and the ONLY state this service has: one JSON
    file per user, holding that user's CalDAV username and password in plain text. A directory that
    is missing or read-only means /caldav answers "не удалось сохранить настройки" for everybody,
    and src/users.py swallows the underlying OSError into a log line, so nothing louder happens.
    """
    rows = []
    exists_target = "{} exists and is a directory".format(DATA_DIR)
    write_target = "...and is writable, so credentials can be stored at all"

    if not os.path.isdir(DATA_DIR):
        rows.append((exists_target, (
            "it is not there (or is not a directory). The Dockerfile creates it and production "
            "mounts a volume over it; without it every /caldav command fails to save")))
        rows.append((write_target, "not attempted: the directory is not there"))
        return rows
    rows.append((exists_target, None))

    probe_path = os.path.join(DATA_DIR, ".smoke-write-probe")
    try:
        with open(probe_path, "w", encoding="utf-8") as handle:
            handle.write("smoke")
        os.unlink(probe_path)
    except Exception as error:
        rows.append((write_target, (
            "writing a file into it raised {}. src/users.py catches that exception and only logs "
            "it, so in production this looks like /caldav quietly never working".format(
                describe(error)))))
        return rows
    rows.append((write_target, None))
    return rows


def check_src_imports():
    """(c) Every module under src/ imports inside the image."""
    rows = []
    for name in SRC_MODULES:
        target = "import {}".format(name)
        try:
            __import__(name)
        except Exception as error:
            rows.append((target, (
                "it raised {}. The module is missing from the image, or one of its imports is: the "
                "Dockerfile copies src/ with a glob, so a partial copy and a wheel that did not "
                "install look the same from here".format(describe(error)))))
            continue
        rows.append((target, None))
    return rows


def check_third_party_symbols():
    """(c) The third-party names this repo's call sites use are present in the installed wheels."""
    rows = []
    for module_name, symbols in THIRD_PARTY_SYMBOLS:
        target = "{} provides {}".format(module_name, ", ".join(symbols))
        try:
            module = __import__(module_name, fromlist=["__name__"])
        except Exception as error:
            rows.append((target, "importing it raised {}".format(describe(error))))
            continue
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        if missing:
            rows.append((target, (
                "it is installed but does not provide {}. Every name here is used at a real call "
                "site in src/, so this is a wheel whose API moved under the pinned code".format(
                    ", ".join(missing)))))
            continue
        rows.append((target, None))
    return rows


def check_facade_symbols():
    """(c) src.llm still re-exports both providers, and src.llm_base still declares the Protocol."""
    rows = []
    for module_name, symbols in FACADE_SYMBOLS:
        target = "{} still exports {}".format(module_name, ", ".join(symbols))
        try:
            module = __import__(module_name, fromlist=["__name__"])
        except Exception as error:
            rows.append((target, "importing it raised {}".format(describe(error))))
            continue
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        if missing:
            rows.append((target, (
                "it no longer exports {}. src/llm.py is the facade the rest of the app and the "
                "test suite import through, so a lost re-export breaks callers that never "
                "referenced the provider module directly".format(", ".join(missing)))))
            continue
        rows.append((target, None))
    return rows


def check_settings():
    """(d) get_settings() really returns the settings, with the documented defaults.

    This is the check that most repays running INSIDE the image, and it is also the most dangerous
    one to run, which is worth saying plainly: `get_settings()` reports a problem by calling
    `os._exit(1)`. There is no exception to catch — the interpreter is gone mid-report. So the
    failure mode of this check is not a FAIL line, it is the probe stopping dead, and the outer
    half's "the probe ran to its end" row is what turns that silence into a verdict. That is also
    exactly what makes the check worth having: on a VALID environment this function must not do
    that, and nothing else in this gate proves it.

    The environment is rewritten around the call rather than being taken as found, because the
    container is deliberately started with production's variables and the DEFAULTS are the thing
    being measured. It is restored in a `finally`, so the later checks still see the container's
    real environment.
    """
    rows = []
    call_target = "get_settings() returns a settings mapping on a valid environment"
    keys_target = "...containing every key the app indexes into ({})".format(
        ", ".join(EXPECTED_SETTINGS_KEYS))
    timezone_target = "...and caldav.timezone defaults to {!r}".format(EXPECTED_TIMEZONE)
    optional_target = "...and telegram_bot_api_server stays None when it is not set"

    try:
        from src.config import get_settings
    except Exception as error:
        reason = "src.config could not be imported: {}".format(describe(error))
        rows.append((call_target, reason))
        rows.append((keys_target, "not attempted: " + reason))
        for name, value in EXPECTED_DEFAULTS:
            rows.append(("...and {} defaults to {!r}".format(name, value), "not attempted"))
        rows.append((timezone_target, "not attempted"))
        rows.append((optional_target, "not attempted"))
        return rows

    saved = dict(os.environ)
    try:
        for name in OPTIONAL_ENV:
            os.environ.pop(name, None)
        os.environ.update(REQUIRED_ENV)
        settings = get_settings()
    except Exception as error:
        rows.append((call_target, (
            "it raised {}. On a valid environment this function is supposed to return a dict; "
            "anything else here is the image, not the configuration".format(describe(error)))))
        return rows
    finally:
        os.environ.clear()
        os.environ.update(saved)

    if not isinstance(settings, dict):
        rows.append((call_target, "it returned {}, not a dict".format(type(settings).__name__)))
        return rows
    rows.append((call_target, None))

    missing = [key for key in EXPECTED_SETTINGS_KEYS if key not in settings]
    if missing:
        rows.append((keys_target, (
            "it is missing {}. Most call sites index the dict directly rather than using .get(), "
            "so a missing key is a KeyError at the moment a user sends a message. Got: "
            "{!r}".format(", ".join(missing), sorted(settings)))))
    else:
        rows.append((keys_target, None))

    for name, expected in EXPECTED_DEFAULTS:
        target = "...and {} defaults to {!r}".format(name, expected)
        actual = settings.get(name)
        if actual == expected and type(actual) is type(expected):
            rows.append((target, None))
        else:
            rows.append((target, (
                "it is {!r} ({}). The deployment sets none of the optional variables, so this "
                "default IS the value production runs on — a drift here changes live behaviour "
                "with nothing in the stack having changed".format(actual, type(actual).__name__))))

    caldav_section = settings.get("caldav")
    if isinstance(caldav_section, dict) and caldav_section.get("timezone") == EXPECTED_TIMEZONE:
        rows.append((timezone_target, None))
    else:
        rows.append((timezone_target, (
            "it is {!r}. That value is written into the DTSTART/DTEND of every event this bot "
            "creates, so a drift books appointments in the wrong zone without failing "
            "anything".format(caldav_section))))

    if settings.get("telegram_bot_api_server") is None:
        rows.append((optional_target, None))
    else:
        rows.append((optional_target, (
            "it is {!r} even though the variable was removed from the environment. An optional "
            "setting that acquired a default would point every deployment that does not set it at "
            "an API server nobody chose".format(settings.get("telegram_bot_api_server")))))
    return rows


def check_user_manager():
    """(e) CalDAV credentials really round-trip through a file on disk.

    This is the service's only persistent state, and it is other people's calendar passwords: one
    plaintext JSON file per user under data/. A save that silently fails, or a read that silently
    returns None, means every user is told to run /caldav again — and src/users.py catches every
    exception and turns it into a `return False` plus a log line, so nothing crashes to announce it.

    Run in a temporary directory rather than against /app/data, and the chdir is not laziness:
    `UserManager.data_dir` is the relative string "data", so the only way to exercise the real code
    path against a scratch location is to move the working directory under it. Restored in a
    `finally`, because every later check depends on being in /app.
    """
    rows = []
    save_target = "UserManager.save_caldav_credentials() reports success"
    file_target = "...and leaves a non-empty file in data/"
    read_target = "...and get_caldav_credentials() returns exactly what was saved"
    has_target = "...and has_caldav_credentials() is True for that user and False for another"
    every_target = (save_target, file_target, read_target, has_target)

    try:
        from src.users import UserManager
    except Exception as error:
        reason = "src.users could not be imported: {}".format(describe(error))
        return [(target, reason) for target in every_target]

    directory = tempfile.mkdtemp(prefix="smoke-users-")
    previous_cwd = os.getcwd()
    try:
        os.chdir(directory)
        manager = UserManager()

        saved = manager.save_caldav_credentials(
            STORE_USER_ID,
            STORE_CREDENTIALS["username"],
            STORE_CREDENTIALS["password"],
            STORE_CREDENTIALS["url"],
            STORE_CREDENTIALS["calendar_name"])
        if saved is not True:
            rows.append((save_target, (
                "it returned {!r}. Every failure inside that method is caught and turned into "
                "False plus a log line, so this is as loud as a broken credential store ever "
                "gets".format(saved))))
            for target in every_target[1:]:
                rows.append((target, "not attempted: the save did not report success"))
            return rows
        rows.append((save_target, None))

        path = os.path.join(directory, "data", "user_{}.json".format(STORE_USER_ID))
        if not os.path.isfile(path):
            rows.append((file_target, (
                "no file at {!r}. The save reported success, so the write went somewhere else "
                "entirely — in production that is the mounted volume it missed".format(path))))
            for target in every_target[2:]:
                rows.append((target, "not attempted: nothing was written"))
            return rows
        size = os.path.getsize(path)
        if size == 0:
            rows.append((file_target, "the file exists and is 0 bytes"))
        else:
            rows.append(("{} ({} bytes)".format(file_target, size), None))

        read_back = manager.get_caldav_credentials(STORE_USER_ID)
        if read_back != STORE_CREDENTIALS:
            rows.append((read_target, (
                "it returned {!r}. The password deliberately contains non-ASCII characters and a "
                "quote, so a mismatch here is a JSON/encoding round trip that hands the user's "
                "calendar server a password that is almost right".format(read_back))))
        else:
            rows.append((read_target, None))

        present = manager.has_caldav_credentials(STORE_USER_ID)
        absent = manager.has_caldav_credentials(STORE_ABSENT_USER_ID)
        if present is not True or absent is not False:
            rows.append((has_target, (
                "it answered {!r} for the stored user and {!r} for an unknown one. src/bot.py "
                "gates every incoming message on this, so a wrong answer either locks a "
                "configured user out or sends an unconfigured one into the LLM path".format(
                    present, absent))))
        else:
            rows.append((has_target, None))
        return rows
    except Exception as error:
        rows.append(("the UserManager check ran to completion",
                     "it raised {}".format(describe(error))))
        return rows
    finally:
        os.chdir(previous_cwd)
        shutil.rmtree(directory, ignore_errors=True)


async def check_bot_construction():
    """(f) CalendarBot can actually be built, with every handler registered.

    The most valuable check in this file, and the one that repays explaining. `CalendarBot.__init__`
    touches no network at all — but it does a great deal of work that only succeeds if the image is
    right: `get_settings()` has to be satisfied, the token has to pass aiogram's format check, the
    custom `TelegramAPIServer` has to build, `get_llm()` has to construct the configured provider,
    `CalendarManager` and `UserManager` have to construct (which creates data/ under the real
    working directory), and eight handlers have to register. All of that is unreachable from
    outside a container whose real command dies at the first Telegram call.

    The handler count is taken from aiogram's own structures rather than from the source text, so a
    registration that silently stopped happening is a finding here — and a handler that stops
    registering is invisible in production: the command simply does nothing in the chat.

    Constructed from the container's own working directory on purpose, unlike the UserManager check
    above: that one needs a scratch directory to write into, this one wants the real /app so that
    the data/ directory it touches is the real mount point.
    """
    rows = []
    build_target = "CalendarBot() constructs without touching the network"
    handlers_target = "...and registers {} handlers ({})".format(
        sum(EXPECTED_HANDLERS.values()),
        ", ".join("{} {}".format(count, name) for name, count in sorted(EXPECTED_HANDLERS.items())))
    provider_target = "...and get_llm() built the provider LLM_PROVIDER asked for"
    wiring_target = "...and it holds a CalendarManager and a UserManager"
    every_target = (build_target, handlers_target, provider_target, wiring_target)

    try:
        from src.bot import CalendarBot
    except Exception as error:
        reason = "src.bot could not be imported: {}".format(describe(error))
        return [(target, reason) for target in every_target]

    bot = None
    try:
        try:
            bot = CalendarBot()
        except Exception as error:
            rows.append((build_target, (
                "it raised {}. Nothing in that constructor is allowed to need a peer, so this is "
                "the image: a wheel whose API moved, a token this aiogram will not accept, or a "
                "data directory it cannot create".format(describe(error)))))
            for target in every_target[1:]:
                rows.append((target, "not attempted: the constructor raised"))
            return rows
        rows.append((build_target, None))

        counts = count_handlers(bot.dp)
        if counts != EXPECTED_HANDLERS:
            rows.append((handlers_target, (
                "the dispatcher holds {!r} instead. Seven message handlers (/start, /google, "
                "/fastmail, /caldav, /stats, the photo filter, the catch-all) and one "
                "callback_query handler are supposed to be registered; a mismatch means either a "
                "handler stopped being registered — in which case that command silently does "
                "nothing in the chat — or one was added and this constant needs "
                "updating".format(counts))))
        else:
            rows.append((handlers_target, None))

        # The container runs with LLM_PROVIDER=groq, the value production runs with, so this also
        # proves config.py's provider table and llm.py's dispatch still agree with each other.
        provider_name = type(getattr(bot, "llm", None)).__name__
        if provider_name != "GroqLLM":
            rows.append((provider_target, (
                "it built a {} while LLM_PROVIDER is groq. src/llm.py falls back to DeepSeek for "
                "anything it does not recognise, so a broken mapping does not raise — it quietly "
                "bills the wrong API".format(provider_name))))
        else:
            rows.append((provider_target, None))

        calendar_name = type(getattr(bot, "calendar", None)).__name__
        users_name = type(getattr(bot, "user_manager", None)).__name__
        if calendar_name != "CalendarManager" or users_name != "UserManager":
            rows.append((wiring_target, (
                "it holds a {} and a {}. Those two are what every handler reaches through to "
                "reach a calendar and the credential store".format(calendar_name, users_name))))
        else:
            rows.append((wiring_target, None))
        return rows
    except Exception as error:
        rows.append(("the CalendarBot construction check ran to completion",
                     "it raised {}".format(describe(error))))
        return rows
    finally:
        if bot is not None:
            try:
                # aiogram opens no socket until a request is made, but it does allocate a session
                # object; closing it keeps the probe from printing an unclosed-session warning into
                # the middle of its own report.
                await bot.bot.session.close()
            except Exception:
                pass


async def run_check(name, function):
    """Run one check, converting an unexpected exception into a row instead of a dead probe.

    Without this a single unforeseen error would take the whole probe down and cost the run every
    other verdict, in exactly the situation where the rest of the report is what tells somebody
    which change did it. It cannot save the probe from `os._exit`, which is why check_settings()
    documents that hazard where it lives.

    The check is passed as a FUNCTION and called in here, not called at the call site and passed as
    a result. That distinction is the whole safety net for the synchronous checks: calling
    `check_src_imports()` in the argument list would run its body BEFORE this function is entered,
    so an exception from it would escape past the `except` below and take the probe down anyway.
    (An async check would survive either way, because calling it only builds a coroutine — which is
    exactly the kind of asymmetry that makes a net look like it works until it is needed.)
    """
    try:
        result = function()
        if hasattr(result, "__await__"):
            result = await result
        return result
    except Exception as error:
        return [("the {} check".format(name), "it raised {}".format(describe(error)))]


async def main():
    rows = []

    async def stage(name, function):
        produced = await run_check(name, function)
        # Printed here, as each group lands, rather than all at once at the end — see the note in
        # this module's docstring about os._exit.
        for target, reason in produced:
            if reason is None:
                print("ok   {}".format(target))
            else:
                print("FAIL {} -> {}".format(target, reason))
        rows.extend(produced)

    await stage("working directory", check_working_directory)
    await stage("baked .env", check_no_dotenv)
    await stage("data directory", check_data_dir)
    await stage("src imports", check_src_imports)
    await stage("third-party symbols", check_third_party_symbols)
    await stage("llm facade", check_facade_symbols)
    await stage("settings", check_settings)
    await stage("user manager", check_user_manager)
    await stage("bot construction", check_bot_construction)

    failures = [target for target, reason in rows if reason is not None]
    if failures:
        print("probe FAILED: {}/{} targets broken".format(len(failures), len(rows)))
        raise SystemExit(1)

    print("{}: {}/{} targets".format(PROBE_MARKER, len(rows), len(rows)))


if __name__ == "__main__":
    asyncio.run(main())
'''


def excerpt(text):
    """Bound what reaches the log, and say so when something was cut."""
    if text is None:
        return ""
    if len(text) <= EXCERPT_CHARS:
        return text
    return text[:EXCERPT_CHARS] + "\n[... truncated at {} characters]".format(EXCERPT_CHARS)


def docker(args, timeout, stdin_text=None):
    """Run a docker command.

    Returns (status, output) with stderr folded into stdout, because everything here is read by a
    human out of a CI log where the interleaving is the useful part. A status of None means the
    command produced no exit code at all — it timed out, or docker is not there — and `output` then
    explains which.
    """
    argv = ["docker"] + args
    try:
        completed = subprocess.run(
            argv,
            input=stdin_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True)
    except FileNotFoundError:
        return None, (
            "`docker` is not on PATH. This gate drives the daemon from the runner, so it cannot "
            "run anywhere the docker CLI is missing")
    except subprocess.TimeoutExpired as error:
        return None, "`{}` did not finish within {} s. Output so far:\n{}".format(
            " ".join(argv), timeout, excerpt(error.output))
    return completed.returncode, completed.stdout or ""


def remove_container(name):
    """Best effort. Never the reason a check fails; the workflow cleans up too."""
    docker(["rm", "-f", name], REMOVE_TIMEOUT)


def environment_flags(pairs):
    """A list of `VAR=value` strings as `-e VAR=value` pairs for a `docker run` argument list."""
    flags = []
    for pair in pairs:
        flags.extend(["-e", pair])
    return flags


def check_image_contract(image):
    """(a) What the image DECLARES: its command, where it runs, that nothing wraps them, and that
    its output is unbuffered."""
    rows = []
    target = "docker inspect {}".format(image)
    status, output = docker(
        ["inspect", "--format", "{{json .Config}}", image], INSPECT_TIMEOUT)
    if status is None:
        # No exit code at all: the docker CLI is missing or the call ran out its timeout. The
        # message from docker() already says which, and neither of them means what a non-zero exit
        # means, so they must not share its wording.
        rows.append((target, output))
        return rows
    if status != 0:
        rows.append((target, (
            "it exited {} — the tag does not exist on this daemon, so the build step and this step "
            "disagree about what was built. Output:\n{}".format(status, excerpt(output)))))
        return rows
    try:
        config = json.loads(output)
    except ValueError as error:
        rows.append((
            "parse the image config of {}".format(image),
            "docker inspect returned something that is not JSON ({}): {}".format(
                error, excerpt(output))))
        return rows

    cmd = config.get("Cmd")
    rows.append((
        "the image's CMD is {}".format(EXPECTED_CMD),
        None if cmd == EXPECTED_CMD else (
            "it is {!r}. Production runs the image's own command and so does check (g) below — a "
            "silent change here means the gate and the deployment are describing two different "
            "programs".format(cmd))))

    # An ENTRYPOINT would prepend itself to the CMD above, so `docker run <image>` would no longer
    # be `python main.py` and check (b)'s verdict would be about something else entirely. There is
    # none today, and this is what keeps it that way.
    entrypoint = config.get("Entrypoint")
    rows.append((
        "the image declares no ENTRYPOINT, so its CMD is the whole command line",
        None if not entrypoint else (
            "it declares {!r}, which is prepended to the CMD. Every `docker run` in this gate now "
            "executes something other than what it reports".format(entrypoint))))

    working_dir = config.get("WorkingDir")
    rows.append((
        "the image's WORKDIR is {}".format(APP_DIR),
        None if working_dir == APP_DIR else (
            "it is {!r}, and that is not cosmetic. UserManager.data_dir is the RELATIVE string "
            "'data', so it resolves against the working directory: move it and every user's CalDAV "
            "username and password are written outside {}, which is where production's volume is "
            "mounted. Nothing fails — the credentials are simply gone at the next "
            "redeploy".format(working_dir, DATA_DIR))))

    # The check that keeps check (b) honest. See the note on EXPECTED_ENV.
    declared_env = config.get("Env") or []
    rows.append((
        "the image declares {}".format(EXPECTED_ENV),
        None if EXPECTED_ENV in declared_env else (
            "it does not; Env is {!r}. src/config.py's startup guard writes one line and then "
            "calls os._exit(1), which skips interpreter shutdown and any flush that would happen "
            "there — so whether that line survives into `docker logs` rests on the internals of "
            "whichever logging library is installed. Check (b) below asserts on the TEXT of that "
            "line, and this variable is what makes the text depend on the image rather than on "
            "loguru's current flushing behaviour".format(declared_env))))

    return rows


def check_missing_env_guard(image, base_name):
    """(b) Run the image with three different broken environments: each must die, and say why.

    This is the check that cannot live inside a container, and the one that most directly describes
    a real deployment failure — the stack redeployed with a variable dropped out of it, or with a
    typo in one. The mechanism is `get_settings()` in src/config.py, called from
    `CalendarBot.__init__` before anything else happens: it logs one line and calls `os._exit(1)`.

    That it fires at all depends on the image carrying no `.env` file, which `load_dotenv()` would
    otherwise read; the probe checks for exactly that, and .dockerignore is what keeps it out.

    A non-zero exit on its own is NOT enough, and that is the whole reason the OUTPUT is matched
    too: a typo in an import, a missing wheel and a syntax error all exit non-zero, and every one
    of them would make an exit-code-only check pass while proving nothing about the guard. One case
    per branch, because src/config.py stops at the first failure — see the note on GUARD_CASES.

    The image's own CMD is kept — no override. That is safe to run to completion precisely because
    the guard makes it terminate on its own.

    `--rm` AND a name: the flag covers the container that exits, the name covers the one that does
    not. See the suffix note at the top of this file for why the second half is not redundant.
    """
    rows = []
    transcripts = []
    for case in GUARD_CASES:
        name = base_name + case["suffix"]
        target = "`docker run --rm {}` {}".format(image, case["label"])
        # A re-run from the UI reuses the run id, so it reuses this name too, and a previous attempt
        # killed by the step timeout may have left the container behind. A no-op when it did not.
        remove_container(name)
        status, output = docker(
            ["run", "--rm", "--name", name] + environment_flags(case["env"]) + [image],
            GUARD_TIMEOUT)
        transcripts.append(("the image run {}".format(case["label"]), output))

        if status is None:
            rows.append((target, output))
            rows.append(("...and its output names the problem ({!r})".format(case["fragment"]),
                         "not attempted: the run produced no exit code"))
            continue
        if status == 0:
            rows.append((target, (
                "it exited 0. The startup guard did not fire, so a stack redeployed in this state "
                "would come up looking perfectly healthy and fail at the first message a user "
                "sends instead. Output:\n{}".format(excerpt(output)))))
            rows.append(("...and its output names the problem ({!r})".format(case["fragment"]),
                         "not attempted: the container did not exit non-zero"))
            continue
        rows.append(("{} exits non-zero (got {})".format(target, status), None))

        rows.append((
            "...and its output names the problem ({!r})".format(case["fragment"]),
            None if case["fragment"] in output else (
                "nothing in the output mentions it. The image exited {}, but either this is not "
                "the guard firing — it is the app dying of something else before it got there — "
                "or {} stopped being checked. Either way an operator is left with an exit code and "
                "no reason. Output:\n{}".format(status, case["branch"], excerpt(output)))))
    return rows, transcripts


def start_probe_container(image, name):
    """Start the container the probe runs inside.

    NO PORT IS PUBLISHED — this service has no listening socket at all, so there is nothing to
    publish, and a job that runs inside act_runner's container while docker drives the host's daemon
    could not have reached one anyway.

    The image's CMD is REPLACED with a sleeping command, and that is deliberate rather than
    convenient: the real command reaches Telegram a few lines into startup, so a container started
    with it would be on its way out while the probe was still importing. Check (g) below starts a
    second container with the real command, which is where that question is answered.
    """
    rows = []
    target = "`docker run -d` starts a probe container of {}".format(image)
    remove_container(name)
    status, output = docker(
        ["run", "-d", "--name", name] + environment_flags(SMOKE_ENV) + [image] + IDLE_COMMAND,
        START_TIMEOUT)
    if status is None:
        rows.append((target, output))
        return rows, False, output
    if status != 0:
        rows.append((target, "`docker run -d` exited {}:\n{}".format(status, excerpt(output))))
        return rows, False, output
    rows.append((target, None))
    return rows, True, output


def probe_report_rows(output):
    """Parse the probe's own report lines back into rows of this gate's report.

    The probe prints exactly the same `ok   <target>` / `FAIL <target> -> <reason>` shape this file
    does, so its verdicts can be merged into one list instead of arriving as a single opaque "the
    probe failed". That matters for the same reason the guard check has one row per case: a run that
    breaks four things should say four things.

    The probe streams its rows, so loguru output from the code under test is interleaved with them
    in this stream. Matching on the two prefixes is what keeps that harmless: loguru's own lines
    start with a timestamp.
    """
    rows = []
    for line in output.splitlines():
        if line.startswith("ok   "):
            rows.append((PROBE_ROW_PREFIX + line[5:], None))
        elif line.startswith("FAIL "):
            target, _, reason = line[5:].partition(" -> ")
            rows.append((PROBE_ROW_PREFIX + target,
                         reason or "the probe reported FAIL with no reason"))
    return rows


def check_probe(name, started):
    """(c)-(f) Run the in-container probe and merge its verdicts into ours.

    `docker exec -i <name> python -u -` feeds the program over stdin: `-i` attaches stdin without
    asking for a tty (none is needed and none is available on a runner), the image's own interpreter
    runs it, and `docker exec` propagates the command's exit status — which is what lets the two
    consistency rows below tell "the probe reported failures" apart from "the probe died before it
    could report anything". The second of those is not hypothetical here: `get_settings()` ends a
    bad run with `os._exit(1)`, and the marker row is what catches it.
    """
    target = "the in-container probe's exit status agrees with its own report"
    marker_target = "the in-container probe ran to its end"
    if not started:
        return [("the in-container probe", "not attempted: the container never started")], ""

    status, output = docker(
        ["exec", "-i", name, "python", "-u", "-"], PROBE_TIMEOUT, stdin_text=PROBE)
    if status is None:
        # Timed out, or docker is missing. Either way there are no verdicts to merge.
        return [("the in-container probe", output)], output

    rows = probe_report_rows(output)
    if not rows:
        # Exit status alone cannot be read here: a probe that printed nothing has told us nothing,
        # whatever it exited with. This is what a `docker exec` that could not start the interpreter
        # looks like, and what a truncated stdin looks like.
        return [("the in-container probe", (
            "it produced no report lines at all, so none of the in-container checks ran. "
            "`docker exec` exited {}. Output:\n{}".format(status, excerpt(output))))], output

    reported_failures = any(reason is not None for _, reason in rows)
    if status != 0 and not reported_failures:
        rows.append((target, (
            "`docker exec` exited {} but every line the probe printed says ok — so it died after "
            "reporting and before finishing, and the report above is incomplete. The likeliest "
            "cause in this repo is get_settings() taking its os._exit(1) path".format(status))))
    elif status == 0 and reported_failures:
        rows.append((target, (
            "the probe printed FAIL lines yet exited 0. Its failures are supposed to leave through "
            "SystemExit(1); an exit of 0 here means they no longer do, and this gate would have "
            "gone green on them")))
    else:
        rows.append((target, None))

    rows.append((marker_target, None if PROBE_MARKER in output else (
        "it never printed {!r}. The marker is on the probe's last line, so its absence means the "
        "program did not run to the end — a truncated stdin, something that killed the interpreter "
        "mid-report, or get_settings() calling os._exit(1) on what should have been a valid "
        "environment".format(PROBE_MARKER))))
    return rows, output


def check_container_alive(name, started):
    """(h) The probe container is still running after everything above.

    It was started with a sleeping command, so there is exactly one correct answer. A container that
    is gone means something killed it — the daemon's OOM killer being the realistic one, on a runner
    shared with every other repository — and that would also mean the probe's verdicts above were
    collected from a process that was being torn down underneath it.
    """
    target = "the probe container is still running after every check above"
    if not started:
        return [(target, "not attempted: the container never started")]
    status, output = docker(
        ["inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", name], INSPECT_TIMEOUT)
    if status != 0:
        return [(target, "docker inspect exited {}: {}".format(status, excerpt(output)))]
    state = output.strip()
    return [(target, None if state.startswith("true") else (
        "docker reports `Running ExitCode` = {!r}. It was started with a command that does nothing "
        "but sleep, so it cannot have finished on its own".format(state)))]


def check_real_command(image, name):
    """(g) The image's real CMD gets through import and construction.

    This is the only check that runs the thing production runs, and its shape is dictated by what
    main.py actually does. In order: it constructs `CalendarBot()` — the startup guard, the aiogram
    Bot, the LLM provider, the calendar manager, the user manager and all eight handler
    registrations — and then awaits `start()`, whose FIRST statement logs STARTUP_MARKER.

    So the marker is a receipt for the whole of startup, and it is also the last thing that can be
    asserted on. `start()` continues straight into `_advertise_commands()` ->
    `bot.set_my_commands(...)`, which is a REQUEST to Telegram. This gate points
    TELEGRAM_BOT_API_SERVER at 127.0.0.1:1, so that request is refused instantly and the process
    dies with a traceback. That traceback is the EXPECTED outcome, which is why the traceback check
    below looks only at the part of the log BEFORE the marker — everything that could go wrong with
    the IMAGE has already happened by then, and everything after it is a network that was never
    meant to answer.

    Splitting the log at the marker is also what makes this check deterministic rather than a race
    against how fast the connection is refused. There is no polling window in which the container
    is "supposed to still be up": the question is only ever what it managed to log before it got to
    the network, and `docker logs` answers that just as happily for a container that has already
    exited.

    What is deliberately NOT a verdict is the container's fate after that call. Matching the exit
    code, or the text of the error aiogram raises, would tie this gate to a third party's wording
    and to a design decision that is legitimately allowed to change — a retry loop around start()
    would be a perfectly sensible improvement and would turn an exit-code check red for no reason.
    The container's final state IS printed into the transcript below, so the information is one
    scroll away when somebody needs it; it is just not a verdict.

    The container is started detached and WITHOUT `--rm`, so that its state can still be inspected
    after it has exited.
    """
    rows = []
    start_target = "`docker run -d` starts a container of {} with the image's own command".format(
        image)
    marker_target = "its log carries {!r} (bounded at {} s)".format(STARTUP_MARKER, STARTUP_BUDGET)
    guard_target = "...and no startup-guard message appears anywhere in the log"
    trace_target = "...and nothing before that line is a traceback"
    every_target = (marker_target, guard_target, trace_target)

    remove_container(name)
    status, output = docker(
        ["run", "-d", "--name", name] + environment_flags(SMOKE_ENV) + [image], START_TIMEOUT)
    if status is None:
        rows.append((start_target, output))
        for target in every_target:
            rows.append((target, "not attempted: the container could not be started"))
        return rows, ""
    if status != 0:
        rows.append((start_target, "`docker run -d` exited {}:\n{}".format(status, excerpt(output))))
        for target in every_target:
            rows.append((target, "not attempted: the container could not be started"))
        return rows, ""
    rows.append((start_target, None))

    # Polled rather than slept on, and bounded in wall clock: `docker logs` works on a container
    # that has already exited, so this reads the same log either way and does not care whether the
    # process is still alive by the time the marker is looked for.
    deadline = time.monotonic() + STARTUP_BUDGET
    logs = ""
    log_status = None
    while True:
        log_status, logs = docker(["logs", name], LOGS_TIMEOUT)
        if log_status == 0 and STARTUP_MARKER in logs:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(STARTUP_PAUSE)

    if log_status != 0:
        for target in every_target:
            rows.append((target, "`docker logs` exited {}: {}".format(log_status, excerpt(logs))))
    else:
        reached = STARTUP_MARKER in logs
        rows.append((marker_target, None if reached else (
            "it never appeared in {} s. That line is the first statement of CalendarBot.start(), "
            "so everything before it is construction: the settings guard, the aiogram Bot, the LLM "
            "provider, the calendar manager, the user manager and eight handler registrations. Its "
            "absence means the process did not get through one of those. Logs:\n{}".format(
                STARTUP_BUDGET, excerpt(logs)))))

        # Checked across the WHOLE log rather than only the part before the marker, and that is not
        # sloppiness: get_settings() runs inside the constructor, so the guard can only ever speak
        # before the marker. Matching everywhere is the same statement and it cannot be defeated by
        # a marker that failed to appear.
        fired = [case["fragment"] for case in GUARD_CASES if case["fragment"] in logs]
        rows.append((guard_target, None if not fired else (
            "it carries {!r}. The environment this container was given is complete and valid, so a "
            "guard firing on it means config.py now rejects something production sets — which is "
            "the same as saying production cannot start. Logs:\n{}".format(fired, excerpt(logs)))))

        if not reached:
            rows.append((trace_target, "not attempted: the startup line never appeared, so there "
                                       "is no 'before' to look at"))
        else:
            before = logs.split(STARTUP_MARKER, 1)[0]
            rows.append((trace_target, None if "Traceback (most recent call last)" not in before else (
                "there is one. Everything logged before that line is import and construction, so a "
                "traceback there is the image failing to come up — as distinct from the traceback "
                "AFTER it, which this gate expects: start() immediately calls set_my_commands() "
                "against the unreachable API server this check points it at. "
                "Before the marker:\n{}".format(excerpt(before)))))

    # Informational only, and deliberately not a verdict — see the docstring above. Printed so that
    # a human reading a failure has the container's fate in front of them without having to go and
    # ask the daemon for it.
    state_status, state = docker(
        ["inspect", "--format", "{{.State.Running}} {{.State.ExitCode}}", name], INSPECT_TIMEOUT)
    footer = "\n[container state after the checks above — `Running ExitCode` = {}]".format(
        state.strip() if state_status == 0 else "unavailable")
    return rows, (logs or "") + footer


def main():
    image = os.environ.get(IMAGE_ENV)
    if not image:
        print("{} is not set: this gate tests the image it is given and has no default, because a "
              "default would silently gate whichever image happened to be on the daemon".format(
                  IMAGE_ENV))
        raise SystemExit(1)
    name = os.environ.get(NAME_ENV)
    if not name:
        # No default here either, and this one is worth spelling out because a default would not
        # fail — it would work, quietly and wrongly, in two ways at once.
        # First, the cleanup: the workflow's `if: always()` step removes `<name>` and its four
        # suffixed siblings, and it builds those names from the SAME variable. A dropped or
        # misspelled `env:` entry would leave this script naming its containers after the fallback
        # while the cleanup step went looking for the run-id ones, found nothing, and swallowed the
        # miss in its `|| true` — the gate stays green and the leak is permanent.
        # Second, collisions: the runner runs one docker daemon for every repository in the fleet,
        # so a constant fallback means two concurrent runs pick the SAME name — and
        # remove_container() below would then delete the other run's live container out from under
        # it.
        print("{} is not set: this gate names every container it starts after it, and a default "
              "would both hide containers from the workflow's cleanup step and make two concurrent "
              "runs collide on one name".format(NAME_ENV))
        raise SystemExit(1)

    rows = []
    transcripts = []

    rows.extend(check_image_contract(image))

    guard_rows, guard_transcripts = check_missing_env_guard(image, name)
    transcripts.extend(guard_transcripts)
    rows.extend(guard_rows)

    try:
        probe_start_rows, started, start_output = start_probe_container(image, name)
        transcripts.append(("`docker run -d` of the probe container", start_output))
        rows.extend(probe_start_rows)

        probe_rows, probe_output = check_probe(name, started)
        transcripts.append(("the probe, from inside the container", probe_output))
        rows.extend(probe_rows)

        # After the probe on purpose: "the container is still up" is only worth anything once
        # everything else has had its turn at it.
        rows.extend(check_container_alive(name, started))

        cmd_rows, cmd_logs = check_real_command(image, name + CMD_SUFFIX)
        transcripts.append(("the image started with its own CMD", cmd_logs))
        rows.extend(cmd_rows)
    finally:
        # The workflow removes both again under `if: always()`, which covers the case where this
        # whole script is killed by the step timeout and never gets here. The three guard containers
        # were started with `--rm` and have exited by now, so they are not repeated here.
        remove_container(name + CMD_SUFFIX)
        remove_container(name)

    # The transcripts first, the verdicts last: in a CI log the verdicts are what somebody scrolls
    # to the bottom for, and the containers' own output is what turns a one-line verdict into a
    # diagnosis.
    for label, text in transcripts:
        print("")
        print("--- {} ---".format(label))
        print(excerpt(text).rstrip() or "(no output)")
    print("")
    print("--- results ---")

    failures = []
    for target, reason in rows:
        if reason is None:
            print("ok   {}".format(target))
        else:
            print("FAIL {} -> {}".format(target, reason))
            failures.append(target)

    if failures:
        print("")
        print("smoke FAILED: {}/{} targets broken:".format(len(failures), len(rows)))
        for target in failures:
            print("  - {}".format(target))
        raise SystemExit(1)

    print("")
    print("smoke ok: {}/{} targets".format(len(rows), len(rows)))


if __name__ == "__main__":
    main()
