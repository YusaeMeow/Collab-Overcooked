# Human-agent browser play

This optional entry point adds real-time keyboard play against the same Python
MDP used by Collab-Overcooked. One human and one agent share a game; the first
browser connection controls the human, and additional tabs are spectators.
There is no human-agent chat or hidden human-goal input.

## Setup and launch

From the repository root, preferably in a dedicated Python environment:

```bash
pip install -r requirements-human.txt
pip install -e lib/overcooked_ai
python src/human_play.py --order boiled_egg --human-index 1 --log-dir play-logs
```

Open http://127.0.0.1:8080. Start the game and click the kitchen to focus the
keyboard. The human can use either chef (`--human-index 0` or `1`). P0 is on the
right with the pot, oven, and serving station; P1 is on the left with the
ingredient dispenser, dishes, chopping board, and blender.

| Control | Operation |
| --- | --- |
| WASD / arrows | Move; facing a blocked station still turns the chef |
| Space / Interact | Take, put down, collect ready food, or serve |
| E / Start appliance | Start a loaded appliance with empty hands |
| Ingredient selector | Raw ingredient taken from the ingredient dispenser |
| Start / Resume, Pause | Control the shared game clock |
| New game | Cancel the current request, clear actions, and start a paused episode |

For the default boiled-egg task as P1: select egg, move up and face left toward
the dispenser, press Space, move down and face right toward the shared counter,
then press Space to pass the egg. The recipe agent on the right cooks and serves
it. With P0 controlled by the human, the recipe agent passes ingredients from
the left; the human operates the pot and serves the order.

The browser also has pointer movement buttons. Leaving the browser window
releases movement, a 0.5-second input lease prevents stuck keys, and disconnecting
the controller pauses the game. A new connection can take control after the
previous controller disconnects. A spectator tab does not automatically promote
itself; reload it to acquire a newly available control slot.

Sessions contain one requested order. Completing it ends the episode; there is
no automatic replenishment. `--horizon` counts simulation ticks (default 1200),
and `--tick-seconds` sets their period (default 0.2). Paused time does not advance
the simulation. The server listens on loopback by default. For an SSH-hosted
game, forward the port, for example `ssh -L 8080:127.0.0.1:8080 your-host`.

## Teammate options

### Recipe baseline (default)

`RecipePolicy` follows recipe dependencies, picks reachable stations, and passes
supplies on shared counters. It reads only public observations. It is a small
deterministic baseline, not a learned human-intent model or an optimal planner.
It may wait when the human holds a needed item or an appliance is occupied; it
does not recover arbitrary incorrect batches, cluttered counters, or every
possible recipe combination. Four representative workflows are exercised in
both player roles by the integration tests, including a plated recipe.

### Chat-completions-compatible model server

```bash
python src/human_play.py --order boiled_egg \
  --agent-model YOUR_MODEL_ID --agent-base-url http://127.0.0.1:8000/v1 \
  --agent-timeout 10 --max-policy-lag 15
```

If the server needs a key, set the `AGENT_API_KEY` environment variable, or use
`--agent-key-env` to select another variable. Keys stay on the game server and
are not included in browser state, prompts, or trajectory logs. The adapter
requests a JSON primitive action from `/chat/completions`. It does not use the
benchmark's paired `LLMAgents` communication mechanism or require its model
downloads. Model quality and latency depend on the chosen endpoint; no live
commercial-model calls are part of the tests.

Primitive actions are intentionally small and validated. This is an integration
baseline, not a claim that a slow LLM can play well at 5 Hz. `max-policy-lag`
allows an explicit tradeoff between response age and rejected actions. A model
can still select a valid action that becomes strategically inappropriate.

### Custom HTTP agent

```bash
python src/human_play.py --agent-url http://127.0.0.1:9000/action
```

The game POSTs one public observation to this endpoint:

```json
{
  "protocol_version": 1,
  "episode_id": "unique episode identifier",
  "agent_index": 0,
  "human_index": 1,
  "state": {
    "players": [], "objects": [],
    "order_list": ["boiled_egg"], "timestep": 12
  },
  "terrain": [], "ingredients": [], "raw_ingredients": [],
  "recipes": {}, "utensils": {}, "need_dish": {}, "history": []
}
```

Arrays above are abbreviated; actual observations contain both players,
terrain, objects with category/state/position, and up to 20 past transitions.
History contains attempted and applied primitive actions, interaction parameters,
completed game actions, and reward. It does not contain future actions, private
key state, intent labels, or text messages. Each request is a deep-copy snapshot.

The response must be a JSON object, for example:

```json
{"action": "interact", "parameter": "egg"}
```

Allowed actions: `up`, `down`, `left`, `right`, `stay`, `interact`.
`parameter` can be omitted/null, an ingredient/item name from the observation,
or `[START]`. Only `interact` accepts a non-null parameter. At the ingredient
dispenser only raw ingredients are permitted; finished products cannot be
created there. Other parameter/state combinations are validated before entering
the legacy MDP. No model output is evaluated as code.

Only one policy request is in flight. Human actions and the environment clock
continue while it runs; the agent stays until an action is available. Each
accepted primitive action executes once. Responses older than
`--max-policy-lag` ticks (default 5) are discarded, with an execution-time age
check as well. Timeouts and invalid responses result in stay, not termination.
Pause/reset cancels requests and clears queued actions. Endpoint-side computation
may continue after cancellation, but its result cannot be applied to a new game.

## Recording and evaluation boundaries

`--log-dir` writes a new JSONL file for each episode. Records include initial
public state, pre/post-transition snapshots, attempted and applied inputs,
reward, policy response age, and whether a response was accepted. Files are
flushed as records are written. A reset creates a separate episode identifier.

These trajectories are not the benchmark's existing `statistics_dict` format.
Do not feed them into `evaluation.py` without an explicit conversion and metric
definition. Keyboard inputs are behavior evidence, not ground-truth intentions.
The original benchmark runner and paired-agent logs are unchanged. The only
MDP adjustment treats an empty/nonpositive order probability distribution as
disabled replenishment instead of crashing on `random.choices([])`.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests run against the real MDP and local mock HTTP/WebSocket servers. They cover
both chef roles, shared-counter handoffs, raw/cooked/multi-stage/plated orders,
slow-policy independence, stale actions, reset cancellation, malformed model
outputs, input leases, trajectory boundaries, spectator ownership, cross-origin
control rejection, and disconnect pausing. No API credentials are required.

For a browser smoke test, install Playwright separately and run the optional
`tests/browser_smoke.cjs` against an already running default game. The script
uses actual keys to supply an egg and waits for the recipe teammate to serve it.
Set `BROWSER_CHANNEL` if using an installed browser instead of Playwright's
Chromium. Screenshots are written only when `SCREENSHOT_PATH` is set.
