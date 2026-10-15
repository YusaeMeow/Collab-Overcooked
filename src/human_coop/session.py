"""One authoritative game clock; policy calls never own or mutate the MDP."""
import asyncio
import copy
import json
import time
import uuid
from collections import deque
from pathlib import Path

import aiohttp

from .policies import ACTIONS, Decision


def object_dict(obj):
    if obj is None:
        return None
    return dict(obj.to_dict(), category=obj.catagory)


class Session:
    def __init__(self, env, policy, human_index=1, tick_seconds=0.2,
                 max_policy_lag=5, log_dir=None):
        self.env, self.policy = env, policy
        self.human_index, self.agent_index = human_index, 1 - human_index
        self.tick_seconds, self.max_policy_lag = tick_seconds, max_policy_lag
        self.clients, self.controller = set(), None
        self.log_dir = Path(log_dir) if log_dir else None
        self.log = None
        self.history = deque(maxlen=20)
        self.policy_task = None
        self.reset()

    def reset(self):
        if self.policy_task:
            self.policy_task.cancel()
            self.policy_task = None
        if self.log:
            self.log.close()
        self.env.reset()
        self.episode_id = uuid.uuid4().hex
        self.score, self.paused, self.done = 0, True, False
        self.keys, self.input_time, self.interaction = [], 0, None
        self.pending_decision, self.policy_status, self.input_notice = None, "idle", ""
        self.history.clear()
        self.log = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log = (self.log_dir / (self.episode_id + ".jsonl")).open("x", encoding="utf-8")
            self.record({"type": "episode", "observation": self.observation(),
                         "tick_seconds": self.tick_seconds})

    def record(self, data):
        if self.log:
            self.log.write(json.dumps(data) + "\n")
            self.log.flush()

    def observation(self):
        mdp, state = self.env.mdp, self.env.state
        players = [dict(player.to_dict(), held_object=object_dict(player.held_object))
                   for player in state.players]
        return copy.deepcopy({
            "protocol_version": 1, "episode_id": self.episode_id,
            "agent_index": self.agent_index, "human_index": self.human_index,
            "state": {"players": players, "objects": [object_dict(obj) for obj in state.objects.values()],
                      "order_list": state.order_list, "timestep": self.env.t},
            "terrain": mdp.terrain_mtx, "ingredients": list(dict.fromkeys(mdp.all_ingredients)),
            "raw_ingredients": mdp.default_ingredients,
            "recipes": mdp.recipes, "utensils": mdp.utensils, "need_dish": mdp.need_dish,
            "history": list(self.history),
        })

    def view(self):
        return dict(self.observation(), score=self.score, paused=self.paused,
                    done=self.done, horizon=self.env.horizon, policy_status=self.policy_status,
                    errors=list(self.env.state.error_message), input_notice=self.input_notice)

    def clear_input(self):
        self.keys, self.interaction = [], None

    def command(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected an object")
        kind = data.get("type")
        if kind == "input":
            keys = data.get("keys", [])
            if not isinstance(keys, list) or len(keys) > 4 or any(
                    not isinstance(key, str) or key not in ("up", "down", "left", "right") for key in keys):
                raise ValueError("Invalid keys")
            interaction = data.get("interaction")
            if interaction is not None:
                interaction = Decision.parse(interaction, self.env.mdp.all_ingredients)
                if interaction.action != "interact":
                    raise ValueError("Expected an interaction")
            self.keys, self.input_time = keys, time.monotonic()
            if interaction:
                self.interaction = interaction
        elif kind == "start":
            if not self.done:
                self.clear_input()
                self.paused = False
        elif kind == "pause":
            self.paused = True
            self.clear_input()
            self.pending_decision = None
            if self.policy_task:
                self.policy_task.cancel()
                self.policy_task = None
        elif kind == "reset":
            self.reset()
        else:
            raise ValueError("Unknown command")

    async def request_decision(self, client, observation):
        started = time.monotonic()
        episode, tick = observation["episode_id"], observation["state"]["timestep"]
        try:
            decision = await self.policy.decide(observation, client)
            decision = Decision.parse({"action": decision.action, "parameter": decision.parameter},
                                      self.env.mdp.all_ingredients)
            lag = self.env.t - tick
            if episode != self.episode_id or self.paused or self.done:
                return
            accepted = lag <= self.max_policy_lag
            self.pending_decision = (decision, episode, tick) if accepted else None
            self.policy_status = "ready" if accepted else "stale response discarded"
            self.record({"type": "policy", "observed_tick": tick, "available_tick": self.env.t,
                         "latency_seconds": time.monotonic() - started, "accepted": accepted,
                         "decision": {"action": decision.action, "parameter": decision.parameter}})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.pending_decision = None
            # Do not expose endpoint headers, API response bodies, or credentials.
            self.policy_status = "policy error: " + type(exc).__name__
            self.record({"type": "policy_error", "observed_tick": tick,
                         "error_type": type(exc).__name__})

    def safe_decision(self, index, decision):
        """Reject unsafe parameter/state combinations before the legacy MDP.

        Its benchmark assumes validated macro actions; arbitrary keyboard or
        endpoint input must not dispense cooked products or append to a cooking
        batch. No changes to the benchmark transition rules are required.
        """
        if decision.action != "interact":
            return decision
        mdp, state = self.env.mdp, self.env.state
        player = state.players[index]
        x, y = player.position
        dx, dy = player.orientation
        pos = x + dx, y + dy
        tile = mdp.get_terrain_type_at_pos(pos)
        obj, held = state.objects.get(pos), player.held_object
        parameter = decision.parameter
        reason = None
        if tile == "I":
            if parameter not in mdp.default_ingredients:
                reason = "Select a raw ingredient before using the dispenser."
        elif tile in mdp.interactive_terrain:
            if parameter == "[START]":
                if held or not obj or obj.catagory != "semi-finished" or obj.state[2] != 0:
                    reason = "Load the appliance and empty your hands before starting it."
            elif held:
                if held.name == "dish" and (not obj or obj.catagory != "finished"):
                    reason = "A dish can collect food only after it is ready."
                elif held.name != "dish" and obj and not isinstance(obj.state[0], list):
                    reason = "This appliance is already processing a recipe."
            elif obj and obj.catagory == "semi-finished":
                # Unprocessed raw ingredients have no recipe cook_time entry.
                reason = "Press E to start a loaded appliance; wait until food is ready."
        else:
            parameter = None
        if reason:
            if index == self.human_index:
                self.input_notice = reason
            return Decision()
        return Decision("interact", parameter)

    def advance(self):
        if self.paused or self.done:
            return
        human = Decision()
        if time.monotonic() - self.input_time <= 0.5:
            human = self.interaction or Decision(self.keys[-1] if self.keys else "stay")
        self.interaction = None
        agent = Decision()
        if self.pending_decision:
            decision, episode, tick = self.pending_decision
            if episode == self.episode_id and self.env.t - tick <= self.max_policy_lag:
                agent = decision
        self.pending_decision = None  # A primitive decision is executed exactly once.
        self.input_notice = ""
        decisions = [None, None]
        decisions[self.human_index], decisions[self.agent_index] = human, agent
        attempted = [{"action": d.action, "parameter": d.parameter} for d in decisions]
        decisions = [self.safe_decision(i, d) for i, d in enumerate(decisions)]
        before = self.observation()
        _, reward, done, _ = self.env.step(tuple(ACTIONS[d.action] for d in decisions),
                                         [d.parameter for d in decisions])
        self.score += reward
        self.done = done
        event = {"timestep": self.env.t, "actions": [d.action for d in decisions],
                 "parameters": [d.parameter for d in decisions],
                 "completed_actions": list(self.env.state.ml_actions), "reward": reward,
                 "attempted": attempted}
        self.history.append(event)
        self.record({"type": "transition", "before": before, "event": event,
                     "after": self.observation(), "done": done})
        if done:
            self.clear_input()

    async def broadcast(self):
        if self.clients:
            await asyncio.gather(*(client.send_json(dict(self.view(), type="state"))
                                   for client in tuple(self.clients) if not client.closed),
                                 return_exceptions=True)

    async def run(self):
        async with aiohttp.ClientSession() as client:
            try:
                while True:
                    started = time.monotonic()
                    self.advance()
                    if not self.paused and not self.done and (
                            self.policy_task is None or self.policy_task.done()):
                        self.policy_status = "thinking"
                        self.policy_task = asyncio.create_task(self.request_decision(client, self.observation()))
                    await self.broadcast()
                    await asyncio.sleep(max(0, self.tick_seconds - (time.monotonic() - started)))
            finally:
                if self.policy_task:
                    self.policy_task.cancel()
                    await asyncio.gather(self.policy_task, return_exceptions=True)
                if self.log:
                    self.log.close()
