"""Primitive-action policy interface; no imports from the benchmark LLM stack."""
from collections import Counter, deque
from dataclasses import dataclass
import json

import aiohttp


DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
ACTIONS = dict(DIRECTIONS, stay=(0, 0), interact="interact")


@dataclass(frozen=True)
class Decision:
    action: str = "stay"
    parameter: str = None

    @classmethod
    def parse(cls, value, ingredients):
        if not isinstance(value, dict) or set(value) - {"action", "parameter"}:
            raise ValueError("Expected {action, parameter?}")
        action, parameter = value.get("action"), value.get("parameter")
        if not isinstance(action, str) or action not in ACTIONS:
            raise ValueError("Unknown primitive action")
        if parameter is not None and (not isinstance(parameter, str) or
                                      parameter not in ingredients + ["[START]"]):
            raise ValueError("Unknown interaction parameter")
        if action != "interact" and parameter is not None:
            raise ValueError("Only interact accepts a parameter")
        return cls(action, parameter)


class HTTPPolicy:
    """POST a public observation to an external policy; return one action."""
    def __init__(self, url, timeout=10):
        self.url, self.timeout = url, timeout

    async def decide(self, observation, client):
        async with client.post(self.url, json=observation,
                               timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
            response.raise_for_status()
            return Decision.parse(await response.json(), observation["ingredients"])


class ChatPolicy:
    """Adapter for a local or remote chat-completions-compatible model server.

    Model output is data: only a validated primitive action is accepted, never
    Python code. Credentials remain on the game server.
    """
    PROMPT = (
        "You control one chef in Collab-Overcooked with a human teammate. "
        "Cooperate using public game actions only; there is no chat. Return only "
        'a JSON object: {"action":"up|down|left|right|stay|interact",'
        '"parameter":null}. Choose exactly one primitive action for agent_index. '
        "Direction actions also turn the chef toward an adjacent station. "
        "For interact at I, parameter must be a raw_ingredients name. "
        'To start a loaded appliance with empty hands use parameter "[START]". '
        "Otherwise parameter is null. X is a shared counter, D dispenses dishes, "
        "S serves the current order; other stations are described in utensils. "
        "Pass items using shared counters. Do not duplicate food that is already "
        "held by the teammate or being processed. A held dish collects a finished "
        "recipe; some orders require a dish. Infer useful cooperation from the "
        "public state and past actions. Never invent an ingredient or teleport."
    )

    def __init__(self, base_url, model, api_key=None, timeout=10):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout

    async def decide(self, observation, client):
        headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
        payload = {"model": self.model, "temperature": 0, "max_tokens": 120,
                   "messages": [{"role": "system", "content": self.PROMPT},
                                {"role": "user", "content": json.dumps(observation)}]}
        async with client.post(self.url, json=payload, headers=headers,
                               timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
            response.raise_for_status()
            value = await response.json()
            content = value["choices"][0]["message"]["content"]
            return Decision.parse(json.loads(content), observation["ingredients"])


class RecipePolicy:
    """Small deterministic teammate, following dependencies and shared counters.

    This is a playable baseline, not an intention-inference model. The policy
    only reads the same public observation made available to external policies.
    """
    async def decide(self, observation, client=None):
        self.o = observation
        self.index = observation["agent_index"]
        self.me = observation["state"]["players"][self.index]
        self.other = observation["state"]["players"][1 - self.index]
        self.objects = observation["state"]["objects"]
        self.grid = observation["terrain"]
        self.floor = {(x, y) for y, row in enumerate(self.grid)
                      for x, tile in enumerate(row) if tile == " "}
        self.reachable = self._reachable(tuple(self.me["position"]))
        self.recipes = {name: (kind, recipe) for kind, recipes in observation["recipes"].items()
                        for name, recipe in recipes.items() if kind != "none"}
        orders = observation["state"]["order_list"]
        if not orders:
            return Decision()
        return self._ensure(orders[0], None, set())

    def _reachable(self, start):
        found, queue = {start}, deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS.values():
                p = x + dx, y + dy
                if p in self.floor and p not in found:
                    found.add(p)
                    queue.append(p)
        return found

    def _positions(self, tile):
        return [(x, y) for y, row in enumerate(self.grid)
                for x, value in enumerate(row) if value == tile]

    def _accessible(self, pos):
        return any((pos[0] - dx, pos[1] - dy) in self.reachable
                   for dx, dy in DIRECTIONS.values())

    def _go(self, targets, parameter=None):
        start = tuple(self.me["position"])
        blocked = tuple(self.other["position"])
        queue, visited = deque([(start, None)]), {start}
        while queue:
            pos, first = queue.popleft()
            for target in targets:
                direction = target[0] - pos[0], target[1] - pos[1]
                if direction in DIRECTIONS.values():
                    if first:
                        return Decision(first)
                    if tuple(self.me["orientation"]) == direction:
                        return Decision("interact", parameter)
                    return Decision(next(k for k, v in DIRECTIONS.items() if v == direction))
            for name, (dx, dy) in DIRECTIONS.items():
                p = pos[0] + dx, pos[1] + dy
                if p in self.floor and p != blocked and p not in visited:
                    visited.add(p)
                    queue.append((p, first or name))
        return Decision()

    def _handoff(self):
        other_reach = self._reachable(tuple(self.other["position"]))
        occupied = {tuple(obj["position"]) for obj in self.objects}
        counters = [p for p in self._positions("X") if p not in occupied and
                    self._accessible(p) and any((p[0] - dx, p[1] - dy) in other_reach
                                               for dx, dy in DIRECTIONS.values())]
        return self._go(counters)

    def _deliver_to(self, destination):
        if destination is None:
            targets = [p for p in self._positions("S") if self._accessible(p)]
        else:
            kind = self.recipes[destination][0]
            targets = [p for p in self._positions(self.o["utensils"][kind]["symbol"])
                       if self._accessible(p)]
        return self._go(targets) if targets else self._handoff()

    def _ensure(self, item, destination, seen):
        if item in seen:
            return Decision()
        seen = seen | {item}
        held, other_held = self.me["held_object"], self.other["held_object"]
        if held and held["name"] == item:
            return self._deliver_to(destination)
        if other_held and other_held["name"] == item:
            return Decision()
        existing = [obj for obj in self.objects if obj["name"] == item]
        if existing:
            obj = next((obj for obj in existing if self._accessible(obj["position"])), None)
            if obj is None:
                if destination is None and self.o["need_dish"].get(item):
                    return self._supply_dish()
                return Decision()
            on_counter = self.grid[obj["position"][1]][obj["position"][0]] == "X"
            if self.o["need_dish"].get(item) and not on_counter and destination is None:
                if not held:
                    return self._ensure_dish()
                if held["name"] != "dish":
                    return self._handoff()
            elif held:
                return self._handoff()
            return self._go([obj["position"]])
        if item not in self.recipes:
            if held:
                return self._handoff()
            return self._go(self._positions("I"), item)
        kind, recipe = self.recipes[item]
        stations = self._positions(self.o["utensils"][kind]["symbol"])
        # Choose the existing compatible batch, otherwise an empty station.
        batches = {tuple(obj["position"]): obj for obj in self.objects}
        station = next((p for p in stations if tuple(p) not in batches or
                        (batches[tuple(p)]["state"] and
                         (batches[tuple(p)]["state"][0] == item or
                          isinstance(batches[tuple(p)]["state"][0], list) and
                          not (Counter(batches[tuple(p)]["state"][0]) -
                               Counter(recipe["recipe"]))))), None)
        if station is None:
            return Decision()
        obj = batches.get(tuple(station))
        if obj and obj["state"][0] == item:
            # A batch is already cooking; do not start a duplicate.
            if not self._accessible(station) and self.o["need_dish"].get(item):
                return self._supply_dish()
            return Decision()
        contents = Counter(obj["state"][0]) if obj else Counter()
        missing = list((Counter(recipe["recipe"]) - contents).elements())
        if not missing:
            return self._go([station], "[START]") if self._accessible(station) else Decision()
        for child in missing:
            result = self._ensure(child, item, seen)
            if result.action != "stay":
                return result
        return Decision()

    def _ensure_dish(self):
        dishes = [obj["position"] for obj in self.objects if obj["name"] == "dish"]
        return self._go(dishes) if dishes else self._go(self._positions("D"))

    def _supply_dish(self):
        held = self.me["held_object"]
        if held and held["name"] == "dish":
            return self._handoff()
        if held or any(obj["name"] == "dish" for obj in self.objects) or (
                self.other["held_object"] and self.other["held_object"]["name"] == "dish"):
            return Decision()
        return self._ensure_dish()
