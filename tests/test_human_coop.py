"""Real MDP integration, async policy boundaries, and WebSocket ownership."""
import asyncio
import contextlib
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lib" / "overcooked_ai"))

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld
from human_coop.policies import ChatPolicy, Decision, HTTPPolicy, RecipePolicy
from human_coop.server import create_app
from human_coop.session import Session


def make_session(order="boiled_egg", human_index=1, **kwargs):
    mdp = OvercookedGridworld.from_layout_name("new_env")
    mdp.start_order_list, mdp.order_probability = [order], {}
    return Session(OvercookedEnv(mdp, horizon=200), RecipePolicy(), human_index, **kwargs)


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_adapter_uses_public_state_and_server_side_key(self):
        received = {}
        async def endpoint(request):
            received["body"] = await request.json()
            received["authorization"] = request.headers.get("Authorization")
            return web.json_response({"choices": [{"message": {"content":
                                      '{"action":"interact","parameter":"egg"}'}}]})
        app = web.Application()
        app.router.add_post("/v1/chat/completions", endpoint)
        async with TestServer(app) as server, ClientSession() as client:
            observation = make_session().observation()
            policy = ChatPolicy(str(server.make_url("/v1")), "test-model", "synthetic-key")
            decision = await policy.decide(observation, client)
            self.assertEqual(decision, Decision("interact", "egg"))
            self.assertEqual(received["body"]["model"], "test-model")
            self.assertEqual(received["authorization"], "Bearer synthetic-key")
            self.assertEqual(json.loads(received["body"]["messages"][1]["content"]),
                             json.loads(json.dumps(observation)))
            self.assertNotIn("synthetic-key", json.dumps(observation))

    async def test_both_player_roles_complete_recipe_with_handoffs(self):
        # A scripted human supplies inputs through the same keyboard command
        # protocol. Each real episode must process and serve the requested food.
        for order in ("apple", "boiled_egg", "boiled_corn_slices", "baked_carrot_soup"):
            for human_index in (0, 1):
                with self.subTest(order=order, human_index=human_index):
                    session = make_session(order, human_index)
                    human = RecipePolicy()
                    session.command({"type": "start"})
                    for _ in range(180):
                        observation = session.observation()
                        human_observation = dict(observation, agent_index=human_index,
                                                 human_index=1 - human_index)
                        hd = await human.decide(human_observation)
                        ad = await session.policy.decide(observation)
                        session.pending_decision = (ad, session.episode_id, session.env.t)
                        session.command({"type": "input", "keys": [hd.action] if hd.action in
                                         ("up", "down", "left", "right") else [],
                                         "interaction": {"action": hd.action, "parameter": hd.parameter}
                                         if hd.action == "interact" else None})
                        with contextlib.redirect_stdout(io.StringIO()):
                            session.advance()
                        if session.done:
                            break
                    self.assertTrue(session.done)
                    self.assertEqual(session.score, 20)
                    self.assertEqual(session.env.state.order_list, [])
                    self.assertTrue(any("place_obj_on_counter()" in row["completed_actions"]
                                        for row in session.history))

    async def test_slow_endpoint_does_not_block_human_and_stale_action_is_discarded(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def endpoint(request):
            await request.json()
            entered.set()
            await release.wait()
            return web.json_response({"action": "down"})
        app = web.Application()
        app.router.add_post("/action", endpoint)
        async with TestServer(app) as server, ClientSession() as client:
            session = make_session(max_policy_lag=1)
            session.policy = HTTPPolicy(str(server.make_url("/action")))
            session.command({"type": "start"})
            task = asyncio.create_task(session.request_decision(client, session.observation()))
            await entered.wait()
            old = session.env.state.players[1].position
            session.command({"type": "input", "keys": ["up"]})
            with contextlib.redirect_stdout(io.StringIO()):
                session.advance()
                session.advance()
            self.assertNotEqual(session.env.state.players[1].position, old)
            self.assertFalse(task.done())
            release.set()
            await task
            self.assertIsNone(session.pending_decision)
            self.assertEqual(session.policy_status, "stale response discarded")

    async def test_reset_cancels_inflight_decision(self):
        session = make_session()
        entered = asyncio.Event()
        class Slow:
            async def decide(self, observation, client):
                entered.set()
                await asyncio.sleep(60)
                return Decision("down")
        session.policy = Slow()
        session.command({"type": "start"})
        task = asyncio.create_task(session.request_decision(None, session.observation()))
        session.policy_task = task
        await entered.wait()
        episode = session.episode_id
        session.reset()
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())
        self.assertNotEqual(episode, session.episode_id)
        self.assertEqual(session.env.t, 0)
        self.assertTrue(session.paused)
        self.assertIsNone(session.pending_decision)

    async def test_endpoint_timeout_falls_back_to_stay(self):
        release = asyncio.Event()
        async def endpoint(request):
            await release.wait()
            return web.json_response({"action": "down"})
        app = web.Application()
        app.router.add_post("/action", endpoint)
        async with TestServer(app) as server, ClientSession() as client:
            session = make_session()
            session.policy = HTTPPolicy(str(server.make_url("/action")), timeout=0.02)
            session.command({"type": "start"})
            await session.request_decision(client, session.observation())
            release.set()
            self.assertIn("Timeout", session.policy_status)
            self.assertIsNone(session.pending_decision)
            with contextlib.redirect_stdout(io.StringIO()):
                session.advance()
            self.assertEqual(session.history[-1]["actions"][session.agent_index], "stay")

    async def test_websocket_owner_spectator_origin_and_disconnect(self):
        session = make_session(tick_seconds=0.01)
        async with TestClient(TestServer(create_app(session))) as client:
            response = await client.get("/")
            self.assertEqual(response.status, 200)
            self.assertIn("Kitchen controls", await response.text())
            denied = await client.get("/ws", headers={"Origin": "https://elsewhere.invalid"})
            self.assertEqual(denied.status, 403)
            ws = await client.ws_connect("/ws")
            self.assertTrue((await ws.receive_json())["controller"])
            await ws.receive_json()
            spectator = await client.ws_connect("/ws")
            self.assertFalse((await spectator.receive_json())["controller"])
            await spectator.receive_json()
            await spectator.send_json({"type": "start"})
            while (await spectator.receive_json())["type"] != "error":
                pass
            self.assertTrue(session.paused)
            await ws.send_json({"type": "start"})
            for _ in range(20):
                if not session.paused:
                    break
                await asyncio.sleep(0.01)
            self.assertFalse(session.paused)
            await ws.close()
            for _ in range(20):
                if session.controller is None:
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(session.paused)
            self.assertEqual(session.keys, [])
            replacement = await client.ws_connect("/ws")
            self.assertTrue((await replacement.receive_json())["controller"])
            await replacement.close()
            await spectator.close()

    async def test_invalid_policy_response_is_safe_and_does_not_end_session(self):
        class Bad:
            async def decide(self, observation, client):
                return Decision("teleport")
        session = make_session()
        session.policy = Bad()
        session.command({"type": "start"})
        await session.request_decision(None, session.observation())
        self.assertEqual(session.policy_status, "policy error: ValueError")
        with contextlib.redirect_stdout(io.StringIO()):
            session.advance()
        self.assertEqual(session.env.t, 1)

    async def test_logs_preserve_public_transitions_and_episode_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            session = make_session(log_dir=directory)
            episode = session.episode_id
            session.command({"type": "start"})
            with contextlib.redirect_stdout(io.StringIO()):
                session.advance()
            session.reset()
            session.log.close()
            records = [json.loads(line) for line in (Path(directory) / (episode + ".jsonl")).read_text().splitlines()]
            self.assertEqual(records[0]["type"], "episode")
            transition = records[1]
            self.assertEqual(transition["before"]["state"]["timestep"], 0)
            self.assertEqual(transition["after"]["state"]["timestep"], 1)
            self.assertNotIn("communicate_history", transition["before"]["state"])


class InputTests(unittest.TestCase):
    def test_input_lease_and_one_shot_agent_action(self):
        session = make_session()
        session.command({"type": "start"})
        session.command({"type": "input", "keys": ["up"]})
        session.input_time = time.monotonic() - 1
        session.pending_decision = (Decision("down"), session.episode_id, 0)
        with contextlib.redirect_stdout(io.StringIO()):
            session.advance()
            session.advance()
        self.assertEqual(session.env.state.players[1].position, (1, 2))
        self.assertEqual(session.history[0]["actions"][0], "down")
        self.assertEqual(session.history[1]["actions"][0], "stay")

    def test_dispenser_cannot_create_finished_food(self):
        session = make_session()
        player = session.env.state.players[1]
        player.position, player.orientation = (1, 1), (-1, 0)
        self.assertEqual(session.safe_decision(1, Decision("interact", "boiled_egg")), Decision())
        self.assertEqual(session.safe_decision(1, Decision("interact", "egg")), Decision("interact", "egg"))

    def test_unstarted_and_cooking_appliance_interactions_are_safe(self):
        session = make_session(human_index=0)
        player = session.env.state.players[0]
        player.position, player.orientation = (3, 1), (0, -1)
        session.env.state.objects[(3, 0)] = ObjectState("soup", "semi-finished", (3, 0), (["egg"], 1, 0))
        self.assertEqual(session.safe_decision(0, Decision("interact")), Decision())
        self.assertEqual(session.safe_decision(0, Decision("interact", "[START]")), Decision("interact", "[START]"))
        session.env.state.objects[(3, 0)].state = ("boiled_egg", 1, 1)
        player.set_object(ObjectState("egg", "ingredient", player.position))
        self.assertEqual(session.safe_decision(0, Decision("interact")), Decision())


if __name__ == "__main__":
    unittest.main()
