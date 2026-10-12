"""Standalone human-agent entry point; does not import collab.modules."""
from argparse import ArgumentParser
import os
from aiohttp import web
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from human_coop.policies import ChatPolicy, HTTPPolicy, RecipePolicy
from human_coop.server import create_app
from human_coop.session import Session


def main():
    parser = ArgumentParser(description="Play Collab-Overcooked with an agent teammate")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--human-index", type=int, choices=(0, 1), default=1)
    parser.add_argument("--order", default="boiled_egg")
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--tick-seconds", type=float, default=0.2)
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument("--agent-url", help="HTTP policy endpoint; default is a recipe teammate")
    policy_group.add_argument("--agent-model", help="Model ID on a chat-completions-compatible server")
    parser.add_argument("--agent-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--agent-key-env", default="AGENT_API_KEY", help="Environment variable holding the model API key")
    parser.add_argument("--agent-timeout", type=float, default=10)
    parser.add_argument("--max-policy-lag", type=int, default=5, help="Maximum response age in game ticks")
    parser.add_argument("--log-dir", help="Optional per-episode JSONL trajectory directory")
    args = parser.parse_args()
    if args.horizon <= 0 or args.tick_seconds <= 0 or args.agent_timeout <= 0 or args.max_policy_lag < 0:
        parser.error("Time limits must be positive; max-policy-lag must be nonnegative")
    mdp = OvercookedGridworld.from_layout_name("new_env")
    if args.order not in mdp.Recipe.recipe_name:
        parser.error("Unknown order: " + args.order)
    mdp.start_order_list = [args.order]
    # Disable replenishment for a single-order play session, including raw orders.
    mdp.order_probability = {}
    policy = RecipePolicy()
    if args.agent_url:
        policy = HTTPPolicy(args.agent_url, args.agent_timeout)
    elif args.agent_model:
        policy = ChatPolicy(args.agent_base_url, args.agent_model,
                            os.environ.get(args.agent_key_env), args.agent_timeout)
    session = Session(OvercookedEnv(mdp, horizon=args.horizon), policy,
                      args.human_index, args.tick_seconds, args.max_policy_lag, args.log_dir)
    web.run_app(create_app(session), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
