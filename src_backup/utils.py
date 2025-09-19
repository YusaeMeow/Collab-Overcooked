import numpy as np
import os

from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.agents.agent import GreedyHumanModel, StayAgent, RandomAgent
from overcooked_ai_py.agents.agent import AgentFromPolicy, AgentPair
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.utils import load_dict_from_file, load_pickle


from collab.collab import LLMAgents

from collections import defaultdict
from collab.modules import EMBEDDING_MODEL


def make_agent(alg: str, mdp, layout, **gptargs):

    if alg == "Stay":
        agent = StayAgent()

    elif alg == "Random":
        agent = RandomAgent()

    elif alg == "LLMPair" or alg == "Greedy":
        MLAM_PARAMS = {
            "start_orientations": False,
            "wait_allowed": True,
            "counter_goals": [],
            "counter_drop": [],
            "counter_pickup": [],
            "same_motion_goals": True,
        }
        counter_locations = mdp.get_counter_locations()
        MLAM_PARAMS["counter_goals"] = counter_locations
        MLAM_PARAMS["counter_drop"] = counter_locations
        MLAM_PARAMS["counter_pickup"] = counter_locations

        if alg == "LLMPair":
            mlam = MediumLevelPlanner.from_pickle_or_compute(
                mdp, MLAM_PARAMS, force_compute=True
            ).ml_action_manager
            agent = LLMAgents(mlam, layout, **gptargs)

        elif alg == "Greedy":
            mlam = MediumLevelPlanner.from_pickle_or_compute(
                mdp, MLAM_PARAMS, force_compute=True
            )
            agent = GreedyHumanModel(mlam)

    else:
        raise ValueError("Unsupported algorithm.")

    agent.set_mdp(mdp)

    return agent


# make the example into embedding for retrieval
def get_example_embedding(example_path, save_path=""):
    input = ""
    import openai
    import os
    import pandas as pd

    key = ""
    del_index = []
    cwd = os.getcwd()
    key_file = os.path.join(cwd, "openai_key.txt")
    with open(key_file, "r") as f:
        key = f.read()
    openai.api_key = key

    with open(example_path, "r") as f:
        input = f.read()
        if input[0] == "\n":
            input = input[1:]
        input = input.split("</example>")
        for index, l in enumerate(input):
            input[index] = input[index].strip("\n\n")
            input[index] = input[index].strip("<example>")
            if input[index] == "":
                del_index.append(index)

    for index in sorted(del_index, reverse=True):
        del input[index]
    BATCH_SIZE = 10  # you can submit up to 2048 embedding inputs per request

    embeddings = []
    for batch_start in range(0, len(input), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(input))
        batch = input[batch_start:batch_end]
        # Only use the content before "[OUTPUT]" for embedding
        batch = list(
            map(lambda x: x[: x.index("[OUTPUT]")] if "[OUTPUT]" in x else x, batch)
        )
        print(f"Batch {batch_start} to {batch_end-1}")
        response = openai.Embedding.create(model=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response.data):
            assert i == be.index  # double check embeddings are in same order as input
        batch_embeddings = [e.embedding for e in response.data]
        embeddings.extend(batch_embeddings)
    df = pd.DataFrame({"text": input, "embedding": embeddings})

    # save embedding
    if save_path == "":
        save_path = (
            f"/home/zsw/Overcooked-Agents/src/data/embedding_"
            + ("chef" if "chef" in example_path else "assistant")
            + ".csv"
        )
    df.to_csv(save_path, index=False)
    print(f"Successfully save embedding lib in {save_path}")


def combine_statistic_dict(dict1, dict2, map, score):
    rs = dict1
    rs["actions"].append(dict2["actions"][0])
    rs["map"] = map
    rs["statistical_data"]["score"] = score
    rs["statistical_data"]["communication"][1] = dict2["statistical_data"]["communication"][1]
    rs["statistical_data"]["error"][1] = dict2["statistical_data"]["error"][1]
    rs["statistical_data"]["error_correction"][1] = dict2["statistical_data"]["error_correction"][1]

    rs["content"]["observation"][1] = dict2["content"]["observation"][1]
    rs["content"]["reflection"][1] = dict2["content"]["reflection"][1]
    rs["content"]["content"][1] = dict2["content"]["content"][1]
    rs["content"]["action_list"][1] = dict2["content"]["action_list"][1]

    return rs
