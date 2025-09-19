from overcooked_ai_py.utils import load_dict_from_file
from overcooked_ai_py.data.layouts import LAYOUTS_DIR, read_layout_dict
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from rich import print as rprint
import re
import time

import os
import json

import numpy as np
import pandas as pd
import glob
import openai
from openai import OpenAI
import pickle
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go
from scipy.special import expit

from dtw import dtw, dtwPlot, stepPattern, warp, warpArea, window
import bisect
from collections import defaultdict

from datetime import datetime

cwd = os.getcwd()

EMBEDDING_CACHE_PATH = "embedding_cache.json"
if os.path.exists(EMBEDDING_CACHE_PATH):
    with open(EMBEDDING_CACHE_PATH, "r") as f:
        embedding_cache = json.load(f)
else:
    embedding_cache = {}


gpt4_key_file = os.path.join(cwd, "openai_key.txt")
REFERENCE_DIR = os.path.join(cwd, "prompts/reference")


EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding_from_openai(content):
    with open(gpt4_key_file, "r") as f:
        context = f.read()
    key = context.split("\n")[0]
    openai.api_key = key

    get_response = False
    openai.api_key = key
    while not get_response:
        try:
            client = OpenAI(api_key=openai.api_key)
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=[content])
            get_response = True
            time.sleep(0.5)
        except Exception as e:
            print("[OPENAI ERROR]:", e)
            time.sleep(1)

    return response.data[0].embedding


def get_embedding_with_cache(content):
    """
    Get the embedding of the content, read it directly if it is already in the cache, otherwise get it and cache it through the API.
    """
    if content in embedding_cache:
        return np.array(embedding_cache[content])
    else:
        embedding = get_embedding_from_openai(content)  # Call API to get embed
        embedding_cache[content] = embedding
        # Write the new embed to the cache file
        with open(EMBEDDING_CACHE_PATH, "w") as f:
            json.dump(embedding_cache, f)
        return np.array(embedding)


def compute_and_save_embeddings(exp_log, embedding_file="embedding_data.pkl"):
    coop_message_action_pair_embedding = {}
    coop_message_action_pair = {}  # Used to store raw message and action content

    for log_idx in range(exp_log.__len__()):
        communication_history = exp_log.get_communication_history(log_idx)
        action_history = exp_log.get_action_list_timestamp(log_idx)

        for timestamp, content in communication_history.items():
            unique_timestamp = f"log_{log_idx}_ts_{timestamp}"

            # The original dialogue is deduplicated and filtered for sentences containing "[NOTHING]"
            turns = content[0]["turn"]
            filtered_turns = list(
                dict.fromkeys([turn for turn in turns if "[NOTHING]" not in turn])
            )  

            # Connect the processed sentences into a new dialogue
            coop_message = " \n\n ".join(filtered_turns)

            # Action History Extraction
            coop_action = str(action_history[timestamp][1])[1:-1]

            coop_message_embedding = get_embedding_from_openai(coop_message)
            coop_action_embedding = get_embedding_from_openai(coop_action)

            coop_message_action_pair_embedding[unique_timestamp] = {
                "message": coop_message_embedding,
                "action": coop_action_embedding,
            }
            coop_message_action_pair[unique_timestamp] = {
                "message": coop_message,
                "action": coop_action,
            }

            print(f"{unique_timestamp} embedding calculation is complete")

    with open(embedding_file, "wb") as f:
        pickle.dump((coop_message_action_pair_embedding, coop_message_action_pair), f)
    print(f"Saved embeddings to {embedding_file}")

    return coop_message_action_pair_embedding, coop_message_action_pair


def compute_cosine_similarity(
    coop_message_action_pair_embedding={}, embedding_file="embedding_data.pkl"
):

    if not os.path.exists(embedding_file):
        coop_message_action_pair_embedding = coop_message_action_pair_embedding
    else:
        with open(embedding_file, "rb") as f:
            coop_message_action_pair_embedding, coop_message_action_pair = pickle.load(
                f
            )
        print(f"Loaded embeddings from {embedding_file}")

    cosine_similarity_result = coop_message_action_pair
    sim_result_all = []
    for timestamp, content in coop_message_action_pair_embedding.items():
        # Convert a 1D array to a 2D array (1, n)
        message_embedding = np.array(content["message"]).reshape(1, -1)
        action_embedding = np.array(content["action"]).reshape(1, -1)

        sim = cosine_similarity(message_embedding, action_embedding)[0][
            0
        ]  
        cosine_similarity_result[timestamp]["cosine_similarity"] = sim
        sim_result_all.append(sim)
    avg = np.mean(sim_result_all)
    var = np.var(sim_result_all)
    result = {"mean": avg, "variance": var}
    return cosine_similarity_result, result


def calculate_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two embeddings.
    """
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]


class Reference:
    def __init__(self, order_name_list, action_encoding={}) -> None:
        self.order_name_list = order_name_list
        self.reference_seq = self.get_recipe_reference()
        self.action_encoding = action_encoding
        self.reference_encoded = self.encoded_reference_seq()

    def get_recipe_reference(self):
        reference_seq = {}
        try:
            for order_name in self.order_name_list:
                # Use glob pattern matching, * to match any number part
                pattern = f"*_{order_name}_ref.txt"
                log_files = glob.glob(os.path.join(REFERENCE_DIR, pattern))

                if not log_files:
                    print(f"No reference file for {order_name} found.")
                    continue

                # Suppose each order_name corresponds to only one file
                log_file = log_files[0]
                file_path = os.path.join(REFERENCE_DIR, log_file)

                with open(file_path, "r") as file:
                    log_content = json.load(file)
                    reference_seq[order_name] = log_content

            print(f"Successfully loaded {len(reference_seq)} reference files.")

        except Exception as e:
            print(f"Error loading reference files: {e}")
            return None
        return reference_seq

    def encoded_reference_seq(self):
        """
        Use the action_encoding of the Evaluation class to encode reference_seq.
        Recursively handle complex structures, maintaining the hierarchical relationship of the original structure.
        """

        def encode_actions(agent_actions):
            """
            Internal function: Encodes the action list of an agent, removing spaces in the action for identification.
            """
            encoded_actions = []
            for action in agent_actions:
                normalized_action = action.replace(" ", "")
                if normalized_action in self.action_encoding:
                    encoded_actions.append(self.action_encoding[normalized_action])
                else:
                    print(
                        f"Warning: Action '{action}' not found in action_encoding (normalized as '{normalized_action}')."
                    )
            return encoded_actions

        encoded_reference_seq = {}

        # Iterate each order_name reference
        for order_name, references in self.reference_seq.items():
            encoded_reference_seq[order_name] = {}
            # Iterate each reference_x
            for reference_key, agents in references.items():
                encoded_reference_seq[order_name][reference_key] = {}
                # Iterate each agent_x
                for agent, actions in agents.items():
                    # Encoding the actions of the agent
                    encoded_reference_seq[order_name][reference_key][agent] = (
                        encode_actions(actions)
                    )

        return encoded_reference_seq

    @property
    def get_reference_seq(self):
        return self.reference_seq

    @property
    def get_reference_encoded_seq(self):
        return self.reference_encoded


class ExpLog:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_data_successed = []
        self.log_data = []
        self.successed = 0
        self.failed = 0
        self.load_exp_log()

    def load_exp_log(self):
        """
        Load all log files that start with 'experiment' from the specified directory.
        """
        try:
            # Iterate through all files starting with'experiment 'in the directory
            log_files = [
                f for f in os.listdir(self.log_dir) if f.startswith("experiment")
            ]

            if not log_files:
                print("No log files starting with 'experiment' found.")
                return None

            # Extract the timestamp part of the file name and sort by time
            def extract_timestamp(file_name):
                # Suppose the timestamp format is 'experiment_YYYY-MM-DD_HH-MM-SS_...'
                # Extract the date time part（2024-12-03_04-20-29）
                timestamp_str = file_name.split("_")[
                    1:3
                ]  
                timestamp = " ".join(timestamp_str) 
                return datetime.strptime(
                    timestamp, "%Y-%m-%d %H-%M-%S"
                ) 

            log_files.sort(key=extract_timestamp)

            for log_file in log_files:
                file_path = os.path.join(self.log_dir, log_file)
                with open(file_path, "r") as file:
                    log_content = json.load(file)
                    if log_content["total_order_finished"] != []:
                        self.log_data.append(log_content)
                        self.log_data_successed.append(log_content)
                        self.successed += 1
                    else:
                        self.log_data.append(log_content)
                        self.failed += 1

            print(f"Successfully loaded {len(log_files)} log files.")

        except Exception as e:
            print(f"Error loading log files: {e}")
            return None

    def __len__(self):
        return len(self.log_data)

    @property
    def get_all_logs(self):
        """
        Return the list of all loaded logs.
        """
        return self.log_data

    @property
    def get_total_timestamp(self):
        """
        Get the 'total_timestamp' list from level 1 log.
        """
        target = []
        for log in self.log_data:
            if "total_timestamp" in log:
                target.append(log["total_timestamp"])
        return target

    @property
    def get_successed_timestamp(self):
        """
        Get the 'total_timestamp' list from level 1 log.
        """
        target = []
        for log in self.log_data_successed:
            if "total_timestamp" in log:
                target.append(log["total_timestamp"])
        return target

    @property
    def get_total_order_finished(self):
        """
        Get the 'total_order_finished' list from level 1 log.
        """
        target = []
        for log in self.log_data:
            if "total_order_finished" in log:
                target.append(log["total_order_finished"])
        return target

    @property
    def get_total_score(self):
        """
        Get the 'total_score' integer from level 1 log.
        """
        target = []
        for log in self.log_data:
            if "total_score" in log:
                target.append(log["total_score"])
        return target

    @property
    def get_total_action_list(self):
        """
        Get the 'total_action_list' from level 1 log.
        """
        target = []
        for log in self.log_data:
            if "total_action_list" in log:
                target.append(log["total_action_list"])
        return target

    @property
    def get_content_l1(self):
        """
        Get the 'content' list from level 1 log.
        """
        target = []
        for log in self.log_data:
            if "content" in log:
                target.append(log["content"])
        return target

    def get_secondary_log(self, idx):
        """
        Get a specific secondary log entry by index.
        """
        content_list = self.get_content_l1
        if content_list and len(content_list) > idx:
            return content_list[idx]
        return None

    def get_secondary_timestamp(self, idx):
        """
        Get 'timestamp' from a secondary log.
        """
        secondary_log = self.get_secondary_log(idx)
        if secondary_log and "timestamp" in secondary_log:
            return secondary_log["timestamp"]
        return None

    def get_secondary_order_list(self, idx, time_stamp):
        """
        Get 'order_list' from a secondary log.
        """
        secondary_log = self.get_secondary_log(idx)
        if secondary_log:
            return secondary_log[time_stamp]["order_list"]
        return None

    def get_tertiary_statistical_data(self, idx):
        """
        Get the 'statistical_data' from a secondary log.
        """
        secondary_log = self.get_secondary_log(idx)
        if secondary_log and "statistical_data" in secondary_log:
            return secondary_log["statistical_data"]
        return None

    def get_communication_history(self, idx):
        """
        Get the 'communication' in 'statistical_data' from secondary log.
        """
        secondary_log = self.get_secondary_log(idx)
        if secondary_log:
            communication_history = {}
            for each_timestamp_log in secondary_log:
                if (
                    each_timestamp_log["statistical_data"]["communication"][0]["call"]
                    != 0
                ):
                    contents = each_timestamp_log["statistical_data"]["communication"]
                    # Delete empty log
                    index = []
                    for idx, content in enumerate(contents):
                        if content["call"] == 0:
                            index.append(idx)
                    for idx in index:
                        del contents[idx]
                    communication_history[each_timestamp_log["timestamp"]] = contents

            return communication_history
        return None

    def get_timestamp_agent_content_with_communication(self, idx):
        """
        Get the 'content' in 'content' from secondary log.
        """
        secondary_log = self.get_secondary_log(idx)
        if secondary_log:
            agent_content_history = {}
            for each_timestamp_log in secondary_log:
                if (
                    each_timestamp_log["statistical_data"]["communication"][0]["call"]
                    != 0
                ):
                    if len(each_timestamp_log["content"]["content"]) == 2:

                        contents = each_timestamp_log["content"]["content"][0]
                        agent_content_history[each_timestamp_log["timestamp"]] = (
                            contents
                        )
                    else:
                        raise ValueError

            return agent_content_history
        return None

    def get_timestamp_agent_content_without_communication(self, idx):
        """
        Get the 'content' in 'content' from secondary log.
        """
        secondary_log = self.get_secondary_log(idx)
        if secondary_log:
            agent_content_history = {}
            for each_timestamp_log in secondary_log:
                if (
                    each_timestamp_log["statistical_data"]["communication"][0]["call"]
                    == 0
                    and each_timestamp_log["statistical_data"]["communication"][1][
                        "call"
                    ]
                    == 0
                ):
                    if len(each_timestamp_log["content"]["content"]) == 2:
                        if (
                            each_timestamp_log["content"]["content"][0] != []
                            or each_timestamp_log["content"]["content"][1] != []
                        ):
                            contents = each_timestamp_log["content"]["content"]
                            agent_content_history[each_timestamp_log["timestamp"]] = (
                                contents
                            )
                    else:
                        raise ValueError
            return agent_content_history
        return None

    def get_action_list_timestamp(self, idx):
        """
        Get the 'action_list' in 'content' from secondary log.
        """
        secondary_log = self.get_secondary_log(idx)
        if secondary_log:
            action_list_timestamp = {}
            for each_timestamp_log in secondary_log:
                action_list_timestamp[each_timestamp_log["timestamp"]] = (
                    each_timestamp_log["content"]["action_list"]
                )

            return action_list_timestamp
        return None

    def get_action_and_content_with_communication(self):
        result_list = []
        total_action_list = self.get_total_action_list

        for idx in range(self.__len__()):
            content_temp = self.get_timestamp_agent_content_with_communication(idx)
            action_temp = self.get_action_list_timestamp(idx)

            for key, content_item in content_temp.items():
                action_item = action_temp.get(key)

                # Get the historical actions of two agents
                history_action = []
                for agent_actions in total_action_list[idx]:
                    filtered_actions = [
                        action for action in agent_actions if action["timestamp"] <= key
                    ]
                    history_action.append(filtered_actions)

                result_list.append(
                    {
                        "content": content_item,
                        "action": action_item,
                        "history_action": history_action,
                        "id": idx,
                    }
                )

        return result_list

    def get_action_and_content_without_communication(self):
        result_list = []
        total_action_list = self.get_total_action_list

        for idx in range(self.__len__()):
            content_temp = self.get_timestamp_agent_content_without_communication(idx)
            action_temp = self.get_action_list_timestamp(idx)

            for key, content_item in content_temp.items():
                action_item = action_temp.get(key)

                # Get the historical actions of two agents
                history_action = []
                for agent_actions in total_action_list[idx]:
                    filtered_actions = [
                        action for action in agent_actions if action["timestamp"] < key
                    ]
                    history_action.append(filtered_actions)

                result_list.append(
                    {
                        "content": content_item,
                        "action": action_item,
                        "history_action": history_action,
                    }
                )

        return result_list


class Evaluation:
    def __init__(
        self,
        order_name_list=[],
        action_space=[],
        predict=[],
        recipe=None,
        layout_name="new_env",
        exp_log=None,
        **kwargs,
    ):
        """
        recipe: the recipe need to measure similarity
        """
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.action_space = action_space

        self.encode_reference = np.array
        self.action_space_agent_0 = []
        self.action_space_agent_1 = []

        self.predict = predict

        self.layout_name = layout_name
        self._get_action_space()

        self.action_encoding = {}
        self.action_encoded_agent_0, self.action_encoded_agent_1 = (
            self._encode_action_space()
        )

        self.order_name_list = order_name_list
        self.reference = Reference(set(order_name_list), self.action_encoding)

        self.reference_encoded = self.reference.get_reference_encoded_seq
        self.exp_log = exp_log
        self.exp_action_agent = self.exp_log.get_total_action_list
        self.log_action_encoded = self._encode_action_log()

    def _get_action_space(self):
        """
        Use various combinations to get the action space for converting to integer coding
        agent0: check_recipe, pickup, put_obj_in_utensil, place_obj_on_counter, fill_dish_with_food, deliver_soup, utensil_operation
        agent1: pickup, place_obj_on_counter, put_obj_in_utensil, utensil_operation
        """
        action_space_agent_0 = [
            "check_recipe()",
            "place_obj_on_counter()",
            "fill_dish_with_food()",
            "deliver_soup()",
            "pickup(dish,counter)",
        ]
        action_space_agent_1 = [
            "place_obj_on_counter()",
            "pickup(dish,counter)",
            "pickup(dish,dish_dispenser)",
        ]

        pick_place_agent_0 = ["counter"]
        pick_place_agent_1 = ["ingredient_dispenser", "dish_dispenser", "counter"]
        # load utensil_operation
        layout = read_layout_dict(self.layout_name)
        utensil_list = self.mdp.generate_utensil_list()
        for utensil, operation in layout["utensil_agent0"].items():
            for utensil_with_num in utensil_list:
                if utensil in utensil_with_num:
                    action_space_agent_0.append(f"{operation}({utensil_with_num})")
                    action_space_agent_0.append(
                        f"put_obj_in_utensil({utensil_with_num})"
                    )
                    action_space_agent_0.append(
                        f"fill_dish_with_food({utensil_with_num})"
                    )
                    pick_place_agent_0.append(utensil_with_num)
        for utensil, operation in layout["utensil_agent1"].items():
            for utensil_with_num in utensil_list:
                if utensil in utensil_with_num:
                    action_space_agent_1.append(f"{operation}({utensil_with_num})")
                    action_space_agent_1.append(
                        f"put_obj_in_utensil({utensil_with_num})"
                    )
                    action_space_agent_0.append(
                        f"fill_dish_with_food({utensil_with_num})"
                    )
                    pick_place_agent_1.append(utensil_with_num)

        # load ingredients, cooked food
        ingredients = []
        for i in layout["ingredients"]:
            ingredients.append(i)
        for _, r in layout["recipes"].items():
            for s in r.keys():
                if s not in ingredients:
                    ingredients.append(s)
        # pickup, add all combination.
        for f in ingredients:
            for p in pick_place_agent_0:
                action_space_agent_0.append(f"pickup({f},{p})")
            for p in pick_place_agent_1:
                action_space_agent_1.append(f"pickup({f},{p})")

        self.action_space_agent_0 = action_space_agent_0
        self.action_space_agent_0.append("[NONE]")
        self.action_space.extend(action_space_agent_0)
        self.action_space_agent_1 = action_space_agent_1
        self.action_space_agent_1.append("[NONE]")
        self.action_space.extend(action_space_agent_1)

        self.action_space = list(set(self.action_space))

        print(
            f"""Number of agent_0 action space is :{len(action_space_agent_0)},\n  
                Number of agent_1 action space is :{len(action_space_agent_1)},\n
                Total action space is: {len(self.action_space)}"""
        )

    def _encode_action_space(self):
        """
        Encodes action_space and its subsets and returns the respective encoding sequence.
        """
        # encoding action_space
        for idx, action in enumerate(self.action_space):
            self.action_encoding[action] = idx

        agent_0_encoded = [
            self.action_encoding[action] for action in self.action_space_agent_0
        ]
        agent_1_encoded = [
            self.action_encoding[action] for action in self.action_space_agent_1
        ]

        return agent_0_encoded, agent_1_encoded

    def _encode_action_log(self):
        """
        Extract the action sequences of the two agents and encode them using self.action_encoding.
        The return value is a list containing the encodings of two agent action sequences.
        """
        encoded_action_log = []

        for agent_actions in self.exp_action_agent:

            encoded_action_one_log = []
            for action_dict in agent_actions:
                agent_encoded_actions = []
                action_sequence = [
                    action["action"] for action in action_dict
                ] 

                if len(action_sequence) == 0:
                    action_sequence.append("[NONE]")

                for action in action_sequence:
                    normalized_action = action.replace(" ", "")

                    # Encodes the action and issues a warning if the action is not found in the mapping table
                    if normalized_action in self.action_encoding:
                        agent_encoded_actions.append(
                            self.action_encoding[normalized_action]
                        )
                    else:
                        print(
                            f"Warning: Action '{action}' (normalized as '{normalized_action}') not found in action_encoding."
                        )

                # Add the encoded action sequence for each agent to the result list
                encoded_action_one_log.append(agent_encoded_actions)
            encoded_action_log.append(encoded_action_one_log)
        return encoded_action_log

    def encode_action_sequence(self, seq):
        encoded_seq = []
        sequence = seq

        for action in sequence:
            normalized_action = action.replace(" ", "")
            if normalized_action in self.action_encoding:
                encoded_seq.append(self.action_encoding[normalized_action])
            else:
                print(
                    f"Warning: Action '{action}' (normalized as '{normalized_action}') not found in action_encoding."
                )

        return encoded_seq

    def dtw_similarity(self, reference, predict):
        """
        Input:
            reference (np, dim: [N, max_sequence_len], N is the number of reference sequence, max_sequence_len is the longest sequence len)
            predict (np, dim:[M,max_predict_len], M is the number of predict sequence, max_predict_len is the longest sequence len)
        Output:
            dtw_distance (M, 1) , each element is the min distance between predict seq and several reference seq
            Visual images in save_dir
        """
        N = 1
        dist = lambda x, y: 0 if x == y else N
        if len(reference) == 0 or len(predict) == 0:
            return None, 0
        dtw_result = dtw(
            predict,
            reference,
            keep_internals=True,
            dist_method=dist,
            step_pattern="symmetric1",
        ).distance
        max_dtw = N * (len(reference) + len(predict))
        min_max_dtw = 1 - (dtw_result / max_dtw)
        return dtw_result, min_max_dtw

    def action_rbo(
        self, reference_encoded, log_action_encoded, log_action_weight=[], p=0.9
    ):
        """ """
        len_log = len(log_action_encoded)
        len_ref = len(reference_encoded)
        max_depth = max(len_log, len_ref)

        def compute_agreement_at_depth(d):
            log_prefix = log_action_encoded[:d]
            ref_prefix = reference_encoded[:d]

            log_multiset = {}
            ref_multiset = {}

            for action in log_prefix:
                log_multiset[action] = log_multiset.get(action, 0) + 1
            for action in ref_prefix:
                ref_multiset[action] = ref_multiset.get(action, 0) + 1

            intersection_size = 0
            for action in log_multiset:
                if action in ref_multiset:
                    intersection_size += min(log_multiset[action], ref_multiset[action])

            return intersection_size / d

        rbo = 0
        max_rbo = 0
        for d in range(1, max_depth + 1):
            Ad = compute_agreement_at_depth(d)
            weight = (1 - p) * (p ** (d - 1))

            rbo += weight * Ad
            max_rbo += weight

        return rbo / max_rbo if max_rbo > 0 else 0

    def calculate_overlap_and_redundancy(
        self, reference_encoded, log_action_encoded, return_multiset=False
    ):
        """
        compute overlap and redundancy of two sequence

        return:
        tuple: (multiset list, overlap, redundancy)
        """

        ref_count = Counter(reference_encoded)
        log_count = Counter(log_action_encoded)

        multiset_intersection = []
        for action in ref_count:
            if action in log_count:
                multiset_intersection.extend(
                    [action] * min(ref_count[action], log_count[action])
                )

        overlap_ratio = (
            len(multiset_intersection) / len(reference_encoded)
            if len(reference_encoded) > 0
            else 0
        )

        redundancy_ratio = (
            (len(log_action_encoded) - len(multiset_intersection))
            / len(log_action_encoded)
            if len(multiset_intersection) > 0
            else 0
        )

        if return_multiset:
            return multiset_intersection, overlap_ratio, redundancy_ratio
        else:
            return (overlap_ratio, redundancy_ratio)

    def tes(self, reference_encoded, log_action_encoded):

        # Preprocess the action sequence
        element_positions = defaultdict(list)
        for idx, elem in enumerate(log_action_encoded):
            element_positions[elem].append(idx)

        max_depth = 0

        # Positions in action where action[pos] == reference[0]
        starting_positions = element_positions.get(reference_encoded[0], [])

        for start_pos in starting_positions:
            pos_action = start_pos
            depth = 1
            pos_ref = 0

            while pos_ref < len(reference_encoded) - 1:
                pos_ref += 1
                ref_elem = reference_encoded[pos_ref]
                positions = element_positions.get(ref_elem, [])
                # Find the first position in positions greater than pos_action
                idx = bisect.bisect_right(positions, pos_action)
                if idx < len(positions):
                    pos_action = positions[idx]
                    depth += 1
                else:
                    break

            if depth > max_depth:
                max_depth = depth

        tp = max_depth / len(reference_encoded)

        if len(log_action_encoded) == 0:
            ea = 0
        else:
            ea = max_depth / len(log_action_encoded)

        BETA = 0.95

        f1 = (1 + BETA * BETA) * max_depth / (len(reference_encoded) +  BETA * BETA * len(log_action_encoded))

        return tp, ea, f1
    

    def ites(self, seq_1, seq_2, agent_id):
        sim_seq_1_list = []
        sim_seq_2_list = []
        for order_name, reference_dict in self.reference_encoded.items():
            for _, reference_seq_encoded in reference_dict.items():
                for agent, content in enumerate(reference_seq_encoded):
                    if agent == agent_id:
                        sim_seq_1_temp, _, f1_seq_1_temp = self.tes(
                            reference_seq_encoded[content], seq_1
                        )
                        sim_seq_1_list.append(f1_seq_1_temp)
                        sim_seq_2_temp, _, f2_seq_1_temp = self.tes(
                            reference_seq_encoded[content], seq_2
                        )
                        sim_seq_2_list.append(f2_seq_1_temp)
        sim_seq_1 = max(sim_seq_1_list)
        sim_seq_2 = max(sim_seq_2_list)

        return sim_seq_1 - sim_seq_2

    def get_action_similarity_with_cache(self, content, agent_id, similarity=0.9):
        # Loading or updating the embedding cache for content and actions
        try:
            with open(EMBEDDING_CACHE_PATH, "r") as f:
                embedding_cache = json.load(f)
        except FileNotFoundError:
            embedding_cache = {}

        action_cache_path = (
            f"{EMBEDDING_CACHE_PATH[:-5]}_agent_{str(1 - int(agent_id))}.json"
        )
        try:
            with open(action_cache_path, "r") as f:
                action_embedding_cache = json.load(f)
        except FileNotFoundError:
            action_embedding_cache = {}

        # Get a list of actions and calculate or find their embeddings
        action_list = getattr(self, f"action_space_agent_{str(1 - int(agent_id))}")
        action_embeddings = []

        new_embedding_action_list = False
        for action in action_list:
            if action in action_embedding_cache:
                action_embedding = np.array(action_embedding_cache[action])
            else:
                action_embedding = get_embedding_from_openai(action)
                action_embedding_cache[action] = action_embedding  
                new_embedding_action_list = True

            action_embeddings.append((action, np.array(action_embedding)))

        if new_embedding_action_list:
            with open(action_cache_path, "w") as f:
                json.dump(action_embedding_cache, f)

        if content in embedding_cache:
            content_embedding = np.array(embedding_cache[content])
        else:
            content_embedding = get_embedding_from_openai(content) 
            content_embedding = np.array(content_embedding)
            embedding_cache[content] = content_embedding.tolist()

        # Write all updated content embed to cache file
        with open(EMBEDDING_CACHE_PATH, "w") as f:
            json.dump(embedding_cache, f)

        # Calculate content_embedding similarity to all actions and find the maximum
        max_similarity = -1
        most_similar_action = None

        for action, action_embedding in action_embeddings:
            current_similarity = calculate_similarity(
                content_embedding, action_embedding
            )
            if current_similarity > max_similarity:
                max_similarity = current_similarity
                most_similar_action = action

        # Return actions that meet the similarity threshold or None
        if max_similarity >= similarity:
            print(
                f"Original Action: {content}, Changed action: {most_similar_action}, similarity is: {max_similarity}"
            )
            return most_similar_action
        else:
            print(f"Original Action: {content}, Changed action: {max_similarity}, No change")
            return None

    def compute_action_matrix(self, log_action_encoded):
        """
        Compute matrix A^i
        """
        agent_1_actions = []
        agent_2_actions = []

        for experiment in log_action_encoded:
            agent_1_actions.append(experiment[0])
            agent_2_actions.append(experiment[1])

        max_length_1 = max(len(actions) for actions in agent_1_actions)
        max_length_2 = max(len(actions) for actions in agent_2_actions)

        action_matrix_1 = np.zeros((len(agent_1_actions), max_length_1), dtype=int)
        action_matrix_2 = np.zeros((len(agent_2_actions), max_length_2), dtype=int)

        for i, actions in enumerate(agent_1_actions):
            action_matrix_1[i, : len(actions)] = actions
        for i, actions in enumerate(agent_2_actions):
            action_matrix_2[i, : len(actions)] = actions

        return action_matrix_1, action_matrix_2

    def compute_transition_matrix(self, action_matrix, action_space_size):
        """
        Compute action transition matrix C^i
        """
        # init
        transition_matrix = np.zeros((action_space_size, action_space_size), dtype=int)

        for actions in action_matrix:
            for t in range(len(actions) - 1):
                current_action = actions[t]
                next_action = actions[t + 1]
                if next_action != 0:
                    transition_matrix[current_action, next_action] += 1

        return transition_matrix

    def compute_task_metrics(self):
        total_timestamp = self.exp_log.get_successed_timestamp
        time_finish = []
        for timestamp_log in total_timestamp:
            time_finish.append(timestamp_log[-1])
        success_rate = self.exp_log.successed / (
            self.exp_log.successed + self.exp_log.failed
        )
        if success_rate == 0:
            time_avg = 0
            time_var = 0
        else:
            time_avg = np.mean(time_finish)
            time_var = np.std(time_finish)

        task_metrics = {
            "time_avg": time_avg,
            "time_var": time_var,
            "success_rate": success_rate,
        }

        return task_metrics

    def evaluate(self, save_dir):
        print("======Start evaluation=====")

        result = {}
        for order_name, reference_dict in self.reference_encoded.items():

            similarity_and_redundancy_result = {"agent_0": [], "agent_1": []}

            # Get the index that matches the current order_name
            matching_indices = [
                i for i, name in enumerate(self.order_name_list) if name == order_name
            ]

            # Iterate only subsets of matching log_action_encoded
            for idx in matching_indices:
                one_log_action_encoded = self.log_action_encoded[idx]

                min_max_dtw_each_reference = {"agent_0": [], "agent_1": []}

                similarity_and_redundancy_each_reference = {
                    "agent_0": [],
                    "agent_1": [],
                }

                for _, reference_seq_encoded in reference_dict.items():

                    for agent, content in enumerate(reference_seq_encoded):

                        min_max_dtw_each_reference[content].append(
                            self.dtw_similarity(
                                reference_seq_encoded[content],
                                one_log_action_encoded[agent],
                            )[1]
                        )
                        similarity_content, redundancy_content, f1_content = (
                            self.tes(
                                reference_seq_encoded[content],
                                one_log_action_encoded[agent],
                            )
                        )
                        similarity_and_redundancy_each_reference[content].append(
                            (similarity_content, redundancy_content, f1_content)
                        )

                # Find the maximum overlap value and the corresponding redundancy
                for agent in similarity_and_redundancy_each_reference:
                    if similarity_and_redundancy_each_reference[agent]:
                        max_similarity, corresponding_redundancy, max_f1 = max(
                            similarity_and_redundancy_each_reference[agent],
                            key=lambda x: x[2],
                        )
                        similarity_and_redundancy_result[agent].append(
                            {
                                "max_f1": max_f1,
                                "corresponding_similarity": max_similarity,
                                "corresponding_redundancy": corresponding_redundancy,
                            }
                        )

            result[order_name] = {
                "similarity_and_redundancy_result": similarity_and_redundancy_result
            }

            result[order_name]["average"] = {}

            result[order_name]["average"]["similarity_and_redundancy"] = {}
            for agent in ["agent_0", "agent_1"]:
                if result[order_name]["similarity_and_redundancy_result"][agent]:
                    max_overlaps = [
                        item["max_f1"]
                        for item in result[order_name][
                            "similarity_and_redundancy_result"
                        ][agent]
                    ]
                    similarities = [
                        item["corresponding_similarity"]
                        for item in result[order_name][
                            "similarity_and_redundancy_result"
                        ][agent]
                    ]
                    redundancies = [
                        item["corresponding_redundancy"]
                        for item in result[order_name][
                            "similarity_and_redundancy_result"
                        ][agent]
                    ]

                    # Calculate the mean and variance of overlap and redundancy
                    avg_overlap = np.mean(max_overlaps)
                    var_overlap = np.std(max_overlaps)
                    avg_similarity = np.mean(similarities)
                    var_similarity = np.std(similarities)
                    avg_redundancy = np.mean(redundancies)
                    var_redundancy = np.std(redundancies)

                    result[order_name]["average"]["similarity_and_redundancy"][
                        agent
                    ] = {
                        "mean_f1": avg_overlap,
                        "mean_similarity": avg_similarity,
                        "mean_redundancy": avg_redundancy,
                        "std_f1": var_overlap,
                        "std_similarity": var_similarity,
                        "std_redundancy": var_redundancy,
                    }
                else:
                    result[order_name]["average"]["similarity_and_redundancy"][
                        agent
                    ] = {
                        "mean_f1": None,
                        "mean_similarity": None,
                        "mean_redundancy": None,
                        "std_f1": None,
                        "std_similarity": None,
                        "std_redundancy": None,
                    }
        a0_sim = result[order_name]["average"]["similarity_and_redundancy"]["agent_0"]["mean_similarity"]
        a1_sim = result[order_name]["average"]["similarity_and_redundancy"]["agent_1"]["mean_similarity"]
        a0_red = result[order_name]["average"]["similarity_and_redundancy"]["agent_0"]["mean_redundancy"]
        a1_red = result[order_name]["average"]["similarity_and_redundancy"]["agent_1"]["mean_redundancy"]

        a0_ref_len = len(self.reference_encoded[order_name]["reference_1"]["agent_0"])
        a1_ref_len = len(self.reference_encoded[order_name]["reference_1"]["agent_1"])
        first_lengths = [len(item[0]) for item in self.log_action_encoded]
        a0_act_len = sum(first_lengths) / len(first_lengths)
        second_lengths = [len(item[1]) for item in self.log_action_encoded]
        a1_act_len = sum(second_lengths) / len(second_lengths)

        overall_sim = (a0_sim * a0_ref_len + a1_sim * a1_ref_len) / (
            a0_ref_len + a1_ref_len
        )
        overall_red = ((1 - a0_red) * a0_act_len + (1 - a1_red) * a1_act_len) / (
            a0_act_len + a1_act_len
        )
        overall_f1 = 2 * overall_sim * overall_red / (overall_sim + overall_red)
        result[order_name]["average"]["similarity_and_redundancy"]["overall"] = {
            "mean_f1": overall_f1
        }

        task_metrics = self.compute_task_metrics()

        result[order_name]["task_metrics"] = task_metrics

        collaboration_confusion_matrix, process_result = self.evaluate_collaboration(save_dir)


        initate_count = (
            collaboration_confusion_matrix[0][0][0]
            + collaboration_confusion_matrix[0][1][0]
            + collaboration_confusion_matrix[1][0][0]
            + collaboration_confusion_matrix[2][0][0]
        )
        respond_count = (
            collaboration_confusion_matrix[0][0][0]
            + collaboration_confusion_matrix[2][0][1]
            + collaboration_confusion_matrix[1][0][0]
            + collaboration_confusion_matrix[2][0][0]
        )
        total_count = (
            collaboration_confusion_matrix[0][0][0]
            + collaboration_confusion_matrix[0][1][0]
            + collaboration_confusion_matrix[0][1][1]
            + collaboration_confusion_matrix[1][0][0]
            + collaboration_confusion_matrix[1][1][1]
            + collaboration_confusion_matrix[2][0][0]
            + collaboration_confusion_matrix[2][0][1]
            + collaboration_confusion_matrix[2][1][1]
        )

        if total_count != 0:
            initiate_collaboration = initate_count / total_count
            respond_collaboration = respond_count / total_count
        else:
            initiate_collaboration = 0
            respond_collaboration = 0

        statistic_result = {
            "initiate_collaboration": initiate_collaboration,
            "respond_collaboration": respond_collaboration
        }
        result[order_name]["statistic"] = statistic_result
        result[order_name]["confusion_matrix"] = collaboration_confusion_matrix

        result[order_name]["process_result"] = process_result

        save_path = os.path.join(save_dir, "evaluation_result.json")
        with open(save_path, "w") as json_file:
            json.dump(result, json_file, indent=4, ensure_ascii=False)

        print(f"Results saved to {save_path}")
        return result


    def extract_actions(self, plan: str):
        # Use regular expressions to match all actions following "plan:"
        match = re.search(r"plan:\s*(.*)", plan)  
        if match:
            # Extract the part after "plan:" and use the regular to find the action
            actions_part = match.group(1)
            actions = re.findall(r"\w+\([^\)]*\)|\w+", actions_part)  
            actions = [action.replace(" ", "") for action in actions]
            actions = [
                action
                for action in actions
                if "request" not in action
                and "wait" not in action
                and "NOTHING" not in action
            ]
            return actions
        return []

    def evaluate_collaboration(self, save_dir):
        confusion_matrix = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        process_result = [[]]

        case_study_3 = []

        current_id = 0

        content_action_list = self.exp_log.get_action_and_content_with_communication()

        for item in content_action_list:
            begin_agent = None
            request_action = []
            request_action_say = []
            history_action = []
            action_agent = []
            plan_action = []
            item["statistic_data"] = {}
            item["result"] = {}

            # Find the first agent to speak
            for content in item["content"]:
                if content["say"] and content["say"] != "[NOTHING]":
                    begin_agent = content["agent"]
                    break

            for actions in item["history_action"][1 - int(begin_agent)]:
                history_action.append(actions["action"])

            action_agent = item["action"][1 - int(begin_agent)]
            action_agent_cleaned = [
                action
                for action in action_agent
                if "request" not in action and "wait" not in action
            ]

            for content in item["content"]:
                if content["agent"] == 1 - int(begin_agent):
                    plan_text = content.get("plan", "")
                    plan_requests = re.findall(r"\w+\(.*?\)", plan_text)
                    if len(plan_requests) != 0:

                        plan_requests = [req.replace(" ", "") for req in plan_requests]

                        # Delete elements with'request 'and'wait'
                        plan_requests = [
                            req
                            for req in plan_requests
                            if "request" not in req and "wait" not in req
                        ]
                        # Check if each element is in action_space_agent
                        plan_requests = [
                            req
                            for req in plan_requests
                            if req
                            in getattr(
                                self, f"action_space_agent_{str(1 - int(begin_agent))}"
                            )
                        ]
                        # temp = [req for req in plan_requests if req not in getattr(self, f'action_space_agent_{str(1 - int(begin_agent))}')]
                        # print(temp)
                        # print(plan_requests)

                        plan_action.extend(plan_requests)
                        break
                    else:
                        continue

            for content in item["content"]:
                if content["agent"] == begin_agent:
                    # Extract the requested action from the plan
                    plan_text = content.get("plan", "")
                    plan_requests = re.findall(r"request\((.+?)\)", plan_text)
                    # Add the missing closing bracket for each matching request action
                    cleaned_plan_requests = [
                        req.strip(" '") + ")" for req in plan_requests
                    ]
                    cleaned_plan_requests = [
                        req.replace(" ", "") for req in cleaned_plan_requests
                    ]
                    request_action.extend(cleaned_plan_requests)

                    # Extract the request action in say
                    say_text = content.get("say", "")
                    say_requests = re.findall(r"request\((.+?)\)", say_text)

                    cleaned_say_requests = [
                        req.strip(" '") + ")" for req in say_requests
                    ]
                    cleaned_say_requests = [
                        req.replace(" ", "") for req in cleaned_say_requests
                    ]
                    request_action_say.extend(cleaned_say_requests)

            request_action_cleaned = [
                action
                for action in request_action
                if "request" not in action and "wait" not in action
            ]

            request_action_say_cleaned = [
                action
                for action in request_action_say
                if "request" not in action and "wait" not in action
            ]

            if request_action != [] or request_action != ["[NONE]"]:
                request_action_processed = []
                for request_action_one in request_action:
                    if request_action_one in getattr(
                        self, f"action_space_agent_{str(1 - int(begin_agent))}"
                    ):
                        request_action_processed.append(request_action_one)
                    else:
                        # Here can adjust the sensitivity of action matching
                        pass
                        # result_temp = self.get_action_similarity_with_cache(request_action_one, begin_agent)
                        # if result_temp is not None:
                        #     request_action_processed.append(result_temp)

            elif request_action_say != [] or request_action_say != ["[NONE]"]:
                request_action_processed = []
                for request_action_one in request_action:
                    if request_action_one in getattr(
                        self, f"action_space_agent_{str(1 - int(begin_agent))}"
                    ):
                        request_action_processed.append(request_action_one)
                    else:
                        # Here can adjust the sensitivity of action matching
                        pass
                        # result_temp = self.get_action_similarity_with_cache(request_action_one, begin_agent)
                        # if result_temp is not None:
                        #     request_action_processed.append(result_temp)


            item["statistic_data"]["begin_agent"] = begin_agent
            # item['statistic_data']['agent_action_collaboration'] = action_agent
            item["statistic_data"]["history_agent_action"] = history_action
            item["statistic_data"]["request_action"] = request_action_processed
            # item['statistic_data']['request_action_say'] = request_action_say_cleaned
            item["statistic_data"]["plan_action"] = plan_action
            item["statistic_data"]["request_action_encoded"] = (
                self.encode_action_sequence(request_action_processed)
            )
            # item['statistic_data']['agent_action_collaboration_encoded'] = self.encode_action_sequence(action_agent_cleaned)
            item["statistic_data"]["plan_action_encoded"] = self.encode_action_sequence(
                plan_action
            )
            # item['statistic_data']['request_action_say_encoded'] = self.encode_action_sequence(request_action_say_cleaned)
            item["statistic_data"]["history_agent_action_encoded"] = (
                self.encode_action_sequence(history_action)
            )


            H = item["statistic_data"]["history_agent_action_encoded"]
            R = item["statistic_data"]["request_action_encoded"]
            R_hat = item["statistic_data"]["plan_action_encoded"]

            c_1 = self.ites(H + R, H, 1 - int(begin_agent))
            c_2 = self.ites(H + R, H + R_hat, 1 - int(begin_agent))

            c_3 = self.ites(H + R_hat, H, 1 - int(begin_agent))
            # c_2 = c_1 - c_3
            item["result"]["c_1"] = c_1
            item["result"]["c_2"] = c_2
            item["result"]["c_3"] = c_3

            if c_2 > 0:
                index_c_2 = 0
                if c_1 > 0 and c_3 > 0:
                    index_c_1 = 0
                    index_c_3 = 0
                elif c_1 > 0 and c_3 <= 0:
                    index_c_1 = 1
                    index_c_3 = 0

                elif c_1 <= 0 and c_3 <= 0:
                    index_c_1 = 1
                    index_c_3 = 1
                else:
                    raise ValueError
            elif c_2 == 0:
                index_c_2 = 1
                if c_1 > 0 and c_3 > 0:
                    index_c_1 = 0
                    index_c_3 = 0
                elif c_1 <= 0 and c_3 <= 0:
                    index_c_1 = 1
                    index_c_3 = 1
                else:
                    raise ValueError
            else:
                index_c_2 = 2
                if c_1 > 0 and c_3 > 0:
                    index_c_1 = 0
                    index_c_3 = 0
                elif c_1 <= 0 and c_3 > 0:
                    index_c_1 = 0
                    index_c_3 = 1
                elif c_1 <= 0 and c_3 <= 0:
                    index_c_1 = 1
                    index_c_3 = 1
                else:

                    raise ValueError

            if item["id"] != current_id:
                process_result.append([])
                current_id = item["id"]

            if c_1 > 0 and c_3 > 0:
                process_result[-1].append(0)
            elif c_1 > 0 and c_3 <= 0:
                process_result[-1].append(1)
            elif c_1 <= 0 and c_3 > 0:
                process_result[-1].append(2)
            elif c_1 <= 0 and c_3 <= 0:
                case_study_3.append(item)
                process_result[-1].append(3)

            confusion_matrix[index_c_2][index_c_1][index_c_3] += 1

        # print(confusion_matrix)
        # print(process_result)
        save_path = os.path.join(save_dir, "case_study_3.json")
        with open(save_path, "w") as json_file:
            json.dump(case_study_3, json_file, indent=4, ensure_ascii=False)
        return confusion_matrix, process_result
