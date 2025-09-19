import itertools, os, json, re
from collections import defaultdict
from typing import Union
import numpy as np
import pkg_resources
from collections import deque
import sys
import copy
from .modules import Module, statistics_dict, turn_statistics_dict
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.search import find_path
from overcooked_ai_py.planning.search import get_intersect_counter
from overcooked_ai_py.planning.search import query_counter_states
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
import queue
import warnings
import copy

from rich import print as rprint
from collab.modules import if_two_sentence_similar_meaning

cwd = os.getcwd()
openai_key_file = os.path.join(cwd, "openai_key.txt")
PROMPT_DIR = os.path.join(cwd, "prompts")

NAME_TO_ACTION = {
    "NORTH": Direction.NORTH,
    "SOUTH": Direction.SOUTH,
    "EAST": Direction.EAST,
    "WEST": Direction.WEST,
    "INTERACT": Action.INTERACT,
    "STAY": Action.STAY,
}


class LLMPair(object):

    def __init__(
        self,
        model="gpt-3.5-turbo-0301",
        model_dirname="~/",
        local_server_api="http://localhost:8000/v1",
    ):
        self.agent_index = None
        self.model = model
        self.model_dirname = model_dirname
        self.local_server_api = local_server_api

        self.openai_api_keys = []
        self.load_openai_keys()
        self.key_rotation = True
        self.proxy = "http://10.29.202.138:7890"

    def load_openai_keys(self):
        with open(openai_key_file, "r") as f:
            context = f.read()
        self.openai_api_keys = context.split("\n")

    def openai_api_key(self):
        if self.key_rotation:
            self.update_openai_key()
        return self.openai_api_keys[0]

    def update_openai_key(self):
        self.openai_api_keys.append(self.openai_api_keys.pop(0))

    def set_agent_index(self, agent_index):
        raise NotImplementedError

    def action(self, state):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class LLMAgents(LLMPair):

    def __init__(
        self,
        mlam,
        layout,
        model="gpt-3.5-turbo-0301",
        model_dirname="~/",
        local_server_api="http://localhost:8000/v1",
        retrival_method="recent_k",
        K=1,
        actor="",
        auto_unstuck=False,
        controller_mode="new",  # the default overcooked-ai Greedy controller
        debug_mode="N",
        agent_index=None,
        outdir=None,
    ):
        super().__init__(
            model=model, model_dirname=model_dirname, local_server_api=local_server_api
        )

        self.trace = True
        self.debug_mode = "Y"
        self.controller_mode = controller_mode
        self.mlam = mlam
        self.layout = layout
        self.mdp = self.mlam.mdp
        self.test_mode = False
        self.test_ml_action = deque([])

        self.out_dir = outdir
        self.agent_index = agent_index

        self.retrival_method = retrival_method
        self.K = K

        self.prev_state = None
        self.auto_unstuck = auto_unstuck

        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.possible_motion_goals = None
        self.pot_id_to_pos = []
        self.action_wait_parse = queue.Queue()
        self.end_talk = False
        self.actor = actor
        self.name = "Chef" if self.actor == "chef" else "Assistant"
        self.communication_role = "ask"
        self.recipe = {}
        self.order = ""
        self.failed_history = []
        self.state = None
        # dict to record if the error in T timestamp  was corrected, or just 'wait(1)'
        self.error_correct = {}
        turn_statistics_dict_cp = copy.deepcopy(turn_statistics_dict)
        self.turn_statistics_dict = turn_statistics_dict_cp
        # self.generate_layout_prompt()

    def set_mdp(self, mdp: OvercookedGridworld):
        self.mdp = mdp

    def create_gptmodule(
        self, module_name, file_type="txt", retrival_method="recent_k", K=10
    ):
        print(f"\n--->Initializing GPT {module_name}<---\n")

        model_name = "gpt"
        if module_name == "planner":
            prompt_file = os.path.join(
                PROMPT_DIR,
                model_name,
                f"{self.actor}_{self.communication_role}.{file_type}",
            )
            self.prompt_file = prompt_file
        # elif module_name == "explainer":
        # 	prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, f'player{self.agent_index}.{file_type}')
        else:
            raise Exception(f"Module {module_name} not supported.")
        # print(prompt_file)
        self.prompt_root = "/".join(self.prompt_file.split("/")[:-1])
        messages = [{"role": "system", "content": ""}]

        return Module(
            messages,
            self.model,
            self.model_dirname,
            self.local_server_api,
            retrival_method,
            K,
        )

    # 	return messages
    def load_prompt_file(self, mode="origin"):
        self.prompt_dir = PROMPT_DIR + "/gpt"
        prompt = ""
        with open(self.prompt_dir + "/prompt.txt", "r") as g:
            prompt = g.read()
        if mode == "origin":
            communication_rule_dir = self.prompt_dir + "/communication_rule.txt"
        elif mode == "correct":
            communication_rule_dir = self.prompt_dir + "/correct_rule.txt"
        elif mode == "reflection":
            communication_rule_dir = self.prompt_dir + "/reflection_rule.txt"

        with open(communication_rule_dir, "r") as g:
            communication_rule = g.read()
            prompt = prompt.replace("{communication_rule}", communication_rule)
            prompt = prompt.replace("{role}", self.name)
            prompt = prompt.replace("{teammate}", self.teammate.name)

        with open(self.prompt_dir + "/environment_rule.txt", "r") as g:
            environment_rule = g.read()
            environment_rule = environment_rule.replace("{role}", self.name)
            prompt = prompt.replace("{environment_rule}", environment_rule)
            character_player = "Suppose you are a player who is proficient in the overcooked_ai game. Your goal is to cooperate with your teammate who is also a LLM agent in order to get a high score."
            character_reflection = "Suppose you are a reflector who is proficient in the overcooked_ai game. Your goal is to analyzes the self-correction process of an LLM player after making a wrong action,  \
				and summarize the experience to help LLM players to reference and avoid making mistakes again."
            if mode == "reflection":
                prompt = prompt.replace("{character}", character_reflection)
            else:
                prompt = prompt.replace("{character}", character_player)
        with open(self.prompt_dir + f"/{self.actor}_skill.txt", "r") as g:
            skill = g.read()
            prompt = prompt.replace("{skill}", skill)
        chef_workflow = """- The usual workflow for the chef is:
  1. Read the cooking process from your recipe. All of your decisions must be strictly guided by the recipe and should not lead to unfounded behavior.
  2. Ask the assistant to pick up ingredients from the ingredient dispenser and use the correct utensil to handle them according to the recipe. Since you do not have access to all the objects, you need to assign some tasks to the assistant while you perform other tasks in parallel.
  3. Work in parallel with the assistant to finish the order in the shortest time possible, unless there is nothing you can do in the current situation. If you have nothing to do, you can wait.
  4. Serve the dish (optional). If the recipe specifies that the dish needs to be served on a plate, you must use `fill_dish_with_food(utensil_name)` to serve the dish from the utensil first; otherwise, just pick up the food from the utensil.
  5. Use `deliver_soup()."""
        assistant_workflow = """The usual workflow for the Assistant is:  
- 1. Ask the Chef for guidance, since you do not have the recipe and need the Chef to help you plan.  
- 2. Follow the Chef’s instructions unless they are incorrect. For example, if the Chef requests a utensil that is not available on your side, you should refuse and inform him. """
        #  -1. Communicate with the chef for instruction and don't make your own plans.\n\
        #  -2. Follow the instructions given by Chef unless his instruction is wrong. For example, if the utensil he wants you to use in not in your side, you should refuse and tell him.\n"
        # load recipe
        recipe_content = ""
        # self.load_recipe()
        if self.actor == "chef":
            self.load_recipe()
            prompt = prompt.replace("{workflow}", chef_workflow)
            prompt = prompt.replace(
                "{has_recipe}",
                "You have recipe, so you need to direct yourself and your teammates to complete the order.",
            )
            # prompt = prompt.replace("{job}","You need to list plan only for yourself, and guide Assistant by telling him your instruction in communication.")
        else:
            self.teammate.load_recipe()
            prompt = prompt.replace("{workflow}", assistant_workflow)
            prompt = prompt.replace(
                "{job}",
                "You only need to ask and follow Chef's instruction in communication without making plan by yourself. Because you do not have recipe, which means your plan is likely to be wrong.",
            )
            prompt = prompt.replace(
                "{has_recipe}",
                "You do not have recipe, and you should always ask the chef for guidance, rather than making bad decisions on your own.",
            )
        for key, value in self.recipe.items():
            recipe_content += value + "\n\n\n"
        self.planner.instruction_head_list = [{"role": "system", "content": prompt}]
        self.planner.instruction_head_list[0][
            "content"
        ] = self.planner.instruction_head_list[0]["content"].replace(
            "{recipe}",
            recipe_content if recipe_content != "" else "You do not have the recipe\n",
        )
        return self.planner.instruction_head_list[0]["content"]

    def reset(self, teammate: LLMPair):
        self.planner.reset()
        # self.explainer.reset()
        self.prev_state = None
        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.possible_motion_goals = None
        self.current_timestep = 0
        self.teammate_ml_actions = []
        self.teammate_intentions_dict = {}

        self.teammate = teammate

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index
        self.planner = self.create_gptmodule(
            "planner", retrival_method=self.retrival_method, K=self.K
        )
        self.planner.name = self.name
        # self.explainer = self.create_gptmodule("explainer", retrival_method='recent_k', K=self.K)

        # print(self.planner.instruction_head_list[0]['content'])

    def generate_grid_layout_prompt(self, state):
        return self.mdp.state_string(state).replace("ø", "o")

    def generate_layout_prompt(self):
        layout_prompt = f"Chef space:"
        if self.name == "Chef":
            utensil_list_chef = self.mdp.utensil_list_chef
            utensil_list_assistant = self.teammate.mdp.utensil_list_assist
        elif self.name == "Assistant":
            utensil_list_chef = self.teammate.mdp.utensil_list_chef
            utensil_list_assistant = self.mdp.utensil_list_assist

        for u in utensil_list_chef:
            layout_prompt += u + "  "
        layout_prompt += "counter \n"

        layout_prompt += "Assistant space:"
        for u in utensil_list_assistant:
            layout_prompt += u + "  "
        layout_prompt += "dish_dispenser  ingredient_dispenser\n"

        return layout_prompt

    def generate_state_prompt(self, state):
        self.layout_prompt = self.generate_layout_prompt()
        ego = state.players[self.agent_index]
        teammate = state.players[1 - self.agent_index]

        time_prompt = f"Scene {state.timestep}: "
        ego_object = ego.held_object.name if ego.held_object else "nothing"
        teammate_object = (
            teammate.held_object.name if teammate.held_object else "nothing"
        )
        ego_state_prompt = f"<{self.name}> holds "

        chef = self if self.actor == "chef" else self.teammate
        if ego_object in list(chef.recipe.keys()):
            ego_state_prompt += f"a dish with {ego_object} and needs to deliver soup.  "
        elif ego_object == "nothing":
            ego_state_prompt += f"{ego_object}. "
        else:
            ego_state_prompt += f"one {ego_object}. "

        action_list = self.current_ml_action if self.current_ml_action else ""
        actions = list(self.action_wait_parse.queue)
        for index, a in enumerate(actions):
            if index == 0 and self.current_ml_action is None:
                action_list += f"{a}"
            else:
                action_list += f";{a}"
        action_state_prompt = f"The planned sequence of actions (yet to be performed) for {self.name} is [{action_list}] "

        teammate_state_prompt = f"<{self.teammate.name}> holds "
        if teammate_object == "soup":
            teammate_state_prompt += f"a dish with {teammate_object}. "
        elif teammate_object == "nothing":
            teammate_state_prompt += f"{teammate_object}. "
        else:
            teammate_state_prompt += f"one {teammate_object}. "

        action_list = (
            self.teammate.current_ml_action if self.teammate.current_ml_action else ""
        )
        actions = list(self.teammate.action_wait_parse.queue)
        for index, a in enumerate(actions):
            if index == 0 and self.teammate.current_ml_action is None:
                action_list += f"{a}"
            else:
                action_list += f";{a}"
        teammate_action_state_prompt = f"The planned sequence of actions (yet to be performed) for {self.teammate.name} is [{action_list}] "

        if self.actor == "chef":
            chef = self
        else:
            chef = self.teammate
        order_prompt = (
            "Order:" + self.order
        )  # +("(Has recipe)"if self.order in chef.recipe.keys() else "(No recipe)")
        if not self.mdp.one_task_mode:
            order_prompt += (
                "<<" + state.current_k_order[1]
            )  # +("(Has recipe)"if state.current_k_order[1] in chef.recipe.keys() else "(No recipe)")
            order_prompt += (
                "<<" + state.current_k_order[2] + "\n"
            )  # +("(Has recipe)\n"if state.current_k_order[2] in chef.recipe.keys() else "(No recipe)\n")
        else:
            order_prompt += "\n"

        kitchen_state_prompt = "Kitchen states: "
        prompt_dict = {
            "empty": "<{utensil_name}> is empty; ",
            "cooking": "<{utensil_name}> starts processing {food},it will be ready after {t} timesteps; ",
            "ready": "<{utensil_name}> has {food}; ",
            "full": "<{utensil_name}> has {food}, which are all the ingredients you need to make {transition}; ",
            "partially_full": "<{utensil_name}> has {food}, which is part of {transition}; ",
            "wrong": "<{utensil_name}> has {food},which seems do not belong to any recipe;",
        }
        utensil_states_dict = self.mdp.get_utensil_states(state)

        for key in utensil_states_dict.keys():
            for utensil in utensil_states_dict[key]:
                if key == "empty":
                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil
                    )
                elif key == "cooking":
                    order = self.mdp.utensil_state_dict[utensil]["order"]
                    utensil_type = utensil[:-1]
                    # The cook time for all kinds ingredient of one certain order in an utensil should be the same
                    middle_ingredient_dict = self.mdp.recipe_config["recipes"][
                        utensil_type
                    ]
                    middle_ingredient_dict_list = list(middle_ingredient_dict.keys())
                    ingredient_in_utensil_first = self.mdp.utensil_state_dict[utensil][
                        "soup"
                    ].state[0]
                    middle_ingredient = ""
                    for m in middle_ingredient_dict_list:
                        if ingredient_in_utensil_first in m:
                            middle_ingredient = m
                    if middle_ingredient == "":
                        raise ValueError(
                            f"No operation time for {ingredient_in_utensil_first} in {utensil_type}"
                        )
                    recipe_cook_time = middle_ingredient_dict[middle_ingredient][
                        "cook_time"
                    ]
                    now_cook_time = self.mdp.utensil_state_dict[utensil]["soup"].state[
                        2
                    ]

                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil,
                        food="chopped_onion",
                        t=recipe_cook_time - now_cook_time,
                    )
                elif key == "ready":
                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil,
                        food=self.mdp.utensil_state_dict[utensil]["soup"].state[0],
                    )
                elif key == "full":
                    for rpe_name, rpe in self.mdp.recipes[utensil[:-1]].items():
                        if set(
                            self.mdp.utensil_state_dict[utensil]["soup"].state[0]
                        ) == set(rpe["recipe"]):
                            transition = rpe_name
                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil,
                        food=" ".join(
                            self.mdp.utensil_state_dict[utensil]["soup"].state[0]
                        ),
                        transition=transition,
                    )
                elif key == "partially_full":
                    for rpe_name, rpe in self.mdp.recipes[utensil[:-1]].items():
                        if (
                            len(
                                set(
                                    self.mdp.utensil_state_dict[utensil]["soup"].state[
                                        0
                                    ]
                                ).intersection(set(rpe["recipe"]))
                            )
                            > 0
                        ):
                            transition = rpe_name
                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil,
                        food=" ".join(
                            self.mdp.utensil_state_dict[utensil]["soup"].state[0]
                        ),
                        transition=transition,
                    )
                elif key == "wrong":
                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil,
                        food=" ".join(
                            self.mdp.utensil_state_dict[utensil]["soup"].state[0]
                        ),
                    )
        intersect_counters = get_intersect_counter(
            state.players_pos_and_or[self.agent_index],
            state.players_pos_and_or[1 - self.agent_index],
            self.mdp,
            self.mlam,
        )
        counter_states = query_counter_states(self.mdp, state)

        kitchen_state_prompt += (
            "{} counters can be visited by <{}>. Their states are as follows: ".format(
                len(intersect_counters), self.name
            )
        )
        count_states = {}
        for i in intersect_counters:
            obj_i = "nothing"
            if counter_states[i] != " ":
                obj_i = counter_states[i]
            if obj_i in count_states:
                count_states[obj_i] += 1
            else:
                count_states[obj_i] = 1
        total_obj = self.mdp.default_ingredients
        for i in count_states:
            if i == "nothing":
                continue
            kitchen_state_prompt += (
                f"{count_states[i]} counters have {i} which you can pick it up. "
            )
        if len(list(count_states.keys())) == 1:
            kitchen_state_prompt += "counters have nothing."
        # for i in total_obj:
        # 	if i not in count_states:
        # 		kitchen_state_prompt += f'No counters have {i}. '
        kitchen_state_prompt += "\n"

        wong_message_prompt = (
            state.error_message[0] if len(state.error_message) > 0 else ""
        )
        # add failed history into state prompt:
        long_term_memory_prompt = f"Successful Action History: {self.teammate.teammate_ml_actions if len(self.teammate.teammate_ml_actions)<=5 else self.teammate.teammate_ml_actions[-4:]}\n"
        reflection_memory = "Lessons from Past Failures\n"
        count = len(reflection_memory)
        temp_history = (
            self.failed_history
            if len(reflection_memory) < 3
            else self.failed_history[-3:]
        )
        if len(self.failed_history) != 0:
            for index, value in enumerate(temp_history):
                reflection = ""
                if value["reflection_content"][-1] == "\n":
                    reflection = value["reflection_content"][:-1]
                else:
                    reflection = f'{index+1}.{value["reflection_content"]}'
                reflection_memory += reflection + "\n"
        if count == len(reflection_memory):
            reflection_memory += "[]\n"
        self.planner.layout = self.layout_prompt
        self.planner.wong_message_prompt = wong_message_prompt
        self.planner.long_term_memory_prompt = reflection_memory
        # 	teammate_state_prompt = ""
        return (
            long_term_memory_prompt
            + self.planner.long_term_memory_prompt
            + self.layout_prompt
            + order_prompt
            + time_prompt
            + ego_state_prompt
            + action_state_prompt
            + teammate_state_prompt
            + teammate_action_state_prompt
            + kitchen_state_prompt
        )

    def build_access_utensil(self, state):
        am = self.mlam
        index_list = [0, 1]
        if self.mdp.utensil_list_chef == [] or self.mdp.utensil_list_assist == []:
            for i in index_list:
                player = state.players[i]
                for utensil in self.mdp.utensil_list:
                    motion_goals = am.go_to_utensil_actions(state, utensil, i)
                    motion_goals = [
                        mg
                        for mg in motion_goals
                        if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                            player.pos_and_or, mg
                        )
                    ]
                    if len(motion_goals) > 0:
                        if i == 0:
                            self.mdp.utensil_list_chef.append(utensil)
                        else:
                            self.mdp.utensil_list_assist.append(utensil)
        else:
            return

    ##################
    """
	The followings are the Planner part
	"""
    ##################

    def action(self, state):
        self.state = state
        self.build_access_utensil(state)
        turn_statistics_dict_cp = copy.deepcopy(turn_statistics_dict)
        self.turn_statistics_dict = turn_statistics_dict_cp
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        # only use to record the teammate ml_action,
        # if teammate finish ml_action in t-1, it will record in s_t,
        # otherwise, s_t will just record None,
        # and we here check this information and store it
        self.current_timestep = state.timestep
        self.planner.current_timestep = state.timestep
        self.teammate.order = state.current_k_order[0]
        self.order = state.current_k_order[0]
        self.change_communication_role("ask", "answer")
        self.planner.dialog_history_list = []
        self.teammate.planner.dialog_history_list = []
        # check if the teammate has finsihed his action
        if self.teammate.current_ml_action_steps > 0:
            current_ml_action_done = self.teammate.check_current_ml_action_done(state)
            if current_ml_action_done:
                self.teammate.current_ml_action = None

        if state.ml_actions[1 - self.agent_index] != None:
            self.teammate_ml_actions.append(
                {
                    "timestamp": self.current_timestep,
                    "action": state.ml_actions[1 - self.agent_index],
                }
            )

        # if current ml action does not exist, generate a new one
        if self.current_ml_action is None:
            self.current_ml_action = self.generate_ml_action(state)

        # when "wait" and has other action in action_wait_parse ,replace wait as the action
        if "wait" in self.current_ml_action and not self.action_wait_parse.empty():
            self.current_ml_action = self.generate_ml_action(state)

        # if the current ml action is in process, Player{self.agent_index} done, else generate a new one
        if self.current_ml_action_steps > 0:
            current_ml_action_done = self.check_current_ml_action_done(state)
            if current_ml_action_done:
                # generate a new ml action
                self.generate_success_feedback(state)
                self.current_ml_action = None
                self.current_ml_action = self.generate_ml_action(state)
        count = 0
        if self.current_ml_action_steps == 0:
            self.failed_message = self.validate_current_ml_action(state)
            if "success" in self.failed_message and self.test_mode:
                self.test_ml_action.popleft()
            # only try to self-correct 1 time
            while "success" not in self.failed_message:
                if self.test_mode:
                    self.current_ml_action = "wait(1)"
                    self.time_to_wait = 1
                    print(self.failed_message)
                else:
                    # del all wait parse action
                    while not self.action_wait_parse.empty():
                        self.action_wait_parse.get()
                    if count >= 1:
                        self.current_ml_action = "wait(1)"
                        self.time_to_wait = 1
                        break
                    self.trace = False
                    self.generate_failure_feedback(
                        self.current_ml_action, self.failed_message
                    )
                    self.change_correct_prompt()
                    self.turn_statistics_dict["statistical_data"]["error"][
                        self.agent_index
                    ]["validator_error"]["error_num"] += 1
                    self.turn_statistics_dict["statistical_data"]["error"][
                        self.agent_index
                    ]["validator_error"]["error_message"].append(self.failed_message)
                    self.current_ml_action = self.generate_ml_action(state)
                    count += 1
                self.failed_message = self.validate_current_ml_action(state)
        # generate rethink if the problem is solved and not just 'wait(1)'
        if not self.trace and count < 1:
            self.turn_statistics_dict["statistical_data"]["error_correction"][
                self.agent_index
            ]["validator_correction"]["correction_num"] += 1
            if self.current_timestep not in self.error_correct.keys():
                self.generate_rethink(state)

        # print(self.current_ml_action_steps)
        # print(self.time_to_wait)
        # print(self.check_current_ml_action_done(state))
        # print(f"{self.name} current action:{self.current_ml_action}")
        # assert(self.current_ml_action_steps<5)
        # statistic
        self.turn_statistics_dict["content"]["reflection"][self.agent_index] = (
            copy.deepcopy(self.failed_history)
        )
        action_list = []
        actions = list(self.action_wait_parse.queue)
        if self.current_ml_action is not None:
            action_list.append(self.current_ml_action)
        for index, a in enumerate(actions):
            action_list.append(f"{a}")
        self.turn_statistics_dict["content"]["action_list"][self.agent_index] = (
            copy.deepcopy(action_list)
        )
        self.turn_statistics_dict["timestamp"] = self.current_timestep
        self.turn_statistics_dict["order_list"] = state.current_k_order
        self.turn_statistics_dict["actions"].append(self.current_ml_action)
        # save statistic data
        # del history
        self.planner.dialog_history_list = []
        self.teammate.planner.dialog_history_list = []

        self.trace = True

        # if "deliver_soup()" in self.current_ml_action:
        # 	self.teammate.teammate_ml_actions.append({'timestamp':self.current_timestep,'action':"deliver_soup()"})

        if "wait" in self.current_ml_action or "recipe" in self.current_ml_action:
            self.current_ml_action_steps += 1
            self.time_to_wait -= 1
            lis_actions = self.mdp.get_valid_actions(state.players[self.agent_index])
            # chosen_action =lis_actions[np.random.randint(0,len(lis_actions))]
            chosen_action = (0, 0)
            if pkg_resources.get_distribution("overcooked_ai").version == "1.1.0":
                self.prev_state = state
                return chosen_action, {}
            elif pkg_resources.get_distribution("overcooked_ai").version == "0.0.1":
                self.prev_state = state
                return chosen_action, ""
        else:
            possible_motion_goals = self.find_motion_goals(state)
            current_motion_goal, chosen_action = self.choose_motion_goal(
                start_pos_and_or, possible_motion_goals, state
            )
        # if "wait" in self.current_ml_action:
        # 	print(f'current motion goal for P{self.agent_index} is wait')
        # else:
        # 	if current_motion_goal is None:
        # 		current_motion_goal = 'None'
        # 	print(f'current motion goal for P{self.agent_index} is {current_motion_goal}')

        if self.auto_unstuck and chosen_action != Action.INTERACT:
            if self.prev_state is not None and state.players == self.prev_state.players:
                if self.agent_index == 0:
                    joint_actions = list(
                        itertools.product(Action.ALL_ACTIONS, [Action.STAY])
                    )
                elif self.agent_index == 1:
                    joint_actions = list(
                        itertools.product([Action.STAY], Action.ALL_ACTIONS)
                    )
                else:
                    raise ValueError("Player index not recognized")

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    if j_a != [Action.INTERACT, Action.STAY] and j_a != [
                        Action.STAY,
                        Action.INTERACT,
                    ]:
                        if (
                            pkg_resources.get_distribution("overcooked_ai").version
                            == "1.1.0"
                        ):
                            new_state, _ = self.mlam.mdp.get_state_transition(
                                state, j_a
                            )
                        elif (
                            pkg_resources.get_distribution("overcooked_ai").version
                            == "0.0.1"
                        ):
                            new_state, _, _ = self.mlam.mdp.get_state_transition(
                                state, j_a
                            )
                        if (
                            new_state.players_pos_and_or
                            != self.prev_state.players_pos_and_or
                        ):
                            unblocking_joint_actions.append(j_a)
                unblocking_joint_actions.append([Action.STAY, Action.STAY])
                chosen_action = unblocking_joint_actions[
                    np.random.choice(len(unblocking_joint_actions))
                ][self.agent_index]

        self.prev_state = state
        if chosen_action is None:
            self.current_ml_action = "wait(1)"
            self.time_to_wait = 1
            chosen_action = Action.STAY

        if self.current_ml_action_steps == 0:
            self.current_ml_action_steps = 1

        # print(f'ml_action = {self.current_ml_action}')
        # print(f'P{self.agent_index} : {Action.to_char(chosen_action)}')
        if pkg_resources.get_distribution("overcooked_ai").version == "1.1.0":
            return chosen_action, {}
        elif pkg_resources.get_distribution("overcooked_ai").version == "0.0.1":
            if "pickup" in self.current_ml_action:
                return chosen_action, self.parse_action_params[0]
            elif any(s in self.current_ml_action for s in self.mdp.interact_actions):
                return chosen_action, "[START]"
            else:
                return chosen_action, ""

    # Parse action as function(parms1,parms2)
    def parse_params_in_action(self, action: str, need_output=False):
        action = action.replace(" ", "")
        function_name = ""
        params = []
        pattern = (
            r"(?:\d+\.\s*)?(\w+)\s*(?:\((.*?)\))?"  # r"^\"?\d*\.?\"?\s*(\w+)\((.*)\)"
        )
        match = re.match(pattern, action)
        # function name and parmas with "()"
        if match:
            function_name = match.group(1)
            if match.group(2) == None:
                params = []
            else:
                params = match.group(2).split(",")
            for index, p in enumerate(params):
                params[index] = params[index].replace(" ", "")
                params[index] = params[index].replace("'", "")
                params[index] = params[index].replace('"', "")
            if len(params) > 1 and ("put_obj_in_utensil" in function_name):
                params = [params[1]]
        elif "NOTHING" in action or "nothing" in action:
            function_name = "wait"
            params = ["1"]
        # function name and parmas without "()"
        else:
            pattern = r"^\"?\d*\.?\"?\s*(\w+)\((.*)\)"
            if need_output:
                print(f"No match function like plan in {action}")
            return function_name, params
        if need_output:
            print(f"function_name : {function_name},params_name:{params}")
        return function_name, params

    def load_recipe(self):
        if self.order in self.recipe.keys():
            warnings.warn("Prompt has load the recipe")
            return
        recipe_name_list = os.listdir(PROMPT_DIR + "/recipe/")
        recipe_filename = ""
        for r in recipe_name_list:
            r_name = r[2:-4]
            if self.order == r_name:
                recipe_filename = r
                break
        with open(PROMPT_DIR + "/recipe/" + recipe_filename) as r:
            self.recipe[self.order] = r.read()

    def parse_ml_action(self, action_string):
        ml_action = ""
        action, params = self.parse_params_in_action(action_string)
        self.parse_action = action
        self.parse_action_params = params
        # compare params with lower objects list
        for index, p in enumerate(self.parse_action_params):
            for u in self.mdp.utensil_list:
                if p == u.lower():
                    self.parse_action_params[index] = u
            for i in self.mdp.default_ingredients:
                if p == i.lower():
                    self.parse_action_params[index] = i
        # parse function like return false,meaning the return plan is a sentence or other wrong content.
        if self.parse_action == "":
            print(action_string)
            print(
                "please check the plan,the plan does not follow the format of function"
            )
            return (
                False,
                f"please check your plan,make sure your plan is a sequence of action seperated by ';' in one line, without any description or serial number.",
            )
        if "place_obj_on_counter()" == action_string:
            ml_action = f"place_obj_on_counter()"
        elif "pickup(" in action_string:
            if len(params) != 2:
                return (
                    False,
                    f"Wrong pickup() params. It should have 2 params: obj and distination.",
                )
            # check pickup from dispenser, and pickup from utensil, pick up from counter
            # check whether the item is in the recipe list
            if params[0] not in self.mdp.all_ingredients + ["dish"]:
                return (
                    False,
                    f"Wrong pickup() params {params[0]}. It does not belong to any recipe ingredients or dish.",
                )
            if (
                (params[1] not in self.mdp.utensil_list)
                and ("counter" not in params[1])
                and ("dispenser" not in params[1])
            ):
                return (
                    False,
                    f"Wrong pickup() params {params[1]}. It does not belong to any utensils, dispenser or counter.",
                )
            if ("dispenser" in params[1]) and (
                params[0] not in self.mdp.default_ingredients + ["dish"]
            ):
                return (
                    False,
                    f"Wrong pickup() params {params[0]},{params[1]}. You can only get raw ingredients and dish from dispenser.",
                )
            # in case of pickup(dish,ingredient_dispenser) , pickup(ingredient,dish_dispenser)
            if ("dish" in params[0]) and ("ingredient_dispenser" in params[1]):
                return (
                    False,
                    f"Wrong pickup() params {params[0]},{params[1]}. You can only get dish from dish_dispenser.",
                )
            if "dish" not in params[0] and "dish_dispenser" in params[1]:
                return (
                    False,
                    f"Wrong pickup() params {params[0]},{params[1]}. You can only get ingredient from ingredient_dispenser.",
                )
            ml_action = f"pickup({params[0]},{params[1]})"
        elif "put_obj_in_utensil(" in action_string:
            if len(params) != 1:
                return (
                    False,
                    f"Wrong put_obj_in_utensil() params. It should have 1 params: utensil.",
                )
            # check if LLM generate put_obj_in_utensil(obj,utensil),then only use the utensil params
            if params[0] in self.mdp.utensil_list:
                ml_action = f"put_obj_in_utensil({params[0]})"
            else:
                return False, f"Wrong put_obj_in_utensil() parmas:{params[0]}"
        elif "fill_dish_with_food(" in action_string:
            if len(params) != 1:
                return (
                    False,
                    f"Wrong fill_dish_with_food() params. It should have 1 params: utensil.",
                )
            if params[0] in self.mdp.utensil_list:
                ml_action = f"fill_dish_with_food({params[0]})"
            else:
                return False, f"Wrong fill_dish_with_food() parmas:{params[0]}"
        elif "deliver_soup()" == action_string:
            ml_action = "deliver_soup()"
        elif "check_recipe()" == action_string:
            ml_action = "check_recipe()"
        # 	check all the action need to interact with utensils
        elif any(item in action_string for item in self.mdp.interact_actions.keys()):
            if len(params) != 1:
                return (
                    False,
                    f"please check your plan,make sure your plan is a sequence of action seperated by ';' in one line, without any description or serial number.",
                )
            for interact_action, utensils in self.mdp.interact_actions.items():
                if interact_action in action_string:
                    if params[0] in utensils:
                        ml_action = f"{interact_action}({params[0]})"
                    else:
                        return False, f"Wrong {interact_action}() parmas:{params[0]}"
        elif "wait" in action_string:
            time_to_wait = self.parse_wait_string(action_string)
            if time_to_wait > 5:
                return False, f"Too long wait time. It should be less than 5."
            else:
                self.time_to_wait = time_to_wait
                ml_action = action_string
        else:
            print(action_string)
            print(
                "please check the plan,the plan does not follow the format of function"
            )
            return (
                False,
                f"please check your plan,make sure your plan is a sequence of action seperated by ';' in one line, without any description or serial number.",
            )

        return True, ml_action

    def team_index_str(self):
        return str(1 - self.agent_index)

    def parse_wait_string(self, s):
        # Check if it's just "wait"
        if s == "wait":
            return 1

        # Remove 'wait' and other characters from the string
        s = (
            s.replace("wait", "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace(".", "")
        )

        # If it's a number, return it as an integer
        if s.isdigit():
            return int(s)

        # If it's not a number, return a default value or raise an exception
        return 1

    def change_communication_role(self, role, team_role):
        # self
        self.communication_role = role
        a = self.load_prompt_file()
        b = self.teammate.load_prompt_file()

    def change_correct_prompt(self):
        self.load_prompt_file(mode="correct")

    def name(self):
        return "Player " + str(self.agent_index)

    def team_name(self):
        return "Player " + self.team_index_str()

    def message_formate_control(
        self, role: str, turn_dialog_self: list, turn_dialog_team: list
    ):
        pre_message = ""
        answer_num = 1
        analysis_num = 1
        flag = False
        # turn 0
        # pre_message += f'''{self.name} analysis history turn 0 : [EMPTY]\n'''
        ##As asker, generate message for self
        # <example>:
        # Order:Onion_soup(No recipe)
        # Scene 0: Chef holds nothing. Assistant holds nothing. Assistant's action:None.Kitchen states: Chef space:Pot0 empty.Pot1 empty.chopping_board1 empty.Stirrer empty. Assistant space:chopping_board0 empty.Cooker empty.Counter:empty.
        # Assistant analysis history turn 0 : [EMPTY]
        # Assistant analysis history turn 1 : I do not know what to do. I should ask chef.
        # Assistant say turn 1 : Can you give me advice?
        # Chef  say turn 1 : You should wait.<END>
        # Assistant analysis : I need to wait.
        # Assistant say : [NOTHING]
        # Assistant plan: wait(1)
        # </example>
        if role == "asker":
            for t in turn_dialog_self:
                # if t["role"] == "analysis":
                # 	pre_message +=  f'''{self.name} analysis history turn {analysis_num} : {t['content']}\n'''  #"L: Analysis:" + t["content"]+"\n"
                if t["role"] == "talk":
                    pre_message += f"""{self.name} say history turn {analysis_num} : {t['content']}\n"""
                    # search teammate response
                    for r in turn_dialog_team:
                        if r["role"] == "talk":
                            if analysis_num == answer_num:
                                pre_message += f"""{self.teammate.name} say history turn {analysis_num} : {r['content']}\n"""
                                break
                            else:
                                answer_num += 1
                    analysis_num += 1
        ##As answer, generate message for self
        # <example>:
        # Assistant analysis history turn 0 : [EMPTY]
        # Chef say history turn 1 : Can you pick up an onion for me ?
        # Assistant analysis history turn 1: I can go the ingredient_dispenser pick an onion for chef. I do not know if it is right.
        # Assistant say history turn 1: Do you want me pick onion from ingredient_dispenser?
        # Chef say history turn 2 : YES.
        # Assistant analysis: I known i need to pick onion from ingredient_dispenser.
        # Assistant say : OK<END>
        # Assistant plan : pickup(onion,ingredient_dispenser)
        # </example>
        elif role == "answer":
            for t in turn_dialog_team:
                if t["role"] == "talk":
                    pre_message += f"""{self.teammate.name} say history turn {analysis_num} : {t['content']}\n"""
                    # search self analysis and talk
                    for r in turn_dialog_self:
                        if r["role"] == "analysis":
                            if analysis_num == answer_num:
                                # pre_message += f'''{self.name} analysis history turn {analysis_num} : {r['content']}\n'''
                                flag = True
                            else:
                                answer_num += 1
                        if r["role"] == "talk" and flag:
                            pre_message += f"""{self.name} say history turn {analysis_num} : {r['content']}\n"""
                            flag = False
                            break
                    analysis_num += 1
        else:
            raise ValueError("Wrong actor for build communication message!")
        return pre_message

    def communication(self, message, state):
        # [observation,Analysis \n Player 0:talk]
        self.end_talk = False
        you_response = message
        team_response = ""
        communication_turn = 0
        last_message = ""

        print("\n\n>>>>>>>>>>>>>>>>>>Begin communication<<<<<<<<<<<<<\n")
        while self.end_talk is False:
            communication_turn += 1
            print(f"Input for {self.teammate.name}" + ":\n")
            format_you_response = self.teammate.message_formate_control(
                "answer",
                self.teammate.planner.dialog_history_list,
                self.planner.dialog_history_list,
            )
            self.teammate.state_prompt = self.teammate.generate_state_prompt(state)
            print(
                self.teammate.state_prompt
                + format_you_response
                + self.teammate.planner.wong_message_prompt
            )
            # teammate must reply
            self.end_talk, team_response = self.teammate.answer(
                format_you_response, "team", state
            )
            print(f"Answer of {self.teammate.name}" + ":\n")
            print(team_response + "\n\n")

            print(f"Input for {self.name}" + ":\n")
            format_team_response = self.message_formate_control(
                "asker",
                self.planner.dialog_history_list,
                self.teammate.planner.dialog_history_list,
            )
            self.state_prompt = self.generate_state_prompt(state)
            print(
                self.state_prompt
                + format_team_response
                + self.planner.wong_message_prompt
            )
            last_message = format_team_response + "\n\n"
            self.pre_message = last_message

            self.end_talk, you_response = self.answer(
                format_team_response, "you", state
            )
            if self.end_talk is False:
                print(f"Answer of {self.name}" + ":\n")
            else:
                print(
                    f">>>>>>>>>>>>>>>>>>>>{self.name} decide to make action:<<<<<<<<<<<<<<<<<\n"
                )
            print(you_response + "\n\n\n")
        print("\n\n>>>>>>>>>>>>>>>>>>Finish communication<<<<<<<<<<<<<\n")
        # parse
        plan = self.parse_response(you_response, "plan")
        if plan == "":
            print("You have not make plan last time.\n")
            plan, _ = self.important_part_no_create(1, "plan", you_response)
        return plan

    def important_part_no_create(self, retry_num, part_type, response):
        # Analsysis did not generate error handling
        prompt = (
            "\n\nYou did not create correct "
            + part_type
            + " part last time, now please remember to add "
            + part_type
            + " according to the format of example strictly!Below is the history:<BEGAIN>\n"
        )
        retry_num_max = retry_num
        final_response = response
        while retry_num > 0:
            # retry query
            self.planner.current_user_message = {
                "role": "user",
                "content": prompt
                + self.planner.current_user_message["content"]
                + "\n\nYour last response is :"
                + response
                + "\n\n<END>Now please return correct answer with your loss part.",
            }
            # print(self.planner.current_user_message)
            response, correction_tokens = self.planner.query(
                key=self.openai_api_key(), proxy=self.proxy, stop="Scene", trace=True
            )
            # statistic
            self.turn_statistics_dict["statistical_data"]["error"][self.agent_index][
                "format_error"
            ]["error_num"] += 1
            self.turn_statistics_dict["statistical_data"]["error"][self.agent_index][
                "format_error"
            ]["error_message"].append(self.planner.current_user_message["content"])

            # print(response)
            parse_result = self.parse_response(response, part_type)
            retry_num -= 1
            final_response = response
            response = "\n\nYour last response is :" + response + "\n\n"
            if retry_num == 0 and parse_result == "":
                warnings.warn(
                    "Failed to create "
                    + part_type
                    + " for "
                    + str(retry_num_max)
                    + " times"
                )
                parse_result = "You do not have " + part_type + " last time."
            if parse_result != "":
                # correction_num means the number of truly correct the format error ,maybe a success need several trys.
                # correction_tokens means all the tokens for trying, so len(correction_tokens)>= correction_num
                self.turn_statistics_dict["statistical_data"]["error_correction"][
                    self.agent_index
                ]["format_correction"]["correction_num"] += 1
                self.turn_statistics_dict["statistical_data"]["error_correction"][
                    self.agent_index
                ]["format_correction"]["correction_tokens"].append(correction_tokens)
                break
        return parse_result, final_response

    # It is used to give organized messages to gpt and organize the generated analysis, talk and other information
    def answer(self, message, role, state):
        self.state_prompt = self.generate_state_prompt(state)
        self.planner.current_user_message = {
            "role": "user",
            "content": self.state_prompt + message,
        }
        response, tokens_num = self.planner.query(
            key=self.openai_api_key(), proxy=self.proxy, stop="Scene", trace=True
        )
        parse_analysis = self.parse_response(response, "analysis")
        # Analsysis did not generate error handling:
        if parse_analysis == "":
            print("Do not create analysis", response)
            # print(self.planner.current_user_message["content"])
            parse_analysis, _ = self.important_part_no_create(1, "analysis", message)

        # Agent1 has decided to stop talking
        # if ~self.end_talk:
        parse_talk, end_talk = self.parse_response(response, "talk")
        self.planner.dialog_history_list.append(
            {"role": "analysis", "content": parse_analysis}
        )

        # statistic
        if role == "you":
            communication_index = self.agent_index
            self.turn_statistics_dict["statistical_data"]["communication"][
                communication_index
            ]["turn"].append(parse_talk)
            self.turn_statistics_dict["statistical_data"]["communication"][
                communication_index
            ]["token"].append(tokens_num)
            self.turn_statistics_dict["content"]["content"][communication_index].append(
                {
                    "agent": self.agent_index,
                    "analysis": self.parse_response(response, "analysis"),
                    "say": parse_talk,
                    "plan": self.parse_response(response, "plan"),
                }
            )
        else:
            communication_index = self.teammate.agent_index
            self.teammate.turn_statistics_dict["statistical_data"]["communication"][
                communication_index
            ]["turn"].append(parse_talk)
            self.teammate.turn_statistics_dict["statistical_data"]["communication"][
                communication_index
            ]["token"].append(tokens_num)
            self.teammate.turn_statistics_dict["content"]["content"][
                communication_index
            ].append(
                {
                    "agent": self.agent_index,
                    "analysis": self.parse_response(response, "analysis"),
                    "say": parse_talk,
                    "plan": self.parse_response(response, "plan"),
                }
            )

        if parse_talk == "":
            parse_talk, response = self.important_part_no_create(1, "talk", response)
        parse_talk, end_talk = self.parse_response(response, "talk")
        if parse_talk == "[NOTHING]" and end_talk:
            parse_talk = "OK.<END>"
        # check if agent repeat same say

        for d in self.planner.dialog_history_list:
            # embedding for every dialog of agent
            if d["role"] == "talk":
                if if_two_sentence_similar_meaning(
                    self.openai_api_key(), self.proxy, d["content"], parse_talk
                ):
                    return True, response
        self.planner.dialog_history_list.append({"role": "talk", "content": parse_talk})

        # check if there is action
        # nothing did not produce an action
        new_plan = self.parse_response(response, "plan")
        pattern = r"(?i)nothing"
        matches = re.findall(pattern, new_plan)
        if matches:
            return end_talk, response

        new_plan = self.parse_ml_action_top(new_plan, False)
        # When correct, new action should also replace the old action list.
        if not self.trace:
            while not self.action_wait_parse.empty():
                self.action_wait_parse.get()

            for index, a in enumerate(new_plan):
                if index > 0:
                    self.action_wait_parse.put(a)
            v = list(self.action_wait_parse.queue)
            # rprint(f"[red][CORRECT][/red]Generate new correct action list <{v}>\n")
            return end_talk, response
        # new_plan : list
        # Only when the number of action to do is 2, replace the last one by the action generated in communication
        # If the number of action to do = 1: self.action_wait_parse.qsize() = 0 ,add it to the sequence
        # If the number of action to do >2 : new generated action may not follow the timestep in the queue actions, it will perform badly.

        if self.action_wait_parse.qsize() == 0:
            v = new_plan
            print(f"\nGenerate new action list <{v}>\n")
            for p in new_plan:
                self.action_wait_parse.put(p)
                # rprint(f"[green][ADD][/green]:Add new plan {p}\n")
        elif self.action_wait_parse.qsize() >= 1:
            pass
            rprint(
                f"[yellow][ADD][/yellow]:Current action are too much. Does not add <{new_plan}> in queue\n"
            )

        return end_talk, response

    # Extract the analysis and TALK from the GPT reply.
    # mode:analysis,<TALK>
    def parse_response(self, response, mode, need_correct=False):
        if self.agent_index == 0:
            role = "Chef"
        else:
            role = "Assistant"
        if mode == "analysis":
            pattern = f"(?:{role})\s+analysis\s*:?\s*(.*?)\s*(?:{role})\s+plan:"
            match = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match[0]
            if need_correct:
                response, _ = self.important_part_no_create(1, "analysis", response)
            else:
                response = ""
            return response
        elif mode == "talk":
            pattern = f"(?:{role})\s+say\s*:?\s*(.*?)(?:\s+|\n+)?$"
            match = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                if "[NOTHING]" in match[0]:
                    return "[NOTHING]", True
                pattern = r"(.*?)(END)"
                end_match = (
                    True
                    if re.findall(pattern, match[0], re.DOTALL | re.IGNORECASE)
                    else False
                )
                return match[0], end_match
            else:
                # response = self.important_part_no_create(1,"talk",response)
                return "[NOTHING]", True
        elif mode == "plan":
            pattern = f"({role})\s+plan\s*:?\s*(.*?)[\s\n]*(?=\s+({role})\s+say:)\s*"
            match = re.search(pattern, response)
            if match:
                # In case of generating output like: "Chef plan: a sentence
                temp_action = match.group(0).replace("\n", "").strip().split(";")
                first_action = temp_action[0].split(":")
                if len(first_action) == 2:
                    action, params = self.parse_params_in_action(first_action[1], False)
                    # max length of action is smaller than 30.
                    if action != "" and len(action) < 30:
                        return match.group(0).strip()
            if need_correct:
                response, _ = self.important_part_no_create(1, "plan", response)
            else:
                response = ""
            return response
        else:
            raise KeyError("Not exist mode for parse_response()")

    def generate_rethink(self, state):
        extracted_text = ""
        failture = ""
        message = self.message_formate_control(
            "asker",
            self.planner.dialog_history_list_storage,
            self.teammate.planner.dialog_history_list_storage,
        )
        failure_message = ""
        for index, s in enumerate(self.planner.dialog_history_list):
            if s["role"] == "failure_explanation":
                # Multi failed message
                if failure_message != "":
                    failure_message += (
                        f"When you handile the error above, {s['content']}"
                    )
                else:
                    failure_message = "Failed Reason:" + s["content"]
                    failture = s["content"]
        success_message = f"Solution: Your action '{self.current_ml_action}' successes at solving the problem.\n"
        self.load_prompt_file("reflection")
        self.state_prompt = self.generate_state_prompt(state)
        self.planner.current_user_message = {
            "role": "user",
            "content": self.state_prompt + message + failure_message + success_message,
        }
        print(f"rethink input content: {self.planner.current_user_message['content']}")
        response, correction_tokens = self.planner.query(
            key=self.openai_api_key(),
            proxy=self.proxy,
            stop="Scene",
            trace=self.trace,
            rethink=True,
        )
        print(f"response of rethink: {response}")
        pattern = r"Rethink:(.*?)\."
        matches = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if matches:
            extracted_text = matches.group(1).strip()
            print(f"success extract rethink content.\n{extracted_text}")
            # count
            self.turn_statistics_dict["statistical_data"]["error_correction"][
                self.agent_index
            ]["validator_correction"]["reflection_obtain"] = self.current_timestep
        else:
            print("\nFailed to create rethink.\n")
        # problem + solution
        flag = 0
        for index, reflection in enumerate(self.failed_history):
            if failture in reflection["error"]:
                self.failed_history[index]["reflection_content"] += extracted_text
                flag = 1
        # If the reflection is not duplicated and reflection successes
        if flag == 0 and matches:
            self.failed_history.append(
                {
                    "error": failture,
                    "reflection_content": extracted_text,
                    "obtain_timestamp": self.current_timestep,
                }
            )
        return

    def generate_ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
                [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of  heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        if self.test_mode:
            self.current_ml_action_steps = 0
            action = ""
            if len(self.test_ml_action) == 0:
                action = "wait(1)"
                self.time_to_wait = 1
                self.test_ml_action.append(action)
            else:
                action = self.test_ml_action[0]
                numbers = re.findall(r"\d+", action)
                if "wait" in action and numbers != []:
                    self.time_to_wait = int(numbers[0])
            return action

        # At most one error at a time
        failure_message = ""
        ml_action = ""
        if self.action_wait_parse.empty() or (not self.trace):
            state_prompt = self.generate_state_prompt(state)
            # Error checking
            for index, s in enumerate(self.planner.dialog_history_list):
                if s["role"] == "failure_explanation":
                    # Multi failed message
                    if failure_message != "":
                        failure_message += (
                            f"When you handile the error above, {s['content']}\n"
                        )
                    else:
                        failure_message = (
                            "\nBelow are the failed and analysis history about your last chosed action,you can use the information to reach correct action alone AND DO NOT COMMUNICATE WITH YOUR TEAMMATE :\n"
                            + s["content"]
                            + "\n"
                        )
            state_prompt += failure_message

            print(f"\n\n### Observation module to " + self.name + "\n")

            send_message = self.message_formate_control(
                "asker",
                self.planner.dialog_history_list,
                self.teammate.planner.dialog_history_list,
            )
            state_message = {
                "role": "user",
                "content": state_prompt
                + send_message
                + self.planner.wong_message_prompt,
            }
            self.state_prompt = state_prompt
            self.planner.current_user_message = state_message
            # print("history information")
            print(f"{state_message['content']}")
            # statistic
            self.turn_statistics_dict["content"]["observation"][self.agent_index] = (
                state_message["content"]
            )

            print(f"\n\n\n### GPT Planner module\n")
            print("====== GPT Query ======")
            response, tokens_num = self.planner.query(
                key=self.openai_api_key(),
                proxy=self.proxy,
                stop="Scene",
                trace=self.trace,
                map=self.mdp.state_string(self.state).replace("ø", "o"),
            )
            print(response)
            # check whether need communication
            # check whether has the plan
            communicate_response, _ = self.parse_response(response, "talk")
            analysis = self.parse_response(response, "analysis")
            plan = self.parse_response(response, "plan")
            if analysis == "":
                print("\n\n\n******No Analysis Part, Correcting*********\n\n\n")
                analysis, response = self.important_part_no_create(1, "plan", response)
            if plan == "":
                print("\n\n\n******No   Plan   Part, Correcting**********\n\n\n")
                plan, response = self.important_part_no_create(1, "plan", response)
            # If important keyword still miss, this timestamp is failed.
            if analysis == "" or plan == "":
                print(
                    "\n\n\nMiss important part after several retry. This timstep is failed!\n\n\n"
                )
                ml_action = "wait(1)"
                return ml_action

            self.planner.add_msg_to_dialog_history(
                {"role": "scene", "content": state_message["content"]}
            )
            self.planner.add_msg_to_dialog_history(
                {"role": "analysis", "content": analysis}
            )

            # statistic
            self.turn_statistics_dict["content"]["content"][self.agent_index].append(
                {
                    "agent": self.agent_index,
                    "analysis": self.parse_response(response, "analysis"),
                    "say": communicate_response,
                    "plan": self.parse_response(response, "plan"),
                }
            )
            # If this function call is to correct action, the tokens should count into the number of correction_tokens.
            if failure_message != "":
                self.turn_statistics_dict["statistical_data"]["error_correction"][
                    self.agent_index
                ]["validator_correction"]["correction_tokens"].append(tokens_num)
                # check if the  plan is empty
                if "plan" == "":
                    plan_parse, plan = self.important_part_no_create(
                        1, "plan", response
                    )
                    if "plan_parse" == "":
                        self.error_correct[self.current_timestep] = False
                        ml_action = "wait(1)"
                    else:
                        ml_action = self.parse_ml_action_top(plan, True)
                else:
                    ml_action = self.parse_ml_action_top(plan, True)
            elif ("[NOTHING]" not in communicate_response) and (
                "[EMPTY]" not in communicate_response
            ):
                self.planner.add_msg_to_dialog_history(
                    {"role": "talk", "content": communicate_response}
                )
                # statistic
                self.turn_statistics_dict["statistical_data"]["communication"][
                    self.agent_index
                ]["turn"].append(communicate_response)
                self.turn_statistics_dict["statistical_data"]["communication"][
                    self.agent_index
                ]["token"].append(tokens_num)
                self.turn_statistics_dict["statistical_data"]["communication"][
                    self.agent_index
                ]["call"] += 1
                response = self.communication(communicate_response, state)
            elif ("[NOTHING]" not in plan) and (plan != ""):
                ml_action = self.parse_ml_action_top(plan, True)
            else:
                # No plan and no communication content, wait
                ml_action == "wait(1)"
        else:
            temp_list = []
            while not self.action_wait_parse.empty():
                item = self.action_wait_parse.get()
                temp_list.append(item)
            for index, t in enumerate(temp_list):
                if index == 0:
                    continue
                self.action_wait_parse.put(t)
            print(f"\n\n\n### Already have plan in pre-communication:\n {temp_list}")
            response = f"{self.name} plan :{';'.join(temp_list)}"
        # when handle several failure, the precious communication history should be jumped
        if self.planner.dialog_history_list_storage == [] and not self.trace:
            self.planner.dialog_history_list_storage = self.planner.dialog_history_list
            self.teammate.planner.dialog_history_list_storage = (
                self.teammate.planner.dialog_history_list
            )
        self.del_dialog_history()
        if ml_action == "":
            ml_action = self.parse_ml_action_top(response, True)
        self.current_ml_action_steps = 0
        return ml_action

    # parse ml_action from  response and self correct
    def parse_ml_action_top(self, response, add_to_queue):
        # add_to_queue: if replace the old action_wait_parse with  new action list.
        print("\n===== Parser =====\n")
        # parse the sentence in plan into action_string
        if self.agent_index == 0:
            pattern = r"Chef\s+plan\s*:?\s*(.*?)(?:\s+|\n+)?$"
        elif self.agent_index == 1:
            pattern = r"Assistant\s+plan\s*:?\s*(.*?)(?:\s+|\n+)?$"
        else:
            return "wait(1)" if add_to_queue else ["wait(1)"]
        match = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        action_string = ""
        if match:
            action_string = match[0]
        else:
            print(f"Response does not follow the format rules:{response}")
            return "wait(1)" if add_to_queue else ["wait(1)"]
        # divide the action list into several action by split ';'
        action_string = action_string.lower()
        action_list = action_string.split(";")
        ml_action_list = action_list

        ml_action = ""
        for i in ml_action_list:
            if any(s in i for s in ["request", "accept", "deny", "clarify"]):
                continue
            else:
                ml_action = i
                break
        if ml_action == "":
            ml_action = "wait(1)"
            ml_action_list = ["wait(1)"]
        if add_to_queue:
            while not self.action_wait_parse.empty():
                self.action_wait_parse.get()
            for index, a in enumerate(ml_action_list):
                if index > 0:
                    self.action_wait_parse.put(a)
            if "wait" in ml_action:
                self.time_to_wait = self.parse_wait_string(ml_action)
        if "wait" not in ml_action:
            self.planner.add_msg_to_dialog_history(
                {"role": "assistant", "content": ml_action}
            )
        # print(f"{self.name}: {ml_action}")
        return ml_action if add_to_queue else ml_action_list

    ##################
    """
	The followings are the Verificator part
	"""
    ##################

    def check_current_ml_action_done(self, state):
        """
        checks if the current ml action is done
        :return: True or False
        """
        action, params = self.parse_params_in_action(self.current_ml_action)
        self.parse_action_params = params
        player = state.players[self.agent_index]
        # pot_states_dict = self.mlam.mdp.get_pot_states(state)
        if "pickup" in self.current_ml_action:
            return (
                player.has_object()
                and player.get_object().name == self.parse_action_params[0]
            )
        elif any(s in self.current_ml_action for s in self.mdp.interact_actions):
            return (
                self.parse_action_params[0]
                in self.mdp.get_utensil_states(state)["cooking"]
                or state.error_message != []
            )
        elif "place" in self.current_ml_action:
            return not player.has_object()
        elif "recpie" in self.current_ml_action:
            return self.order in self.recipe.keys()
        elif "fill" in self.current_ml_action:
            return player.held_object != None and self.order in player.held_object.name
        elif "put" in self.current_ml_action or "place" in self.current_ml_action:
            return not player.has_object()
        elif "deliver" in self.current_ml_action:
            return not player.has_object()
        elif "recipe" in self.current_ml_action:
            if self.actor == "chef":
                chef = self
                if chef.time_to_wait == 0:
                    self.load_recipe()
                else:
                    self.time_to_wait -= 1
            else:
                chef = self.teammate
            return self.order in chef.recipe
        elif "wait" in self.current_ml_action:
            return self.time_to_wait <= 0

    def check_pickup_food_dish(self, food, utensil):
        valide = True
        utensil = re.sub(r"\d+", "", utensil)
        if food in self.mdp.need_dish.keys():
            # check if the utensil is the finally step, which can not directly pick up without dish.
            for utensil_name in self.mdp.recipes.keys():
                u = self.mdp.recipes[utensil_name]
                for f in u.keys():
                    if (food == f) and (utensil == utensil_name):
                        valide = False
        return valide or not (self.mdp.need_dish[food] == 1)

    def validate_current_ml_action(self, state):
        """
        make sure the current_ml_action exists and is valid
        return: success, or failed reason.
        """
        failed_message = "success"
        if self.current_ml_action is None:
            return f"There is no action for {self.actor}.\n"
        player = state.players[self.agent_index]
        has_object = player.has_object()
        self.parse_action, self.parse_action_params = self.parse_params_in_action(
            self.current_ml_action
        )

        def valide_obj(obj):
            return player.has_object() and obj in player.get_object().name

        empty_counter = self.mdp.get_empty_counter_locations(state)
        utensil_state = self.mdp.get_utensil_states(state)
        # Action format check
        format_valide, format_error_message = self.parse_ml_action(
            self.current_ml_action
        )
        if not format_valide:
            return format_error_message
        elif "wait" in format_error_message:
            self.current_ml_action = format_error_message
            return failed_message
        # Assessing the logical Soundness of actions
        if "pickup" in self.parse_action:
            if (self.parse_action_params[1] in self.mdp.utensil_list) and (
                self.parse_action_params[1] in utensil_state["cooking"]
            ):
                failed_message = f"{self.actor} can not pick up {self.parse_action_params[0]} from {self.parse_action_params[1]}. The utensil {self.parse_action_params[1]} is cooking, and you should wait for it is ready.\n"
                return failed_message
            # In case of directly picking up cooked food which need dish in the finally step
            if not self.check_pickup_food_dish(
                self.parse_action_params[0], self.parse_action_params[1]
            ):
                failed_message = f"{self.actor} can not pick up {self.parse_action_params[0]} from {self.parse_action_params[1]}. {self.order} need a dish.\n"
                return failed_message
            if self.parse_action_params[0] in self.mdp.utensil_list:
                failed_message = f"You can not move utensil. It is fixed! If you can not access utensil, you should use counter as transfer station\n"
                return failed_message
            # check if agent access the pickup destination
            flag2 = len(self.find_motion_goals(state)) == 0
            if flag2:
                failed_message = f"{self.actor} can not reach the destination.Please check if the destination is in your space.Please check if you can interact it. You can let your teammate pick the ingredient from utensil and put in on the counter, then pick up ingredient from the counter.\n"
                if "counter" in self.parse_action_params:
                    failed_message = (
                        f"There is no {self.parse_action_params[0]} on counter {self.actor} can visited."
                        + (
                            "Assistant can directly pick ingredients from dispenser.\n"
                            if self.actor == "assistant"
                            else "If assistant is already preparing ingredient for you,you should wait.\n"
                        )
                    )
                elif (
                    "dispenser" in self.parse_action_params[1] and self.actor == "chef"
                ):
                    failed_message = f"Chef can not directly access dispenser.Chef can gain ingredients with the help of assistant."
            elif has_object:
                failed_message = f"There is object in {self.actor}'s hand, so can not pick other thing.\n"
            # check if the item is in the destination
            elif (
                (self.parse_action_params[1] in utensil_state["empty"])
                or (self.parse_action_params[1] in self.mdp.utensil_list)
                and (
                    self.parse_action_params[0]
                    != self.mdp.utensil_state_dict[self.parse_action_params[1]][
                        "soup"
                    ].state[0]
                )
            ):
                failed_message = f"There is no {self.parse_action_params[0]} in {self.parse_action_params[1]}, please check the item your want to pickup.\n"
            return failed_message + "\n"
        elif "put" in self.current_ml_action:
            # check if the food in utensil is full
            if self.parse_action_params[0] in utensil_state["cooking"]:
                failed_message = f"{self.actor} can not put obj into  {self.parse_action_params[0]}. The utensil {self.parse_action_params[0]} is cooking, and you should wait for it is ready.\n"
                return failed_message
            if self.parse_action_params[0] in utensil_state["ready"]:
                failed_message = f"{self.actor} can not put obj into  {self.parse_action_params[0]}. The utensil {self.parse_action_params[0]} is ready, and you should pick up it or fill with dish according to the recipe.\n"
                return failed_message
            # if  self.parse_action_params[0] in utensil_state['full']:
            # 	return f"The {self.parse_action_params[0]} is full. You can not put more ingredients in it.\n"

            # If the utensil is empty, add the order name TODO:maybe a bug here:If agent add ingredient not for current order?
            if self.parse_action_params[0] in utensil_state["empty"]:
                self.mdp.utensil_state_dict[self.parse_action_params[0]][
                    "order"
                ] = self.order
            flag2 = len(self.find_motion_goals(state)) == 0
            if flag2:
                failed_message = f"{self.actor} can not reach the {self.parse_action_params[0]}.Please check if you can interact it. You can put ingredient on the counter,then let your teammate pick the ingredient from counter and put it into utensil.\n"
            elif len(empty_counter) == 0:
                failed_message = f"There is no empty counter to place object.\n"
            elif (
                self.parse_action_params[0] in utensil_state["ready"]
                or self.parse_action_params[0] in utensil_state["cooking"]
            ):
                failed_message = f"{self.parse_action_params[0]} is busy. You can not add more ingredients in it.\n"
            elif not player.has_object():
                failed_message = f"There is no object in {self.actor}'s hand, so can not put it on {self.parse_action_params[0]}.\n"
            elif player.get_object().name == "dish":
                failed_message = f"You can not put dish into any utensil. Dish can only be placed on counter.\n"
            return failed_message
        elif "place_obj_on_counter" in self.current_ml_action:
            if not has_object:
                failed_message = f"There is no object in {self.actor}'s hand, so can not place object on counter.\n"
            elif len(empty_counter) == 0:
                failed_message = f"There is no empty counter to place object.\n"
            return failed_message
        elif "fill" in self.current_ml_action:
            # check if the recipe need dish
            if self.mdp.need_dish[self.order] == 0:
                failed_message = f"{self.order} does not need dish. Please directly pick cooked food from utensil and deliver it to the service location.\n"
            # fill only when utensil is cooking or ready
            elif not (
                self.parse_action_params[0] in utensil_state["ready"]
                or self.parse_action_params[0] in utensil_state["cooking"]
            ):
                failed_message = f"{self.parse_action_params[0]} is not ready or cooking for filled.\n"
            return (
                f"There is no dish in hand.\n"
                if (not valide_obj("dish") and self.mdp.need_dish[self.order] == 1)
                else failed_message
            )
        elif "deliver" in self.current_ml_action:
            has_soup = False
            for s in self.mdp.need_dish.keys():
                if valide_obj(s):
                    has_soup = True
            if not has_soup:
                if has_object:
                    return f"Item in your hand is not finished food. You should put it on counter or right utensil first. Then check whether the order need dish to fill with. If it need a dish, you should ask assistant for a dish and then you fill dish with food from utensil. If not, you can directly pick up food from utensil. Finally you can deliver soup again.\n"
                else:
                    return f"Your hand is empty. You should check whether the order need dish to fill with first. If it need a dish, you should ask assistant for a dish and then you fill dish with food from utensil. If not, you can directly pick up food from utensil. Finally you can deliver soup again.\n"
            else:
                return failed_message

        elif any(s in self.parse_action for s in self.mdp.interact_actions):
            flag2 = len(self.find_motion_goals(state)) == 0
            if flag2:
                failed_message = f"{self.actor} can not reach the {self.parse_action_params[0]}..Please check if the utensil is in your space.\n"
                return failed_message
            # only when utensil is partially_full can be operated. when Empty、ready、cooking ,utensil can not be operated.
            if self.parse_action_params[0] in utensil_state["empty"]:
                failed_message = f"Ingredients in {self.parse_action_params[0]} are not enough to begin the operation.\n"
            # check if the utensil is already cooking or ready
            elif (
                self.parse_action_params[0] in utensil_state["ready"]
                or self.parse_action_params[0] in utensil_state["cooking"]
            ):
                failed_message = (
                    f"{self.parse_action_params[0]} is busy. You can not operate it.\n"
                )
            elif has_object:
                failed_message = f"There is object in {self.actor}'s hand, so can not interact with utensil.\n"
            return failed_message
        # the same as pickup(toast,counter)
        elif "check_recipe" in self.current_ml_action:
            if self.actor != "chef":
                failed_message = (
                    f"Assistant can not check recipe.Only chef can do it.\n"
                )
            elif self.order in self.recipe.keys():
                failed_message = f"You have get the recipe before. Look at the <Recipe need to know> part.\n"
            else:
                self.time_to_wait = 2
            return failed_message
        elif "wait" in self.current_ml_action:
            match = re.search(r"\d+", self.current_ml_action)
            number = 1
            if match:
                number = match.group()
            else:
                number = 30
            return (
                failed_message
                if 0 < int(number) <= 20
                else f"Wait time is not valide.\n"
            )
        else:
            raise ValueError("Wrong action")

    def generate_success_feedback(self, state):
        success_feedback = f"### Controller Validation\n {self.name} succeeded at {self.current_ml_action}. \n"
        print(success_feedback)
        if "wait" not in success_feedback:
            self.planner.add_msg_to_dialog_history(
                {
                    "role": "user",
                    "content": f"{self.name} succeeded at {self.current_ml_action}.",
                }
            )

    def del_dialog_history(self):
        del_list = []
        for index, dialog in enumerate(self.planner.dialog_history_list):
            if any(s == dialog["role"] for s in ["talk", "analysis"]):
                del_list.append(index)
        self.planner.dialog_history_list = [
            value
            for idx, value in enumerate(self.planner.dialog_history_list)
            if idx not in del_list
        ]
        del_list = []
        for index, dialog in enumerate(self.teammate.planner.dialog_history_list):
            if any(s == dialog["role"] for s in ["talk", "analysis", "plan"]):
                del_list.append(index)
        self.teammate.planner.dialog_history_list = [
            value
            for idx, value in enumerate(self.teammate.planner.dialog_history_list)
            if idx not in del_list
        ]

    def generate_failure_feedback(self, action, failed_message):
        failure_feedback = f"Your action {action} raised an error: " + failed_message
        print(f"\n~~~~~~~~ Explainer~~~~~~~~\n{failure_feedback}")
        self.del_dialog_history()
        self.planner.add_msg_to_dialog_history(
            {"role": "failure_explanation", "content": failure_feedback}
        )

    ##################
    """
	The followings are the Controller part almost inherited from GreedyHumanModel class
	"""
    ##################

    def find_shared_counters(self, state, mlam):
        counter_dicts = query_counter_states(self.mdp, state)

        counter_list = get_intersect_counter(
            state.players_pos_and_or[self.agent_index],
            state.players_pos_and_or[1 - self.agent_index],
            self.mdp,
            self.mlam,
        )

        print("counter_list = {}".format(counter_list))
        lis = []
        for i in counter_list:
            if counter_dicts[i] == " ":
                lis.append(i)
        available_plans = mlam._get_ml_actions_for_positions(lis)
        return available_plans

    def find_motion_goals(self, state):
        """
        Generates the motion goals for the given medium level action.
        :param state:
        :return:
        """
        am = self.mlam
        motion_goals = []
        player = state.players[self.agent_index]
        pot_states_dict = self.mdp.get_pot_states(state)
        counter_objects = self.mdp.get_counter_objects_dict(
            state, list(self.mdp.terrain_pos_dict["X"])
        )
        self.parse_action, self.parse_action_params = self.parse_params_in_action(
            self.current_ml_action
        )
        # pickup dish or ingredients
        if "pick" in self.current_ml_action:
            motion_goals = am.pickup_obj_actions(
                state,
                self.parse_action_params[0],
                self.parse_action_params[1],
                self.agent_index,
                counter_objects,
            )
        elif "add_toast" in self.current_ml_action:
            motion_goals = am.pickup_obj_actions(
                state, "toast", "counter", self.agent_index, counter_objects
            )
        elif "put" in self.current_ml_action:
            motion_goals = am.go_to_utensil_actions(
                state, self.parse_action_params[0], self.agent_index
            )
        elif "place_obj_on_counter" in self.current_ml_action:
            motion_goals = self.find_shared_counters(state, self.mlam)
            if len(motion_goals) == 0:
                motion_goals = am.place_obj_on_counter_actions(state)
        elif "fill_dish_with_food" in self.current_ml_action:
            motion_goals = am.go_to_utensil_actions(
                state, self.parse_action_params[0], self.agent_index
            )
        elif "deliver_soup" in self.current_ml_action:
            motion_goals = am.deliver_soup_actions()
        elif any(s in self.parse_action for s in ["cook", "cut", "stir", "bake"]):
            motion_goals = am.go_to_utensil_actions(
                state, self.parse_action_params[0], self.agent_index
            )
        elif "wait" in self.current_ml_action:
            motion_goals = am.wait_actions(player)
        else:
            raise ValueError("Invalid action: {}".format(self.current_ml_action))

        motion_goals = [
            mg
            for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]

        return motion_goals

    def choose_motion_goal(self, start_pos_and_or, motion_goals, state=None):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.
        """

        if self.controller_mode == "new":
            (
                chosen_goal,
                chosen_goal_action,
            ) = self.get_lowest_cost_action_and_goal_new(
                start_pos_and_or, motion_goals, state
            )
        else:
            (
                chosen_goal,
                chosen_goal_action,
            ) = self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
        return chosen_goal, chosen_goal_action

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
                start_pos_and_or, goal
            )
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def get_lowest_cost_action_and_goal_new(
        self, start_pos_and_or, motion_goals, state
    ):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, plan_cost = self.real_time_planner(
                start_pos_and_or, goal, state
            )
            if plan_cost < min_cost:
                best_action = action_plan
                min_cost = plan_cost
                best_goal = goal
        if best_action is None:
            # print('\n\n\nBlocking Happend, executing default path\n\n\n')
            # print('current position = {}'.format(start_pos_and_or))
            # print('goal position = {}'.format(motion_goals))
            if np.random.rand() < 0.5:
                return None, Action.STAY
            else:
                return self.get_lowest_cost_action_and_goal(
                    start_pos_and_or, motion_goals
                )
        return best_goal, best_action

    def real_time_planner(self, start_pos_and_or, goal, state):
        terrain_matrix = {
            "matrix": copy.deepcopy(self.mlam.mdp.terrain_mtx),
            "height": len(self.mlam.mdp.terrain_mtx),
            "width": len(self.mlam.mdp.terrain_mtx[0]),
        }
        other_pos_and_or = state.players_pos_and_or[1 - self.agent_index]
        action_plan, plan_cost = find_path(
            start_pos_and_or, other_pos_and_or, goal, terrain_matrix
        )

        return action_plan, plan_cost
