import itertools, copy
import numpy as np
import re
import random
from functools import reduce
from collections import defaultdict, Counter
from overcooked_ai_py.utils import pos_distance, load_from_json, OvercookedException, classproperty
from overcooked_ai_py.data.layouts import read_layout_dict
from overcooked_ai_py.mdp.actions import Action, Direction

FULL = 1
PARTIALLY_FULL = 2
WRONG = 3


class Recipe:

    def __init__(self, ingredients, recipe_config):
        self.recipe_config = recipe_config['recipes']
        self.recipe_name = self.get_all_recipe_names(self.recipe_config) + ingredients
        self.recipe_list = self.get_all_recipe(self.recipe_config)
        self.recipe_need_dish = self.get_all_recipe_need_dish(recipe_config['need_dish'])

    def get_all_recipe_names(self, recipes):
        recipe_names = []
        for category in recipes.values():
            for recipe_name in category.keys():
                recipe_names.append(recipe_name)
        return recipe_names
    
    def get_all_recipe(self, recipes):
        recipe_list = {}
        for category in recipes.values():
            recipe_list.update(category)
        return recipe_list
    
    def get_all_recipe_need_dish(self, recipe_need_dish):
        recipe_names = []
        for name in recipe_need_dish.keys():
            if recipe_need_dish[name] != 0:
                recipe_names.append(name)
        return recipe_names



class ObjectState(object):
    """
    State of an object in OvercookedGridworld.
    """

    def __init__(self, name, catagory, position, state=None):
        """
        name (str): The name of the object
        position (int, int): Tuple for the current location of the object.
        state (tuple or None):  
            Extra information about the object. Is None for all objects 
            except soups, for which `state` is a tuple:
            (soup_type, num_items, cook_time)
            where cook_time is how long the soup has been cooking for.
        """
        self.name = name

        assert catagory in ['ingredient', 'dish', 'semi-finished', 'finished', 'with_dish'], 'Wrong Object Catatory'
        self.catagory = catagory
        self.position = tuple(position)
        if name == 'soup':
            assert len(state) == 3
        self.state = None if state is None else tuple(state)

    def is_valid(self):
        
        if self.name in ['onion', 'tomato', 'dish', 'toast', 'egg']:
            return self.state is None
        elif self.name == 'soup':
            soup_type, num_items, cook_time = self.state

            valid_item_num = (1 <= num_items <= 3)
            valid_cook_time = (0 <= cook_time)
            return valid_item_num and valid_cook_time # and valid_soup_type
        # Unrecognized object
        return False

    def deepcopy(self):
        return ObjectState(self.name, self.catagory, self.position, self.state)

    def __eq__(self, other):
        return isinstance(other, ObjectState) and \
            self.name == other.name and \
            self.position == other.position and \
            self.state == other.state

    def __hash__(self):
        return hash((self.name, self.position, self.state))

    def __repr__(self):
        if self.state is None:
            return '{}@{}'.format(self.name, self.position)
        return '{}@{} with state {}'.format(
            self.name, self.position, str(self.state))

    def to_dict(self):
        return {
            "name": self.name,
            "position": self.position,
            "state": self.state
        }

    @staticmethod
    def from_dict(obj_dict):
        obj_dict = copy.deepcopy(obj_dict)
        return ObjectState(**obj_dict)
    




class PlayerState(object):
    """
    State of a player in OvercookedGridworld.

    position: (x, y) tuple representing the player's location.
    orientation: Direction.NORTH/SOUTH/EAST/WEST representing orientation.
    held_object: ObjectState representing the object held by the player, or
                 None if there is no such object.
    """
    def __init__(self, position, orientation, held_object=None):
        self.position = tuple(position)
        self.orientation = tuple(orientation)
        self.held_object = held_object

        assert self.orientation in Direction.ALL_DIRECTIONS
        if self.held_object is not None:
            assert isinstance(self.held_object, ObjectState)
            assert self.held_object.position == self.position

    @property
    def pos_and_or(self):
        return (self.position, self.orientation)

    def has_object(self):
        return self.held_object is not None

    def get_object(self):
        assert self.has_object()
        return self.held_object

    def set_object(self, obj):
        assert not self.has_object()
        obj.position = self.position
        self.held_object = obj
 
    def remove_object(self):
        assert self.has_object()
        obj = self.held_object
        self.held_object = None
        return obj
    
    def update_pos_and_or(self, new_position, new_orientation):
        self.position = new_position
        self.orientation = new_orientation
        if self.has_object():
            self.get_object().position = new_position

    def deepcopy(self):
        new_obj = None if self.held_object is None else self.held_object.deepcopy()
        return PlayerState(self.position, self.orientation, new_obj)

    def __eq__(self, other):
        return isinstance(other, PlayerState) and \
            self.position == other.position and \
            self.orientation == other.orientation and \
            self.held_object == other.held_object

    def __hash__(self):
        return hash((self.position, self.orientation, self.held_object))

    def __repr__(self):
        return '{} facing {} holding {}'.format(
            self.position, self.orientation, str(self.held_object))
    
    def to_dict(self):
        return {
            "position": self.position,
            "orientation": self.orientation,
            "held_object": self.held_object.to_dict() if self.held_object is not None else None
        }

    @staticmethod
    def from_dict(player_dict):
        player_dict = copy.deepcopy(player_dict)
        held_obj = player_dict["held_object"]
        if held_obj is not None:
            player_dict["held_object"] = ObjectState.from_dict(held_obj)
        return PlayerState(**player_dict)


class OvercookedState(object):
    """A state in OvercookedGridworld."""
    def __init__(self, players, objects, order_list, timestep=0):
        """
        players: List of PlayerStates (order corresponds to player indices).
        objects: Dictionary mapping positions (x, y) to ObjectStates. 
                 NOTE: Does NOT include objects held by players (they are in 
                 the PlayerState objects).
        order_list: Current orders to be delivered
        timestep (int):  The current timestep of the state

        NOTE: Does not contain time left, which is handled from the environment side.
        """
    
        for pos, obj in objects.items():
            assert obj.position == pos
        self.players = tuple(players)
        self.objects = objects
        self.order_list = order_list
        self.timestep = timestep # The current timestep of the state
        self.ml_actions = [None, None] # I add to restore the ml_actions in t-1
        self.communicate_history = []
        self.error_message = []

        """
        Temp
        """
        self.k_order = 3

    @property
    def get_communicate_history(self):
        return self.communicate_history
    
    def add_communicate_history(self, res):
        return self.communicate_history.append(res)
    
    @property
    def player_positions(self):
        return tuple([player.position for player in self.players])

    @property
    def player_orientations(self):
        return tuple([player.orientation for player in self.players])

    @property
    def players_pos_and_or(self):
        """Returns a ((pos1, or1), (pos2, or2)) tuple"""
        return tuple(zip(*[self.player_positions, self.player_orientations]))

    @property
    def unowned_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, NOT including
        ones held by players.
        """
        objects_by_type = defaultdict(list)
        for pos, obj in self.objects.items():
            objects_by_type[obj.name].append(obj)
        return objects_by_type

    @property
    def player_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects held by players.
        """
        player_objects = defaultdict(list)
        for player in self.players:
            if player.has_object():
                player_obj = player.get_object()
                player_objects[player_obj.name].append(player_obj)
        return player_objects

    @property
    def all_objects_by_type(self):
        """
        Returns dictionary of (obj_name: ObjState)
        for all objects in the environment, including
        ones held by players.
        """
        all_objs_by_type = self.unowned_objects_by_type.copy()
        all_objs_by_type.update(self.player_objects_by_type)
        return all_objs_by_type

    @property
    def all_objects_list(self):
        all_objects_lists = list(self.all_objects_by_type.values()) + [[], []]
        return reduce(lambda x, y: x + y, all_objects_lists)

    @property
    def curr_order(self):
        return "any" if self.order_list is None else self.order_list[0]

    @property
    def next_order(self):
        return "any" if self.order_list is None else self.order_list[1]

    @property
    def current_k_order(self):
        return "any" if self.order_list is None else self.order_list[0:self.k_order]

    @property
    def num_orders_remaining(self):
        return np.Inf if self.order_list is None else len(self.order_list)

    def has_object(self, pos):
        return pos in self.objects

    def get_object(self, pos):
        assert self.has_object(pos)
        return self.objects[pos]

    def add_object(self, obj, pos=None):
        if pos is None:
            pos = obj.position

        assert not self.has_object(pos)
        obj.position = pos
        self.objects[pos] = obj

    def remove_object(self, pos):
        assert self.has_object(pos)
        obj = self.objects[pos]
        del self.objects[pos]
        return obj

    def fill_order_list(self, order_list, order_probability):
        pass

    @staticmethod
    def from_players_pos_and_or(players_pos_and_or, order_list):
        """
        Make a dummy OvercookedState with no objects based on the passed in player
        positions and orientations and order list
        """

        return OvercookedState(
            [PlayerState(*player_pos_and_or) for player_pos_and_or in players_pos_and_or], 
            objects={}, order_list=order_list)

    @staticmethod
    def from_player_positions(player_positions, order_list):
        """
        Make a dummy OvercookedState with no objects and with players facing
        North based on the passed in player positions and order list
        """
        dummy_pos_and_or = [(pos, Direction.NORTH) for pos in player_positions]
        return OvercookedState.from_players_pos_and_or(dummy_pos_and_or, order_list)

    def deepcopy(self):
        return OvercookedState(
            [player.deepcopy() for player in self.players],
            {pos:obj.deepcopy() for pos, obj in self.objects.items()}, 
            None if self.order_list is None else list(self.order_list),
            timestep=self.timestep)

    def __eq__(self, other):
        order_list_equal = type(self.order_list) == type(other.order_list) and \
            ((self.order_list is None and other.order_list is None) or \
            (type(self.order_list) is list and np.array_equal(self.order_list, other.order_list)))

        return isinstance(other, OvercookedState) and \
            self.players == other.players and \
            set(self.objects.items()) == set(other.objects.items()) and \
            order_list_equal

    def __hash__(self):
        return hash(
            (self.players, tuple(self.objects.values()), tuple(self.order_list))
        )

    def __str__(self):
        return 'Players: {}, Objects: {}, Order list: {}'.format( 
            str(self.players), str(list(self.objects.values())), str(self.order_list))

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "order_list": self.order_list,
            "timestep": self.timestep,
        }

    @staticmethod
    def from_dict(state_dict):
        state_dict = copy.deepcopy(state_dict)
        state_dict["players"] = [PlayerState.from_dict(p) for p in state_dict["players"]]
        object_list = [ObjectState.from_dict(o) for o in state_dict["objects"]]
        state_dict["objects"] = { ob.position : ob for ob in object_list }
        try: 
            del state_dict["bonus_orders"] 
            del state_dict["all_orders"] 
            state_dict["order_list"] = None
        except:  
            pass 
        return OvercookedState(**state_dict)
    
    @staticmethod
    def from_json(filename):
        return load_from_json(filename)


NO_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 0,
    "DISH_PICKUP_REWARD": 0,
    "SOUP_PICKUP_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0
}

EVENT_TYPES = [
    # Tomato events
    "tomato_pickup",
    "useful_tomato_pickup",
    "tomato_drop",
    "useful_tomato_drop",
    "potting_tomato",
    # Onion events
    "onion_pickup",
    "useful_onion_pickup",
    "onion_drop",
    "useful_onion_drop",
    "potting_onion",
    # Dish events
    "dish_pickup",
    "useful_dish_pickup",
    "dish_drop",
    "useful_dish_drop",
    # Soup events
    "soup_pickup",
    "soup_delivery",
    "soup_drop",
    # Potting events
    "optimal_onion_potting",
    "optimal_tomato_potting",
    "viable_onion_potting",
    "viable_tomato_potting",
    "catastrophic_onion_potting",
    "catastrophic_tomato_potting",
    "useless_onion_potting",
    "useless_tomato_potting",
]

class OvercookedGridworld(object):
    """An MDP grid world based off of the Overcooked game."""
    # ORDER_TYPES = ObjectState.SOUP_TYPES + ['any']

    def __init__(self, terrain, 
                 start_player_positions, 
                 start_all_orders=[], 
                 start_order_list=None,
                 order_probability=None,
                 recipes=None, 
                 cook_time=20, 
                 num_items_for_soup=3, 
                 delivery_reward=20, 
                 utensils=None,
                 ingredients=[],
                 rew_shaping_params=None, 
                 layout_name="unnamed_layout",
                 old_dynamics=False, 
                 need_dish={},
                 **kwargs
                 ):
        """
        terrain: a matrix of strings that encode the MDP layout
        layout_name: string identifier of the layout
        start_player_positions: tuple of positions for both players' starting positions
        start_order_list: either a tuple of orders or None if there is not specific list
        cook_time: amount of timesteps required for a soup to cook
        delivery_reward: amount of reward given per delivery
        rew_shaping_params: reward given for completion of specific subgoals
        """
        # self._configure_recipes(start_all_orders, num_items_for_soup, **kwargs)
        # self.start_all_orders = (
        #     [r.to_dict() for r in Recipe.ALL_RECIPES]
        #     if not start_all_orders
        #     else start_all_orders
        # )
        
        self.recipes = recipes
        self.utensils = utensils
        self.need_dish = need_dish
        self.default_ingredients = ingredients
        assert Counter(self.utensils.keys()) == Counter(self.recipes.keys()), "The utensil and the utensil recipe do not match."
        self._configure_recipes(recipes, need_dish, utensils, **kwargs)
        assert all(item in (self._get_all_recipe_names(self.recipes) + ingredients) for item in start_order_list)
        self.start_all_orders = start_all_orders
        self.start_order_list = start_order_list
        self.order_probability = order_probability
        self.start_order_list = self.add_elements_based_on_probability(start_order_list, order_probability)
        if old_dynamics:
            assert all(
                [
                    len(order["ingredients"]) == 3
                    for order in self.start_all_orders
                ]
            ), "Only accept orders with 3 items when using the old_dynamics"        

        self.height = len(terrain)
        self.width = len(terrain[0])
        self.shape = (self.width, self.height)
        self.terrain_mtx = terrain
        self.interactive_terrain = [utensil['symbol'] for utensil in utensils.values()]
        self.terrain_pos_dict = self._get_terrain_type_pos_dict()
        self.start_player_positions = start_player_positions
        self.num_players = len(start_player_positions)
        self.soup_cooking_time = cook_time
        self.num_items_for_soup = num_items_for_soup
        self.delivery_reward = delivery_reward
        self.reward_shaping_params = NO_REW_SHAPING_PARAMS if rew_shaping_params is None else rew_shaping_params
        self.layout_name = layout_name
        self.interact_actions = {}
        self.utensil_list_chef = []
        self.utensil_list_assist = []
        self.utensil_list = self.generate_utensil_list()
        self.init_utensil_states()
        self.all_ingredients = self.Recipe.recipe_name+self.default_ingredients
        self.one_task_mode = False


    def generate_utensil_list(self):
        utensil_list = []

        # Create a dictionary to keep track of counts for each utensil
        utensil_counts = {utensil: 0 for utensil in self.utensils}

        # Traverse the terrain_mtx
        for row in self.terrain_mtx:
            for symbol in row:
                # Check if the symbol matches any utensil symbol
                for utensil, properties in self.utensils.items():
                    if symbol == properties['symbol']:
                        # Append utensil with count to the list
                        utensil_list.append(f"{utensil}{utensil_counts[utensil]}")
                        # Append utensil name into the interact_actions list for parser
                        if properties['operation'] not in self.interact_actions.keys():
                            self.interact_actions[properties['operation']] = [f"{utensil}{utensil_counts[utensil]}"]
                        else:
                            self.interact_actions[properties['operation']].append(f"{utensil}{utensil_counts[utensil]}")
                        # Increment the count for the utensil
                        utensil_counts[utensil] += 1

        return utensil_list

    def __eq__(self, other):
        return np.array_equal(self.terrain_mtx, other.terrain_mtx) and \
                self.start_player_positions == other.start_player_positions and \
                self.start_order_list == other.start_order_list and \
                self.soup_cooking_time == other.soup_cooking_time and \
                self.num_items_for_soup == other.num_items_for_soup and \
                self.delivery_reward == other.delivery_reward and \
                self.reward_shaping_params == other.reward_shaping_params and \
                self.layout_name == other.layout_name
    
    def copy(self):
        return OvercookedGridworld(
            terrain=self.terrain_mtx.copy(),
            start_player_positions=self.start_player_positions,
            start_order_list=None if self.start_order_list is None else list(self.start_order_list),
            cook_time=self.soup_cooking_time,
            num_items_for_soup=self.num_items_for_soup,
            delivery_reward=self.delivery_reward,
            rew_shaping_params=copy.deepcopy(self.reward_shaping_params),
            layout_name=self.layout_name
        )

    @property
    def mdp_params(self):
        return {
            "layout_name": self.layout_name,
            "terrain": self.terrain_mtx,
            "start_player_positions": self.start_player_positions,
            "start_order_list": self.start_order_list,
            "cook_time": self.soup_cooking_time,
            "num_items_for_soup": self.num_items_for_soup,
            "delivery_reward": self.delivery_reward,
            "rew_shaping_params": copy.deepcopy(self.reward_shaping_params)
        }

    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a OvercookedGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        base_layout_params = read_layout_dict(layout_name)

        grid = base_layout_params['grid']
        del base_layout_params['grid']
        base_layout_params['layout_name'] = layout_name

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return OvercookedGridworld.from_grid(grid, base_layout_params, params_to_overwrite)

    @staticmethod
    def from_grid(layout_grid, base_layout_params={}, params_to_overwrite={}, debug=False):
        """
        Returns instance of OvercookedGridworld with terrain and starting 
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = base_layout_params.copy()

        layout_grid = [[c for c in row] for row in layout_grid]
        OvercookedGridworld._assert_valid_grid(layout_grid)

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    layout_grid[y][x] = ' '

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert player_positions[int(c) - 1] is None, 'Duplicate player in grid'
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions


        for k, v in params_to_overwrite.items():   
            if k == 'start_bonus_orders': 
                continue 
            try: 
                curr_val = mdp_config[k]
                if debug:
                    print("Overwriting mdp layout standard config value {}:{} -> {}".format(k, curr_val, v))
                mdp_config[k] = v
            except: 
                continue 

        return OvercookedGridworld(**mdp_config)

    def _get_all_recipe_names(self, recipes):
        recipe_names = []
        for category in recipes.values():
            for recipe_name in category.keys():
                recipe_names.append(recipe_name)
        return recipe_names

    def add_elements_based_on_probability(self, start_order_list, order_probability, num_elements=3):
        if len(start_order_list) <= 3:
            items = list(order_probability.keys())
            probabilities = list(order_probability.values())
            
            # Filter out items with zero probability
            non_zero_items = [item for item, prob in zip(items, probabilities) if prob > 0]
            non_zero_probabilities = [prob for prob in probabilities if prob > 0]

            # Normalize probabilities
            total_prob = sum(non_zero_probabilities)
            normalized_probabilities = [prob / total_prob for prob in non_zero_probabilities]
            
            # Select elements based on their probabilities
            selected_elements = random.choices(non_zero_items, weights=normalized_probabilities, k=num_elements - len(start_order_list))
            start_order_list.extend(selected_elements)
        
        return start_order_list
        
    
    def _configure_recipes(
        self, recipe, need_dish, utensil, **kwargs):
        
        self.recipe_config = {
            "recipes": recipe,
            "need_dish": need_dish
        }
        self.Recipe = Recipe(self.default_ingredients, self.recipe_config)

    def get_actions(self, state):
        """
        Returns the list of lists of valid actions for 'state'.

        The ith element of the list is the list of valid actions that player i
        can take.
        """
        self._check_valid_state(state)
        return [self._get_player_actions(state, i) for i in range(len(state.players))]

    def _get_player_actions(self, state, player_num):
        """All actions are allowed to all players in all states."""
        return Action.ALL_ACTIONS

    def _check_action(self, state, joint_action):
        for p_action, p_legal_actions in zip(joint_action, self.get_actions(state)):
            if p_action not in p_legal_actions:
                raise ValueError('Invalid action')

    def get_standard_start_state(self):
        start_state = OvercookedState.from_player_positions(
            self.start_player_positions, order_list=self.start_order_list
        )
        return start_state

    def custom_start_state_fn(self):
        start_pos = self.start_player_positions
        start_state = OvercookedState.from_player_positions(start_pos, order_list=self.start_order_list)
        for player in start_state.players:
            if player.position == (1,2):
                player.set_object(
                            ObjectState('chopped_onion', 'ingredient', player.position)
                        )
        return start_state


    def get_random_start_state_fn(self, random_start_pos=False, rnd_obj_prob_thresh=0.0):
        def start_state_fn():
            if random_start_pos:
                valid_positions = self.get_valid_joint_player_positions()
                start_pos = valid_positions[np.random.choice(len(valid_positions))]
            else:
                start_pos = self.start_player_positions

            start_state = OvercookedState.from_player_positions(start_pos, order_list=self.start_order_list)

            if rnd_obj_prob_thresh == 0:
                return start_state

            # Arbitrary hard-coding for randomization of objects
            # For each pot, add a random amount of onions with prob rnd_obj_prob_thresh
            pots = self.get_pot_states(start_state)["empty"]
            for pot_loc in pots:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    n = int(np.random.randint(low=1, high=4))
                    start_state.objects[pot_loc] = ObjectState("soup", pot_loc, ('onion', n, 0))

            # For each player, add a random object with prob rnd_obj_prob_thresh
            for player in start_state.players:
                p = np.random.rand()
                if p < rnd_obj_prob_thresh:
                    # Different objects have different probabilities
                    obj = np.random.choice(["dish", "onion", "soup"], p=[0.2, 0.6, 0.2])
                    if obj == "soup":
                        player.set_object(
                            ObjectState(obj, player.position, ('onion', self.num_items_for_soup, self.soup_cooking_time))
                        )
                    else:
                        player.set_object(ObjectState(obj, player.position))
            return start_state
        return start_state_fn

    def is_terminal(self, state):
        # There is a finite horizon, handled by the environment.
        if state.order_list is None:
            return False
        return len(state.order_list) == 0

    def get_valid_player_positions(self):
        return self.terrain_pos_dict[' ']

    def get_valid_joint_player_positions(self):
        """Returns all valid tuples of the form (p0_pos, p1_pos, p2_pos, ...)"""
        valid_positions = self.get_valid_player_positions() 
        all_joint_positions = list(itertools.product(valid_positions, repeat=self.num_players))
        valid_joint_positions = [j_pos for j_pos in all_joint_positions if not self.is_joint_position_collision(j_pos)]
        return valid_joint_positions

    def get_valid_player_positions_and_orientations(self):
        valid_states = []
        for pos in self.get_valid_player_positions():
            valid_states.extend([(pos, d) for d in Direction.ALL_DIRECTIONS])
        return valid_states

    def get_valid_joint_player_positions_and_orientations(self):
        """All joint player position and orientation pairs that are not
        overlapping and on empty terrain."""
        valid_player_states = self.get_valid_player_positions_and_orientations()

        valid_joint_player_states = []
        for players_pos_and_orientations in itertools.product(valid_player_states, repeat=self.num_players):
            joint_position = [plyer_pos_and_or[0] for plyer_pos_and_or in players_pos_and_orientations]
            if not self.is_joint_position_collision(joint_position):
                valid_joint_player_states.append(players_pos_and_orientations)

        return valid_joint_player_states

    def get_valid_actions(self, player):  
        pos = player.position 
        lis = [Action.STAY] 
        for d in Direction.ALL_DIRECTIONS:  
            adj_pos = Action.move_in_direction(pos, d)  
            if self.get_terrain_type_at_pos(adj_pos) == ' ':  
                lis.append(d)  
        return lis
    
    def get_adjacent_features(self, player):
        adj_feats = []
        pos = player.position
        for d in Direction.ALL_DIRECTIONS:
            adj_pos = Action.move_in_direction(pos, d)
            adj_feats.append((pos, self.get_terrain_type_at_pos(adj_pos)))
        return adj_feats

    def get_terrain_type_at_pos(self, pos):
        x, y = pos
        return self.terrain_mtx[y][x]

    def get_dish_dispenser_locations(self):
        return list(self.terrain_pos_dict['D'])

    # def get_onion_dispenser_locations(self):
    #     return list(self.terrain_pos_dict['O'])

    def get_ingredient_dispenser_locations(self):
        return list(self.terrain_pos_dict['I'])

    # def get_tomato_dispenser_locations(self):
    #     return list(self.terrain_pos_dict['T'])

    def get_serving_locations(self):
        return list(self.terrain_pos_dict['S'])

    def get_pot_locations(self):
        return list(self.terrain_pos_dict['P'])

    def get_chopping_board_locations(self):
        return list(self.terrain_pos_dict['C'])
    
    def get_stir_locations(self):
        return list(self.terrain_pos_dict['S'])

    def get_oven_locations(self):
        return list(self.terrain_pos_dict['O'])

    def get_counter_locations(self):
        return list(self.terrain_pos_dict['X'])
    
    def from_pos_to_utensil_name(self, pos):
        target_symbol = ''
        target_index = -1
        for symbol, symbol_pos in self.terrain_pos_dict.items():
            if pos in symbol_pos:
                target_symbol = symbol
                target_index = symbol_pos.index(pos)
                break
        if target_symbol != '':
            for full_name, symbol_dict in self.utensils.items():
                if symbol_dict['symbol'] == target_symbol:
                    return full_name + str(target_index)
            raise ValueError()
        else:
            raise ValueError()

    def from_utensil_name_to_pos(self, utensil_name):
        assert isinstance(utensil_name, str), "Utensil name must be str"
        utensil_index = int(utensil_name[-1])
        symbol = self.utensils[utensil_name[:-1]]['symbol']
        utensil_pos = self.terrain_pos_dict[symbol][utensil_index]
        return utensil_pos


    def init_utensil_states(self):
        #get {order_name,} dict for utensil
        '''Returns dict with structure:
        {
            utensil_state:{
                name:string,
                position:tuple,
                order:string,
                soup:OvercookState,
            }
        }'''
        utensil_state_dict = defaultdict(list)
        index_in_same_utensil = 0
        for utensil_name in self.utensil_list:
            pattern = r'\d+'
            numbers = re.findall(pattern,utensil_name)
            if len(numbers)==1:
                index_in_same_utensil = numbers[0]
            else:
                raise ValueError("Utensil's name is not valid, it should has at least one index number")
            position = list(self.terrain_pos_dict[self.utensils[utensil_name[:-1]]['symbol']])[int(index_in_same_utensil)]
            new_utensil = {"position": position, "order": None,"soup": None}
            utensil_state_dict[utensil_name] = new_utensil
        self.utensil_state_dict = utensil_state_dict

    # judge the food in utensil for operation is full?
    def check_food_full(self, utensil_state, state: OvercookedState):
        utensil_state_keys = list(utensil_state.keys())[0]
        utensil_state_values = list(utensil_state.values())[0]
        for s in state.objects.values():
            if s.catagory in ['semi-finished'] and s.position == utensil_state_values['position']:
                if isinstance(s.state[0], list):
                    for rpe_name, rpe in self.recipes[utensil_state_keys[:-1]].items(): 
                        if set(s.state[0]) < set(rpe['recipe']): 
                            return PARTIALLY_FULL
                        elif s.state[0] == rpe['recipe']:
                            return FULL
                else:
                    return FULL
        #TODO:FIX BUG  THEN DELETE IT        
        return WRONG
    
        #switch utensil state to utensil_state
        """utensil_type:"Pot,Chooping_board,"
        Returns dict with structure:
        empty: [ObjStates]
        ingredient: {
        'wrong':[soup with wrong food]
        'full':[full but not cooking]
        'cooking': [ready soup objs]
        'ready': [ready soup objs],
        'partially_full': [all non-empty and non-full soups]
        }}"""
    def get_utensil_states(self, state: OvercookedState): 
        utensil_states_dict = defaultdict(list)
        utensil_states_dict['empty'] = []
        utensil_states_dict['cooking'] = []
        utensil_states_dict['ready'] = []
        utensil_states_dict['full'] = []
        utensil_states_dict['partially_full'] = []
        utensil_states_dict['wrong'] = []
        # bone soup with utensil
        for utensil in self.utensil_state_dict.values():
            # print(utensil)
            # print(self.utensil_state_dict.values())
            utensil['soup'] = None

        for s in state.objects.values():
            if s.state != None:
                for utensil_key, utensil in self.utensil_state_dict.items():
                    if utensil['position'] == s.position:
                        self.utensil_state_dict[utensil_key]['soup'] = s

                        
        for key,value in self.utensil_state_dict.items():
            uten = {}
            uten[key] = value
            u = uten[list(uten.keys())[0]]
            utensil_name = list(uten.keys())[0]
            #Pot,Oven,Chopping_board,Stir: empty,cooking,ready,full,partially_full
            if u['soup'] is None:
                utensil_states_dict['empty'].append(list(uten.keys())[0])
            elif u['soup'].state[2] > 0 and u['soup'].state[2] < self.Recipe.recipe_list[u['soup'].state[0]]['cook_time']:
                utensil_states_dict['cooking'].append(utensil_name)
            elif u['soup'].state[2] > 0 and u['soup'].state[2] == self.Recipe.recipe_list[u['soup'].state[0]]['cook_time']:
                utensil_states_dict['ready'].append(utensil_name)
            elif u['soup'].state[2] == 0 and self.check_food_full(uten,state)==FULL:  #check the food is not full for cook
                utensil_states_dict['full'].append(utensil_name)
            elif u['soup'].state[2] == 0 and self.check_food_full(uten,state)==PARTIALLY_FULL:
                utensil_states_dict['partially_full'].append(utensil_name)
            elif u['soup'].state[2] == 0 and self.check_food_full(uten,state)==WRONG:
                utensil_states_dict['wrong'].append(utensil_name)
            else:
                raise ValueError("A soup state in utensil is in wrong state")

        return utensil_states_dict

    def get_pot_states(self, state):
        """Returns dict with structure:
        {
         empty: [ObjStates]
         ingredient: {
            'x_items': [soup objects with x items],
            'cooking': [ready soup objs]
            'ready': [ready soup objs],
            'partially_full': [all non-empty and non-full soups]
            }
         tomato: same dict structure as above
        }
        """
        pots_states_dict = {}
        pots_states_dict['empty'] = []
        # pots_states_dict['onion'] = defaultdict(list)
        # pots_states_dict['tomato'] = defaultdict(list)
        pots_states_dict['ingredient'] = defaultdict(list)
        for pot_pos in self.get_pot_locations():
            if not state.has_object(pot_pos):
                pots_states_dict['empty'].append(pot_pos)
            else:
                soup_obj = state.get_object(pot_pos)
                soup_type, num_items, cook_time = soup_obj.state
                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict['ingredient']['{}_items'.format(num_items)].append(pot_pos)
                elif num_items == self.num_items_for_soup:
                    assert cook_time <= self.soup_cooking_time
                    if cook_time == self.soup_cooking_time:
                        pots_states_dict['ingredient']['ready'].append(pot_pos)
                    else:
                        pots_states_dict['ingredient']['cooking'].append(pot_pos)
                else:
                    raise ValueError("Pot with more than {} items".format(self.num_items_for_soup))

                if 0 < num_items < self.num_items_for_soup:
                    pots_states_dict['ingredient']['partially_full'].append(pot_pos)
                
        return pots_states_dict

    def get_counter_objects_dict(self, state, counter_subset=None):
        """Returns a dictionary of pos:objects on counters by type"""
        counters_considered = self.terrain_pos_dict['X'] if counter_subset is None else counter_subset
        counter_objects_dict = defaultdict(list)
        for obj in state.objects.values():
            if obj.position in counters_considered:
                counter_objects_dict[obj.name].append(obj.position)
        return counter_objects_dict

    def get_empty_counter_locations(self, state):
        counter_locations = self.get_counter_locations()
        return [pos for pos in counter_locations if not state.has_object(pos)]

    def get_state_transition(self, state, joint_action, parm=None):
        """Gets information about possible transitions for the action.

        Returns the next state, sparse reward and reward shaping.
        Assumes all actions are deterministic.

        NOTE: Sparse reward is given only when soups are delivered, 
        shaped reward is given only for completion of subgoals 
        (not soup deliveries).
        """
        assert not self.is_terminal(state), "Trying to find successor of a terminal state: {}".format(state)
        for action, action_set in zip(joint_action, self.get_actions(state)):
            if action not in action_set:
                raise ValueError("Illegal action %s in state %s" % (action, state))

        new_state = state.deepcopy()

        # print(new_state)

        # Resolve interacts first
        sparse_reward, shaped_reward = self.resolve_interacts(new_state, joint_action, parm) ## I add the ml_actions feature through this fun
        print(new_state.ml_actions)
        assert new_state.player_positions == state.player_positions
        assert new_state.player_orientations == state.player_orientations

        # Resolve player movements
        self.resolve_movement(new_state, joint_action)

        # Finally, environment effects
        sparse_reward += self.step_environment_effects(new_state) ## realy important, include add timestep

        # Additional dense reward logic
        # shaped_reward += self.calculate_distance_based_shaped_reward(state, new_state)

        return new_state, sparse_reward, shaped_reward

    def get_commucation_state(self, state, comm_history):
        new_state = state.deepcopy()
        new_state.add_communicate_history(comm_history)
        return new_state

    def resolve_interacts(self, new_state, joint_action, parm=None):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact 
        first and then player 2's, without doing anything like collision checking.
        """
        pot_states = self.get_pot_states(new_state) # print(self.state_string(new_state)) # can use to check current state
        ready_pots = pot_states["ingredient"]["ready"]
        cooking_pots = ready_pots + pot_states["ingredient"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["ingredient"]["partially_full"]
        # ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        # cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        # nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]

        sparse_reward, shaped_reward, ml_actions = 0, 0, [None, None]
        for i, (player, action) in enumerate(zip(new_state.players, joint_action)):
            if action != Action.INTERACT:
                continue

            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o) # imagine to move to the intection place
            terrain_type = self.get_terrain_type_at_pos(i_pos)

            if terrain_type == 'X':
                if player.has_object() and not new_state.has_object(i_pos):
                    ml_actions[i] = f'place_obj_on_counter()'
                    new_state.add_object(player.remove_object(), i_pos)
                elif not player.has_object() and new_state.has_object(i_pos):
                    player.set_object(new_state.remove_object(i_pos))
                    ml_actions[i] = f'pickup({player.held_object.name}, counter)'

            # elif terrain_type == 'O' and player.held_object is None:
            #     player.set_object(ObjectState('onion', pos))
            #     ml_actions[i] = 'pickup_onion'
            # elif terrain_type == 'T' and player.held_object is None:
            #     player.set_object(ObjectState('tomato', pos))
            #     ml_actions[i] = 'pickup_tomato'
                    
            elif terrain_type == 'D' and player.held_object is None:
                dishes_already = len(new_state.player_objects_by_type['dish'])
                player.set_object(ObjectState('dish', 'dish', pos))
                ml_actions[i] = 'pickup(dish, dish_dispenser)'

                dishes_on_counters = self.get_counter_objects_dict(new_state)["dish"]
                if len(nearly_ready_pots) > dishes_already and len(dishes_on_counters) == 0:
                    shaped_reward += self.reward_shaping_params["DISH_PICKUP_REWARD"]

            elif terrain_type == 'I' and player.held_object is None and parm[i] is not None:
                ingredient = parm[i]
                player.set_object(ObjectState(ingredient, 'ingredient', pos))
                ml_actions[i] = f'pickup({ingredient}, ingredient_dispenser)'

            elif (terrain_type in self.interactive_terrain and 
                  # player.held_object is None and 
                  new_state.has_object(i_pos) and
                  hasattr(new_state.get_object(i_pos), 'catagory') and 
                  new_state.get_object(i_pos).catagory == 'semi-finished' and
                  new_state.get_object(i_pos).state[2] == 0 and
                  parm[i] == '[START]'):
                if terrain_type == 'P':
                    ml_actions[i] = f'cook({self.from_pos_to_utensil_name(i_pos)})'
                elif terrain_type == 'C':
                    ml_actions[i] = f'cut({self.from_pos_to_utensil_name(i_pos)})'
                elif terrain_type == 'O':
                    ml_actions[i] = f'bake({self.from_pos_to_utensil_name(i_pos)})'
                elif terrain_type == 'B':
                    ml_actions[i] = f'stir({self.from_pos_to_utensil_name(i_pos)})'

            elif terrain_type == 'P' and player.has_object():
                if player.get_object().name == 'dish' and new_state.has_object(i_pos):
                    obj = new_state.get_object(i_pos)
                    assert obj.state[0] in self.Recipe.recipe_name, 'Object in pot was not soup'
                    name, num_items, cook_time = obj.state
                    if cook_time >= self.Recipe.recipe_list[name]['cook_time']:
                        player.remove_object()  # Turn the dish into the soup

                        object_in_hand = new_state.remove_object(i_pos)
                        object_in_hand.catagory = 'with_dish'

                        player.set_object(object_in_hand)
                        shaped_reward += self.reward_shaping_params["SOUP_PICKUP_REWARD"]
                        
                        ml_actions[i] = f'fill_dish_with_food({self.from_pos_to_utensil_name(i_pos)})'

                elif player.get_object().name != 'dish':   # Not equal to dish is equivalent to ingredients
                    item_type = player.get_object().name

                    if not new_state.has_object(i_pos):
                        # Pot was empty
                        player.remove_object()
                        new_state.add_object(ObjectState('soup', 'semi-finished', i_pos, ([item_type], 1, 0)), i_pos)
                        shaped_reward += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]

                    else:
                        # Pot has already items in it
                        obj = new_state.get_object(i_pos)
                        # assert obj.name in self.Recipe.recipe_name, 'Object in pot was not soup'
                        soup_type, num_items, cook_time = obj.state
                        if num_items < self.num_items_for_soup:
                            player.remove_object()
                            soup_type.append(item_type)
                            obj.state = (soup_type, num_items + 1, 0)
                            shaped_reward += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]
                    
                    ml_actions[i] = f'put_obj_in_utensil({self.from_pos_to_utensil_name(i_pos)})'

            elif terrain_type in self.interactive_terrain:
                if player.has_object():
                    utensil_name = next(name for name, properties in self.utensils.items() if properties['symbol'] == terrain_type)
                    item_type = player.get_object().name
                    
                    if not new_state.has_object(i_pos):
                        # Utensil was empty
                        player.remove_object()
                        new_state.add_object(ObjectState('soup', 'semi-finished', i_pos, ([item_type], 1, 0)), i_pos)
                        shaped_reward += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]
                    elif item_type == 'dish':
                        obj = new_state.get_object(i_pos)
                        assert obj.state[0] in self.Recipe.recipe_name, 'Object in pot was not soup'
                        name, num_items, cook_time = obj.state
                        if cook_time >= self.Recipe.recipe_list[name]['cook_time']:
                            player.remove_object()  # Turn the dish into the soup

                            object_in_hand = new_state.remove_object(i_pos)
                            object_in_hand.catagory = 'with_dish'

                            player.set_object(object_in_hand)
                            shaped_reward += self.reward_shaping_params["SOUP_PICKUP_REWARD"]
                            
                            ml_actions[i] = f'fill_dish_with_food({self.from_pos_to_utensil_name(i_pos)})'
                    else:
                        # Utensil has already items in it
                        obj = new_state.get_object(i_pos)
                        # assert obj.name in self.Recipe.recipe_name, 'Object in pot was not soup'
                        soup_type, num_items, cook_time = obj.state
                        assert isinstance(soup_type, list), "Validator error, Ingredients should not be placed in cooking utensils"
                        if num_items < self.num_items_for_soup:
                            player.remove_object()
                            soup_type.append(item_type)
                            obj.state = (soup_type, num_items + 1, 0)
                            shaped_reward += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]
                    
                    ml_actions[i] = f'put_obj_in_utensil({self.from_pos_to_utensil_name(i_pos)})'
                
                elif new_state.has_object(i_pos):
                    obj = new_state.get_object(i_pos)
                    name, num_items, cook_time = obj.state
                    if parm[i] is not None and parm[i] != '' and parm[i] in name:
                        if isinstance(name, str):
                            if name == parm[i]:
                                obj_removed = new_state.remove_object(i_pos)
                                player.set_object(ObjectState(parm[i], 'ingredient', pos))
                            else:
                                obj.state[0].remove(parm[i])
                                player.set_object(ObjectState(parm[i], 'ingredient', pos))
                        elif isinstance(name, list):
                            if name == [parm[i]]:
                                obj_removed = new_state.remove_object(i_pos)
                                player.set_object(ObjectState(parm[i], 'ingredient', pos))
                            else:
                                obj.state[0].remove(parm[i])
                                player.set_object(ObjectState(parm[i], 'ingredient', pos))
                                
                        ml_actions[i] = f'pickup({parm[i]}, {self.from_pos_to_utensil_name(i_pos)})'
                    elif  (isinstance(name, str) and cook_time >= self.Recipe.recipe_list[name]['cook_time']) or (isinstance(name, list) and cook_time >= self.Recipe.recipe_list[name[0]]['cook_time']):
                        # player.remove_object()  # Turn the dish into the soup
                        player.set_object(new_state.remove_object(i_pos))
                        shaped_reward += self.reward_shaping_params["SOUP_PICKUP_REWARD"]
                        
                        ml_actions[i] = f'pickup({name}, {self.from_pos_to_utensil_name(i_pos)})'


            elif terrain_type == 'S' and player.has_object():
                obj = player.get_object()
                if obj.name in self.Recipe.recipe_name:

                    new_state, delivery_rew = self.deliver_soup(new_state, player, obj)
                    sparse_reward += delivery_rew   
                    ml_actions[i] = f'deliver_soup()'                     

                    # If last soup necessary was delivered, stop resolving interacts
                    if new_state.order_list is not None and len(new_state.order_list) == 0:
                        break
        
        new_state.ml_actions = ml_actions

        return sparse_reward, shaped_reward

    def deliver_soup(self, state, player, soup_obj):
        """
        Deliver the soup, and get reward if there is no order list
        or if the type of the delivered soup matches the next order.
        """
        if soup_obj.state == None:
            soup_type = soup_obj.name

        else:
            soup_type, num_items, cook_time = soup_obj.state
            # assert soup_type in ObjectState.SOUP_TYPES
            # assert num_items == self.num_items_for_soup
            assert cook_time >= self.Recipe.recipe_list[soup_type]['cook_time'], "Cook time {} mdp cook time {}".format(cook_time, self.recipe_config['recipes'][state.curr_order]['cook_time'])
        player.remove_object()

        if state.order_list is None:
            return state, self.delivery_reward
    
        # If the delivered soup is the one currently required
        assert not self.is_terminal(state)
        current_order = state.order_list[0]
        if current_order == 'any' or soup_type == current_order:
            if (soup_type in self.Recipe.recipe_need_dish) and soup_obj.catagory == 'with_dish':
                state.order_list = state.order_list[1:]
                if len(state.order_list) <= state.k_order:
                    state.order_list = self.add_elements_based_on_probability(state.order_list, self.order_probability)
                return state, self.delivery_reward
            elif soup_type not in self.Recipe.recipe_need_dish and soup_obj.catagory != 'with_dish':
                state.order_list = state.order_list[1:]
                if len(state.order_list) <= state.k_order:
                    state.order_list = self.add_elements_based_on_probability(state.order_list, self.order_probability)
                return state, self.delivery_reward

        
        return state, 0

    def match_recipe(self, soup_obj, utensil_name):
        soup_type, _, _ = soup_obj.state
        for recipe_name, recipe_detail in self.Recipe.recipe_config[utensil_name].items():
            if Counter(soup_type) == Counter(recipe_detail['recipe']):
                return recipe_name
        return None


    def resolve_movement(self, state, joint_action):
        """Resolve player movement and deal with possible collisions"""
        new_positions, new_orientations = self.compute_new_positions_and_orientations(state.players, joint_action)
        for player_state, new_pos, new_o in zip(state.players, new_positions, new_orientations):
            player_state.update_pos_and_or(new_pos, new_o)

    def compute_new_positions_and_orientations(self, old_player_states, joint_action):
        """Compute new positions and orientations ignoring collisions"""
        new_positions, new_orientations = list(zip(*[
            self._move_if_direction(p.position, p.orientation, a) \
            for p, a in zip(old_player_states, joint_action)]))
        old_positions = tuple(p.position for p in old_player_states)
        new_positions = self._handle_collisions(old_positions, new_positions)
        return new_positions, new_orientations

    def is_transition_collision(self, old_positions, new_positions):
        # Checking for any players ending in same square
        if self.is_joint_position_collision(new_positions):
            return True
        # Check if any two players crossed paths
        for idx0, idx1 in itertools.combinations(range(self.num_players), 2):
            p1_old, p2_old = old_positions[idx0], old_positions[idx1]
            p1_new, p2_new = new_positions[idx0], new_positions[idx1]
            if p1_new == p2_old and p1_old == p2_new:
                return True
        return False

    def is_joint_position_collision(self, joint_position):
        return any(pos0 == pos1 for pos0, pos1 in itertools.combinations(joint_position, 2))
            
    def step_environment_effects(self, state):
        state.timestep += 1 ## that's why the state.timestep works
        reward = 0
        wrong_recipe_flag = False
        break_outer_loop = False
        state.error_message = []
        
        for obj in state.objects.values():
            if obj.catagory in ['semi-finished']:
                x, y = obj.position
                soup_type, num_items, cook_time = obj.state
                for act in state.ml_actions:
                    if act is not None:
                        pattern = r'^(cut|cook|bake|stir)\((.*)\)$'
                        # print(act)
                        match = re.match(pattern, act)
                        
                        if match:
                            utensil_pos = self.from_utensil_name_to_pos(match.group(2))

                            x_pot, y_pot = utensil_pos[0], utensil_pos[1]

                            if x_pot != x and y_pot != y: continue

                            # x_pot, y_pot = int(x_pot), int(y_pot)

                            utensil_name = next(name for name, properties in self.utensils.items() if properties['symbol'] == self.terrain_mtx[y_pot][x_pot])
                            assert utensil_name is not None

                            recipe_name = self.match_recipe(obj, utensil_name)

                            if recipe_name is not None:
                                if x == x_pot and y == y_pot and cook_time == 0:
                                    obj.state = recipe_name, num_items, cook_time + 1
                                    break_outer_loop = True
                                    break

                            elif recipe_name is None and cook_time == 0:
                                error_recipe_obj = obj
                                wrong_recipe_flag = True

                if break_outer_loop: continue

                soup_type, num_items, cook_time = obj.state
                if self.terrain_mtx[y][x] in self.interactive_terrain and cook_time != 0:
                    if cook_time < self.Recipe.recipe_list[obj.state[0]]['cook_time']:
                        obj.state = soup_type, num_items, cook_time + 1 ## here add the cook_time
                    soup_type, num_items, cook_time = obj.state
                    if cook_time == self.Recipe.recipe_list[obj.state[0]]['cook_time']:
                        obj.name = soup_type
                        obj.catagory = 'finished'

        if wrong_recipe_flag:
            state.remove_object(error_recipe_obj.position)
            wrong_recipe_flag = False
            wrong_utensil_name = self.from_pos_to_utensil_name(error_recipe_obj.position)
            state.error_message.append(f"TIMESTEP {state.timestep}: Due to the items in {wrong_utensil_name} not matching any recipe, an error occurred, and the items in {wrong_utensil_name} has been automatically deleted.")
            print("Wrong recipe!")

        return reward

    def _handle_collisions(self, old_positions, new_positions):
        """If agents collide, they stay at their old locations"""
        if self.is_transition_collision(old_positions, new_positions):
            return old_positions
        return new_positions

    def _get_terrain_type_pos_dict(self):
        pos_dict = defaultdict(list)
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, terrain_type in enumerate(terrain_row):
                pos_dict[terrain_type].append((x, y))
        return pos_dict

    def _move_if_direction(self, position, orientation, action):
        """Returns position and orientation that would 
        be obtained after executing action"""
        if action == Action.INTERACT:
            return position, orientation
        new_pos = Action.move_in_direction(position, action)
        new_orientation = orientation if action == Action.STAY else action
        if new_pos not in self.get_valid_player_positions():
            return position, new_orientation
        return new_pos, new_orientation

    def _check_valid_state(self, state):
        """Checks that the state is valid.

        Conditions checked:
        - Players are on free spaces, not terrain
        - Held objects have the same position as the player holding them
        - Non-held objects are on terrain
        - No two players or non-held objects occupy the same position
        - Objects have a valid state (eg. no pot with 4 onions)
        """
        all_objects = list(state.objects.values())
        for player_state in state.players:
            # Check that players are not on terrain
            pos = player_state.position
            assert pos in self.get_valid_player_positions()

            # Check that held objects have the same position
            if player_state.held_object is not None:
                all_objects.append(player_state.held_object)
                assert player_state.held_object.position == player_state.position

        for obj_pos, obj_state in state.objects.items():
            # Check that the hash key position agrees with the position stored
            # in the object state
            assert obj_state.position == obj_pos
            # Check that non-held objects are on terrain
            assert self.get_terrain_type_at_pos(obj_pos) != ' '

        # Check that players and non-held objects don't overlap
        all_pos = [player_state.position for player_state in state.players]
        all_pos += [obj_state.position for obj_state in state.objects.values()]
        assert len(all_pos) == len(set(all_pos)), "Overlapping players or objects"

        # Check that objects have a valid state
        for obj_state in all_objects:
            print(obj_state)
            # assert obj_state.is_valid()
    
    @staticmethod
    def _assert_valid_grid(grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a counter), ' ' (an empty
        space), 'O' (onion supply), 'P' (pot), 'D' (dish supply), 'S' (serving
        location), '1' (player 1) and '2' (player 2).

        Update: "I" (ingredients supply)
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), 'Ragged grid'

        # Borders must not be free spaces
        def is_not_free(c):
            return c in 'XOPDSTICB'  # Add ingredient grid

        for y in range(height):
            assert is_not_free(grid[y][0]), 'Left border must not be free'
            assert is_not_free(grid[y][-1]), 'Right border must not be free'
        for x in range(width):
            assert is_not_free(grid[0][x]), 'Top border must not be free'
            assert is_not_free(grid[-1][x]), 'Bottom border must not be free'

        all_elements = [element for row in grid for element in row]
        digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        layout_digits = [e for e in all_elements if e in digits]
        num_players = len(layout_digits)
        assert num_players > 0, "No players (digits) in grid"
        layout_digits = list(sorted(map(int, layout_digits)))
        assert layout_digits == list(range(1, num_players + 1)), "Some players were missing"

        # add ingredient
        assert all(c in 'XOPCDSTIB123456789 ' for c in all_elements), 'Invalid character in grid'
        assert all_elements.count('1') == 1, "'1' must be present exactly once"
        assert all_elements.count('D') >= 1, "'D' must be present at least once"
        assert all_elements.count('S') >= 1, "'S' must be present at least once"
        assert all_elements.count('P') >= 1, "'P' must be present at least once"
        assert all_elements.count('O') >= 1 or all_elements.count('T') >= 1 or all_elements.count('I') >= 0, "'O' or 'T' or 'I' must be present at least once"

    #####################
    # TERMINAL GRAPHICS #
    #####################

    
    
    def state_string(self, state):
        """String representation of the current state"""
        players_dict = {player.position: player for player in state.players}

        grid_string = ""
        for y, terrain_row in enumerate(self.terrain_mtx):
            for x, element in enumerate(terrain_row):
                grid_string_add = ""
                if (x, y) in players_dict.keys():
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS

                    grid_string_add += Action.ACTION_TO_CHAR[orientation]
                    player_object = player.held_object
                    # if player_object:
                    #     grid_string_add += player_object.name[:1]
                    # else:
                    #     player_idx_lst = [i for i, p in enumerate(state.players) if p.position == player.position]
                    #     assert len(player_idx_lst) == 1
                    #     grid_string_add += str(player_idx_lst[0])
                    player_idx_lst = [i for i, p in enumerate(state.players) if p.position == player.position]
                    assert len(player_idx_lst) == 1
                    grid_string_add += str(player_idx_lst[0])
                    if player_object:
                        grid_string_add += player_object.name[:1]
                else:
                    if element == "X" and state.has_object((x, y)):
                        state_obj = state.get_object((x, y))
                        grid_string_add = grid_string_add + element + state_obj.name[:1]

                    elif element == "P" and state.has_object((x, y)):
                        soup_obj = state.get_object((x, y))
                        soup_type, num_items, cook_time = soup_obj.state
                        # print(soup_type)
                        grid_string_add += "ø"
                        # if soup_type == "onion":
                        #     grid_string_add += "ø"
                        # elif soup_type == "tomato":
                        #     grid_string_add += "†"
                        # elif soup_type == 'egg':
                        #     grid_string_add += "T"
                        # else:
                        #     raise ValueError()

                        if cook_time != 0:
                            grid_string_add += str(cook_time)
                        
                        # NOTE: do not currently have terminal graphics 
                        # support for cooking times greater than 3.
                        elif num_items == 2:
                            grid_string_add += "="
                        else:
                            grid_string_add += "-"
                    
                    elif element == 'C' and state.has_object((x, y)):
                        soup_obj = state.get_object((x, y))
                        soup_type, num_items, cook_time = soup_obj.state
                        # print(soup_type)
                        grid_string_add += "C"

                        if cook_time != 0:
                            grid_string_add += str(cook_time)
                        
                        # NOTE: do not currently have terminal graphics 
                        # support for cooking times greater than 3.
                        elif num_items == 2:
                            grid_string_add += "="
                        else:
                            grid_string_add += "-"
                    elif element == 'B' and state.has_object((x, y)):
                        soup_obj = state.get_object((x, y))
                        soup_type, num_items, cook_time = soup_obj.state
                        # print(soup_type)
                        grid_string_add += "B"

                        if cook_time != 0:
                            grid_string_add += str(cook_time)
                        
                        # NOTE: do not currently have terminal graphics 
                        # support for cooking times greater than 3.
                        elif num_items == 2:
                            grid_string_add += "="
                        else:
                            grid_string_add += "-"
                    elif element == 'O' and state.has_object((x, y)):
                        soup_obj = state.get_object((x, y))
                        soup_type, num_items, cook_time = soup_obj.state
                        # print(soup_type)
                        grid_string_add += "O"

                        if cook_time != 0:
                            grid_string_add += str(cook_time)
                        
                        # NOTE: do not currently have terminal graphics 
                        # support for cooking times greater than 3.
                        elif num_items == 2:
                            grid_string_add += "="
                        else:
                            grid_string_add += "-"
                    else:
                        grid_string_add += element + " "

                grid_string += grid_string_add
                grid_string += "".join([" "] * (7 - len(grid_string_add)))
                grid_string += " "

            grid_string += "\n\n"
        
        # if state.order_list is not None:
        #     grid_string += "Current orders: {}/{} are any's\n".format(
        #         len(state.order_list), len([order == "any" for order in state.order_list])
        #     )
        return grid_string

    ###################
    # STATE ENCODINGS #
    ###################

    def lossless_state_encoding(self, overcooked_state, debug=False):
        """Featurizes a OvercookedState object into a stack of boolean masks that are easily readable by a CNN"""
        assert type(debug) is bool
        base_map_features = ["pot_loc", "counter_loc", "onion_disp_loc", "dish_disp_loc", "serve_loc"]
        variable_map_features = ["onions_in_pot", "onions_cook_time", "onion_soup_loc", "dishes", "onions"]

        all_objects = overcooked_state.all_objects_list

        def make_layer(position, value):
                layer = np.zeros(self.shape)
                layer[position] = value
                return layer

        def process_for_player(primary_agent_idx):
            # Ensure that primary_agent_idx layers are ordered before other_agent_idx layers
            other_agent_idx = 1 - primary_agent_idx
            ordered_player_features = ["player_{}_loc".format(primary_agent_idx), "player_{}_loc".format(other_agent_idx)] + \
                        ["player_{}_orientation_{}".format(i, Direction.DIRECTION_TO_INDEX[d])
                        for i, d in itertools.product([primary_agent_idx, other_agent_idx], Direction.ALL_DIRECTIONS)]

            LAYERS = ordered_player_features + base_map_features + variable_map_features
            state_mask_dict = {k:np.zeros(self.shape) for k in LAYERS}

            # MAP LAYERS
            for loc in self.get_counter_locations():
                state_mask_dict["counter_loc"][loc] = 1

            for loc in self.get_pot_locations():
                state_mask_dict["pot_loc"][loc] = 1

            for loc in self.get_onion_dispenser_locations():
                state_mask_dict["onion_disp_loc"][loc] = 1

            for loc in self.get_dish_dispenser_locations():
                state_mask_dict["dish_disp_loc"][loc] = 1

            for loc in self.get_serving_locations():
                state_mask_dict["serve_loc"][loc] = 1

            # PLAYER LAYERS
            for i, player in enumerate(overcooked_state.players):
                player_orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
                state_mask_dict["player_{}_loc".format(i)] = make_layer(player.position, 1)
                state_mask_dict["player_{}_orientation_{}".format(i, player_orientation_idx)] = make_layer(player.position, 1)

            # OBJECT & STATE LAYERS
            for obj in all_objects:
                if obj.name == "soup":
                    soup_type, num_onions, cook_time = obj.state
                    if soup_type == "onion":
                        if obj.position in self.get_pot_locations():
                            soup_type, num_onions, cook_time = obj.state
                            state_mask_dict["onions_in_pot"] += make_layer(obj.position, num_onions)
                            state_mask_dict["onions_cook_time"] += make_layer(obj.position, cook_time)
                        else:
                            # If player soup is not in a pot, put it in separate mask
                            state_mask_dict["onion_soup_loc"] += make_layer(obj.position, 1)
                    else:
                        raise ValueError("Unrecognized soup")

                elif obj.name == "dish":
                    state_mask_dict["dishes"] += make_layer(obj.position, 1)
                elif obj.name == "onion":
                    state_mask_dict["onions"] += make_layer(obj.position, 1)
                else:
                    raise ValueError("Unrecognized object")

            if debug:
                print(len(LAYERS))
                print(len(state_mask_dict))
                for k, v in state_mask_dict.items():
                    print(k)
                    print(np.transpose(v, (1, 0)))

            # Stack of all the state masks, order decided by order of LAYERS
            state_mask_stack = np.array([state_mask_dict[layer_id] for layer_id in LAYERS])
            state_mask_stack = np.transpose(state_mask_stack, (1, 2, 0))
            assert state_mask_stack.shape[:2] == self.shape
            assert state_mask_stack.shape[2] == len(LAYERS)
            # NOTE: currently not including time left or order_list in featurization
            return np.array(state_mask_stack).astype(int)

        # NOTE: Currently not very efficient, a decent amount of computation repeated here
        num_players = len(overcooked_state.players)
        final_obs_for_players = tuple(process_for_player(i) for i in range(num_players))
        return final_obs_for_players

    def featurize_state(self, overcooked_state, mlp):
        """
        Encode state with some manually designed features. 
        NOTE: currently works for just two players.
        """

        all_features = {}

        def make_closest_feature(idx, name, locations):
            "Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict"
            all_features["p{}_closest_{}".format(idx, name)] = self.get_deltas_to_closest_location(player, locations, mlp)

        IDX_TO_OBJ = ["onion", "soup", "dish"]
        OBJ_TO_IDX = { o_name:idx for idx, o_name in enumerate(IDX_TO_OBJ) }

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_state = self.get_pot_states(overcooked_state)

        # Player Info
        for i, player in enumerate(overcooked_state.players):
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[orientation_idx]
            obj = player.held_object
            
            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[obj_idx]

            # Closest feature of each type
            if held_obj_name == "onion":
                all_features["p{}_closest_onion".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "onion", self.get_onion_dispenser_locations() + counter_objects["onion"])

            make_closest_feature(i, "empty_pot", pot_state["empty"])
            make_closest_feature(i, "one_onion_pot", pot_state["onion"]["one_onion"])
            make_closest_feature(i, "two_onion_pot", pot_state["onion"]["two_onion"])
            make_closest_feature(i, "cooking_pot", pot_state["onion"]["cooking"])
            make_closest_feature(i, "ready_pot", pot_state["onion"]["ready"])

            if held_obj_name == "dish":
                all_features["p{}_closest_dish".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "dish", self.get_dish_dispenser_locations() + counter_objects["dish"])

            if held_obj_name == "soup":
                all_features["p{}_closest_soup".format(i)] = (0, 0)
            else:
                make_closest_feature(i, "soup", counter_objects["soup"])

            make_closest_feature(i, "serving", self.get_serving_locations())

            for direction, pos_and_feat in enumerate(self.get_adjacent_features(player)):
                adj_pos, feat = pos_and_feat

                if direction == player.orientation:
                    # Check if counter we are facing is empty
                    facing_counter = (feat == 'X' and adj_pos not in overcooked_state.objects.keys())
                    facing_counter_feature = [1] if facing_counter else [0]
                    all_features["p{}_facing_empty_counter".format(i)] = facing_counter_feature

                all_features["p{}_wall_{}".format(i, direction)] = [0] if feat == ' ' else [1]

        features_np = { k:np.array(v) for k, v in all_features.items() }
        
        p0, p1 = overcooked_state.players
        p0_dict = { k:v for k,v in features_np.items() if k[:2] == "p0" }
        p1_dict = { k:v for k,v in features_np.items() if k[:2] == "p1" }
        p0_features = np.concatenate(list(p0_dict.values()))
        p1_features = np.concatenate(list(p1_dict.values()))

        p1_rel_to_p0 = np.array(pos_distance(p1.position, p0.position))
        abs_pos_p0 = np.array(p0.position)
        ordered_features_p0 = np.squeeze(np.concatenate([p0_features, p1_features, p1_rel_to_p0, abs_pos_p0]))

        p0_rel_to_p1 = np.array(pos_distance(p0.position, p1.position))
        abs_pos_p1 = np.array(p0.position)
        ordered_features_p1 = np.squeeze(np.concatenate([p1_features, p0_features, p0_rel_to_p1, abs_pos_p1]))
        return ordered_features_p0, ordered_features_p1

    def get_deltas_to_closest_location(self, player, locations, mlp):
        _, closest_loc = mlp.mp.min_cost_to_feature(player.pos_and_or, locations, with_argmin=True)
        #_, closest_loc = mlp.motion_planner.min_cost_to_feature(player.pos_and_or, locations, with_argmin=True)
        if closest_loc is None:
            # "any object that does not exist or I am carrying is going to show up as a (0,0)
            # but I can disambiguate the two possibilities by looking at the features 
            # for what kind of object I'm carrying"
            return (0, 0)
        dy_loc, dx_loc = pos_distance(closest_loc, player.position)
        return dy_loc, dx_loc

    ##############
    # DEPRECATED #
    ##############

    def calculate_distance_based_shaped_reward(self, state, new_state):
        """
        Adding reward shaping based on distance to certain features.
        """
        distance_based_shaped_reward = 0
        
        pot_states = self.get_pot_states(new_state)
        ready_pots = pot_states["tomato"]["ready"] + pot_states["onion"]["ready"]
        cooking_pots = ready_pots + pot_states["tomato"]["cooking"] + pot_states["onion"]["cooking"]
        nearly_ready_pots = cooking_pots + pot_states["tomato"]["partially_full"] + pot_states["onion"]["partially_full"]
        dishes_in_play = len(new_state.player_objects_by_type['dish'])
        for player_old, player_new in zip(state.players, new_state.players):
            # Linearly increase reward depending on vicinity to certain features, where distance of 10 achieves 0 reward
            max_dist = 8

            if player_new.held_object is not None and player_new.held_object.name == 'dish' and len(nearly_ready_pots) >= dishes_in_play:
                min_dist_to_pot_new = np.inf
                min_dist_to_pot_old = np.inf
                for pot in nearly_ready_pots:
                    new_dist = np.linalg.norm(np.array(pot) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(pot) - np.array(player_old.position))
                    if new_dist < min_dist_to_pot_new:
                        min_dist_to_pot_new = new_dist
                    if old_dist < min_dist_to_pot_old:
                        min_dist_to_pot_old = old_dist
                if min_dist_to_pot_old > min_dist_to_pot_new:
                    distance_based_shaped_reward += self.reward_shaping_params["POT_DISTANCE_REW"] * (1 - min(min_dist_to_pot_new / max_dist, 1))

            if player_new.held_object is None and len(cooking_pots) > 0 and dishes_in_play == 0:
                min_dist_to_d_new = np.inf
                min_dist_to_d_old = np.inf
                for serving_loc in self.terrain_pos_dict['D']:
                    new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
                    if new_dist < min_dist_to_d_new:
                        min_dist_to_d_new = new_dist
                    if old_dist < min_dist_to_d_old:
                        min_dist_to_d_old = old_dist

                if min_dist_to_d_old > min_dist_to_d_new:
                    distance_based_shaped_reward += self.reward_shaping_params["DISH_DISP_DISTANCE_REW"] * (1 - min(min_dist_to_d_new / max_dist, 1))

            if player_new.held_object is not None and player_new.held_object.name == 'soup':
                min_dist_to_s_new = np.inf
                min_dist_to_s_old = np.inf
                for serving_loc in self.terrain_pos_dict['S']:
                    new_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_new.position))
                    old_dist = np.linalg.norm(np.array(serving_loc) - np.array(player_old.position))
                    if new_dist < min_dist_to_s_new:
                        min_dist_to_s_new = new_dist

                    if old_dist < min_dist_to_s_old:
                        min_dist_to_s_old = old_dist
                
                if min_dist_to_s_old > min_dist_to_s_new:
                    distance_based_shaped_reward += self.reward_shaping_params["SOUP_DISTANCE_REW"] * (1 - min(min_dist_to_s_new / max_dist, 1))

        return distance_based_shaped_reward
