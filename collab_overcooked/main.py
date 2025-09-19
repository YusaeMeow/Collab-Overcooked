import time
import datetime
import os
import json
import datetime
from argparse import ArgumentParser
import numpy as np
from rich import print as rprint
import copy
from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
work_dir = os.getcwd()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*")

from distutils.util import strtobool

def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

def check_recipe_parse(variant):
    recipe_name_list = os.listdir(PROMPT_DIR+'/recipe/') 
    recipe_filename = ""
    for r in recipe_name_list:
        if variant['order'] in r.lower():
            recipe_filename = r
            break
    if recipe_filename == "":
        raise ValueError("Not valid order name!")
    else:
        return True

# Load YAML for new configuration system
try:
    import yaml
except ImportError:
    print("PyYAML not installed. Install with: pip install PyYAML")
    yaml = None

import importlib_metadata
VERSION = importlib_metadata.version("overcooked_ai")
cwd = os.getcwd()
PROMPT_DIR = os.path.join(cwd, "prompts")
print(f'\n----This overcook version is {VERSION}----\n')

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action

# Import from new modular system
try:
    from .agents import statistics_dict, turn_statistics_dict
    from .agents.web_util import output_to_port, check_port_in_use, change_port
    from .utils import make_agent, get_example_embedding, combine_statistic_dict
    
    # Define make_agent_from_config for new system
    def make_agent_from_config(agent_config, mdp, layout):
        """Create agent from YAML configuration using existing LLMAgents"""
        from .agents.collab import LLMAgents
        from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
        
        # Create planner
        mlam = MediumLevelPlanner.from_pickle_or_compute(
            mdp, NO_COUNTERS_PARAMS, force_compute=False
        )
        
        # Map role to actor name
        role = agent_config.get('role', 'Chef')
        actor = 'chef' if role.lower() == 'chef' else 'assistant'
        
        # Create LLMAgents with config
        agent = LLMAgents(
            mlam, 
            layout, 
            model=agent_config.get('model', 'gpt-3.5-turbo'),
            model_dirname=agent_config.get('model_dirname', '~/'),
            local_server_api=agent_config.get('base_url', 'http://localhost:8000/v1'),
            retrival_method=agent_config.get('retrieval_method', 'recent_k'),
            K=agent_config.get('history_k', 1),
            actor=actor
        )
        
        # Set API key and other config if needed
        if agent_config.get('api_key'):
            agent.api_key = agent_config['api_key']
        
        return agent
except ImportError:
    # Fallback to old system  
    from .agents.modules import statistics_dict, turn_statistics_dict
    from .agents.web_util import output_to_port, check_port_in_use, change_port
    from .utils import make_agent, get_example_embedding, combine_statistic_dict
    make_agent_from_config = None

import socket


def load_config_from_yaml(config_path):
    """Load configuration from YAML file"""
    if not yaml:
        raise ImportError("PyYAML is required for YAML configuration. Install with: pip install PyYAML")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def convert_yaml_to_variant(config):
    """Convert YAML config to old-style variant dict"""
    env_config = config.get('environment', {})
    agents_config = config.get('agents', {})
    
    # Get model from first agent config
    gpt_model = 'gpt-3.5-turbo'  # default
    if agents_config:
        first_agent = next(iter(agents_config.values()))
        if isinstance(first_agent, dict) and 'model' in first_agent:
            gpt_model = first_agent['model']
    
    variant = {
        'layout': env_config.get('layout', 'cramped_room'),
        'horizon': env_config.get('horizon', 10),
        'order': env_config.get('order', 'boiled_egg'),
        'episode': 1,  # Single episode for now
        'mode': 'exp',
        'test_mode': 'single_task',  # Default test mode
        'p0': 'LLMPair',
        'p1': 'LLMPair',
        'gpt_model': gpt_model,
        'agent_configs': agents_config,
        'use_new_system': True
    }
    
    return variant


def main(variant=None, config_path=None):
    """
    Main function supporting both old variant dict and new YAML config
    """
    
    # Handle new YAML configuration
    if config_path:
        config = load_config_from_yaml(config_path)
        variant = convert_yaml_to_variant(config)
        variant['yaml_config'] = config
    
    if variant is None:
        raise ValueError("Either variant dict or config_path must be provided")

    layout = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']

    mode = variant.get('mode', 'exp')
    
    mdp = OvercookedGridworld.from_layout_name(layout)

    #set order according to parser
    if variant['order'] !="" and check_recipe_parse(variant):
        mdp.start_order_list = [variant['order']]
        # 1 task mode
        mdp.one_task_mode = True

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()

    
    p0_algo = variant.get('p0', 'LLMPair')
    p1_algo = variant.get('p1', 'LLMPair')
    print(f"\n===P0 agent: {p0_algo} | P1 agent: {p1_algo}===\n")

    start_time = time.time()
    results = []

    actor_num = 0
    actor_list = ['chef','assistant']
    for i in range(episode):  
        
        agents_list = []

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Handle save directory for new config system
        if variant.get('use_new_system'):
            save_dir = f"results/{current_time}_{variant['order']}"
        else:
            save_dir = f"{args.statistics_save_dir}/{args.gpt_model}/{args.order}"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = f"{save_dir}/experiment_{current_time}_{variant['order']}.json"

        if mode == 'develop':
            """
            You can customize the 'action_list' and 'parm' to test the environment
            """
            action_list = []
            parm = []

            env.reset()
            r_total = 0
            for t in range(horizon):
                s_t = env.state
                # print(s_t.timestep, env.t)
                print(f'\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n')
                print(env.mdp.state_string(s_t).replace('ø', 'o'))


                obs, reward, done, env_info = env.step(action_list[t], parm[t])
                print(env.mdp.get_utensil_states(s_t))
                ml_actions = obs.ml_actions
                skills = f""
                for i, ml_action in enumerate(ml_actions):
                    if ml_action == None:
                        continue
                    skills += f"P{i} finished <{ml_action}>. "
                print(skills)

                r_total += reward
                rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')
            break

        
        # Create agents - support both old and new systems
        if variant.get('use_new_system') and make_agent_from_config:
            # Use new configuration system
            agent_configs = variant.get('agent_configs', {})
            for i, (agent_id, agent_config) in enumerate(agent_configs.items()):
                if agent_id.startswith('agent_'):
                    print(f"\n----Use {agent_config.get('model', 'unknown')} ({agent_config.get('type', 'unknown')})----\n")
                    agent = make_agent_from_config(agent_config, mdp, layout)
                    agents_list.append(agent)
        else:
            # Use old system
            for alg in [p0_algo, p1_algo]:
                if alg == "LLMPair":
                    if mode!="human":
                        assert variant.get('gpt_model') is not None, print(f'you should choose a gpt model')
                    if mode == "OpenSource":
                        assert os.path.exists(variant.get('model_dirname', '')), print(f"you should input right open-source model absolute path")
                    print(f"\n----Use {variant.get('gpt_model', 'gpt-3.5-turbo')}----\n")
                    if variant.get('gpt_model') == "human":
                        assert check_port_in_use(variant.get("local_server_api", "http://localhost:8080")), print(f"port {variant.get('local_server_api', 'http://localhost:8080')} is busy")
                        change_port(variant.get("local_server_api", "http://localhost:8080"))
                    
                    gpt_model = variant.get('gpt_model', 'gpt-3.5-turbo')
                    model_dirname = variant.get('model_dirname', '~/')
                    local_server_api = variant.get('local_server_api', 'http://localhost:8000/v1')
                    retrival_method = variant.get('retrival_method', 'recent_k')
                    K = variant.get('K', 3)
                    
                    agent = make_agent(alg, mdp, layout, model=gpt_model, model_dirname=model_dirname,
                                     local_server_api=local_server_api, retrival_method=retrival_method, 
                                     K=K, actor=actor_list[actor_num])
                else:
                    agent = make_agent(alg, mdp, layout)
                agents_list.append(agent)
                actor_num += 1

        team = AgentGroup(*agents_list)
        team.reset()

        env.reset()
        r_total = 0

        
        if mode == 'exp':
            for t in range(horizon):
                s_t = env.state
                # print(s_t.timestep, env.t)
                print(f'\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n')
                map = env.mdp.state_string(s_t).replace('ø', 'o')
                print(map)   
                a_t, ingredient_for_pickup = team.joint_action(s_t) 
                print(a_t)
                dialogue_t = team.reset_dialogue()
                print(f"\n-----------Controller-----------\n")    
                print(f"action: P0 {Action.to_char(a_t[0])} | P1 {Action.to_char(a_t[1])}")
                parm = ingredient_for_pickup

                obs, reward, done, env_info = env.step(a_t,parm)

                ml_actions = obs.ml_actions
                skills = f""
                for i, ml_action in enumerate(ml_actions):
                    if ml_action == None:
                        continue
                    skills += f"P{i} finished <{ml_action}>. "
                print(skills)

                r_total += reward
                if reward>0:
                    statistics_dict['total_order_finished'].append(s_t.current_k_order[0])
                    team.agents[1].teammate_ml_actions.append({'timestamp':t,'action':"deliver_soup()"})
                rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')
                print(f"P0's real behavior: {team.agents[1].teammate_ml_actions}")
                print(f"P1's real behavior: {team.agents[0].teammate_ml_actions}")


                #save statistics 
                turn_statistics_dict_agent0 = team.agents[0].turn_statistics_dict
                turn_statistics_dict_agent1 = team.agents[1].turn_statistics_dict

                turn_statistics_dict_both = combine_statistic_dict(turn_statistics_dict_agent0,turn_statistics_dict_agent1,map,reward)

                statistics_dict['total_timestamp'].append(t)
                statistics_dict['total_score'] = r_total
                statistics_dict['total_action_list'][0] = team.agents[1].teammate_ml_actions
                statistics_dict['total_action_list'][1] = team.agents[0].teammate_ml_actions
                statistics_dict['content'].append(turn_statistics_dict_both)
                #statistics_dict['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(filename, 'w') as f:
                    json.dump(statistics_dict,f,indent=4)
                
                if variant['test_mode'] == 'fix_task':
                    if reward != 0:
                        print("Task successed!")
                        #Human-eval: set task success message
                        if variant['gpt_model'] == "human":
                            for a in range(len(team.agents)):
                                output_to_port(f"agent{a}","Success!",mission="success",port=variant['local_server_api'])
                        break
            #Human-eval: set task failed message
            if variant['gpt_model'] == "human":
                for a in range(len(team.agents)):
                    output_to_port(f"agent{a}","Fail to finish task in time!",mission="fail",port=variant['local_server_api'])
        print(f"Episode {i+1}/{episode}: {r_total}\n====\n\n")
        results.append(r_total)
   
    end_time = time.time()
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")


    
if __name__ == '__main__':

    parser = ArgumentParser(description='OvercookedAI Experiment')

    # these are basis parses
    parser.add_argument('--layout', '-l', type=str, default='new_env', choices=['new_env'])
    parser.add_argument('--p0',  type=str, default='LLMPair', choices=['LLMPair', 'Human'], help='Algorithm for P0 agent 0')
    parser.add_argument('--p1', type=str, default='LLMPair', choices=['LLMPair', 'Human'], help='Algorithm for P1 agent 1')
    parser.add_argument('--horizon', type=int, default=120, help='Horizon steps in one game')
    parser.add_argument('--episode', type=int, default=1, help='Number of episodes')

    # these parsers are only required when using LLMPair.

    # model:'gpt-3.5-turbo-0125', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4o','gpt-o1mini','gpt4-turbo','llama3-8B','Llama-3.1-8B-Instruct','Llama-3.1-70B-Instruct',"Yi-1.2-34B","yi-lightning","yi-large",'yi-medium',"Qwen2.5-7B-Instruct","Qwen2.5-72B-Instruct","Qwen2.5-14B-Instruct","Qwen2.5-32B-Instruct",'claude3_sonnet'
    parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo-0125')
    
    parser.add_argument('--retrival_method', type=str, default="recent_k", choices=['recent_k', 'bert_topk'], help='Use similarity-based(BERT, CLIP) retrieval or retrieve recent K history in dialog.')
    parser.add_argument('--K', type=int, default=0, help="The number of dialogues you want to retrieve.")

    # 
    parser.add_argument('--model_dirname', type=str, default='.', help='absolute path of open-source model')      
    parser.add_argument('--local_server_api', type=str, default= "http://localhost:8000/v1", help='IP and port address to connect with local open source llm')     
    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'debug_validator', 'develop'], help='exp mode run step-by-step, demo mode run via traj')                                
    parser.add_argument('--test_mode', type=str, default='fix_task', choices=['fix_task', 'fix_time'])
    parser.add_argument('--save', type=boolean_argument, default=True, help='Whether save the result')
    parser.add_argument('--log_dir', type=str, default=None, help='dir to save result')
    parser.add_argument('--debug', type=boolean_argument, default=True, help='debug mode')
    parser.add_argument('--order', type=str, default="", help='1 task order name')

    #
    parser.add_argument('--statistics_save_dir', type=str, default='data', help='save directory of LLM statistics')


    args = parser.parse_args()
    variant = vars(args)

    start_time = time.time()
    main(variant)
    end_time = time.time()
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")
