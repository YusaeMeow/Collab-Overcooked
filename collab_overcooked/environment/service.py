from collab import web_util 
from argparse import ArgumentParser



if __name__ == '__main__':

    '''
    python main.py --layout cramped_room --p0 Greedy --p1 Greedy --horizon 100
    python main.py --layout cramped_room --p0 LLMPair --p1 LLMPair --horizon 400 -pl l2-ap
    python main.py --layout counter_circuit --p0 LLMPair --p1 Greedy --gpt_model gpt-3.2-turbo-0301 --prompt_level l2-ap --retrival_method recent_k --K 1
    python main.py --layout cramped_room --p0 LLMPair --p1 LLMPair --horizon 400 -pl l2-ap -l forced_coordination

    python main.py --layout new_env --p0 LLMPair --p1 LLMPair --horizon 400 -pl l2-ap --mode develop
    python main.py --layout new_env --p0 LLMPair --p1 LLMPair --horizon 400 -pl l2-ap --mode exp --gpt_model llama3:70b-instruct-fp16

    python main.py --layout new_env --p0 LLMPair --p1 LLMPair --horizon 70 -pl l2-ap --mode exp
    '''
    #os.system("clear")
    parser = ArgumentParser(description='OvercookedAI Experiment')

    # these are basis parses
    parser.add_argument('--port', type=str)

    args = parser.parse_args()
    variant = vars(args)
    port = variant['port']
    web_util.start_server(port)