import os
import json
import time
import pandas as pd
from argparse import ArgumentParser

models = ["gpt-4o"]

orders = ["baked_bell_pepper", "baked_sweet_potato", "boiled_egg", "boiled_mushroom", "boiled_sweet_potato", "baked_potato_slices", "baked_pumpkin_slices", "boiled_corn_slices", "boiled_green_bean_slices", "boiled_potato_slices", "baked_bell_pepper_soup", "baked_carrot_soup", "baked_mushroom_soup", "baked_potato_soup", "baked_pumpkin_soup", "sliced_bell_pepper_and_corn_stew", "sliced_bell_pepper_and_lentil_stew", "sliced_eggplant_and_chickpea_stew", "sliced_pumpkin_and_chickpea_stew", "sliced_zucchini_and_chickpea_stew", "mashed_broccoli_and_bean_patty", "mashed_carrot_and_chickpea_patty", "mashed_cauliflower_and_lentil_patty", "mashed_potato_and_pea_patty", "mashed_sweet_potato_and_bean_patty", "potato_carrot_and_onion_patty", "romaine_lettuce_pea_and_tomato_patty", "sweet_potato_spinach_and_mushroom_patty", "taro_bean_and_bell_pepper_patty", "zucchini_green_pea_and_onion_patty"]

def main(variant):

    order = variant['order']
    eval_result_dir = 'eval_result' + '/' + variant['model']
    
    order_dir = os.path.join(eval_result_dir, order)
    eval_file = os.path.join(order_dir, 'evaluation_result.json')
    
    if not os.path.exists(eval_file):
        print(f"Error: File {eval_file} not found.")
        return

    with open(eval_file, 'r') as file:
        data = json.load(file)
    
    if order not in data:
        print(f"Error: Key '{order}' not found in evaluation_result.json.")
        return
    
    order_data = data[order]
    average = order_data.get('average', {})
    task_metrics = order_data.get('task_metrics', {})
    statistic = order_data.get('statistic', {})

    excel_path = os.path.join('eval_result', 'statistics_data.csv')
    
    if os.path.exists(excel_path):
        df = pd.read_csv(excel_path)
    else:
        df = pd.DataFrame(columns=['model',
                                    'order', 
                                    'success_rate', 
                                    'time_avg', 
                                    'time_var',
                                    'mean_f1_agent_0', 
                                    'mean_similarity_agent_0',
                                    'mean_redundancy_agent_0',
                                    'std_f1_agent_0', 
                                    'std_similarity_agent_0',
                                    'std_redundancy_agent_0',
                                    'mean_f1_agent_1', 
                                    'mean_similarity_agent_1',
                                    'mean_redundancy_agent_1',
                                    'std_f1_agent_1', 
                                    'std_similarity_agent_1',
                                    'std_redundancy_agent_1',
                                    'initiate_collaboration',
                                    'respond_collaboration',
                                    'overall_collaboration'])
    
    new_row = pd.DataFrame([{
    'model': variant['model'],
    'order': order,
    'success_rate': task_metrics['success_rate'],
    'time_avg': task_metrics['time_avg'],
    'time_var': task_metrics['time_var'],
    'mean_f1_agent_0': average['similarity_and_redundancy']['agent_0']['mean_f1'],
    'mean_similarity_agent_0': average['similarity_and_redundancy']['agent_0']['mean_similarity'],
    'mean_redundancy_agent_0': average['similarity_and_redundancy']['agent_0']['mean_redundancy'],
    'std_f1_agent_0': average['similarity_and_redundancy']['agent_0']['std_f1'],
    'std_similarity_agent_0': average['similarity_and_redundancy']['agent_0']['std_similarity'],
    'std_redundancy_agent_0': average['similarity_and_redundancy']['agent_0']['std_redundancy'],
    'mean_f1_agent_1': average['similarity_and_redundancy']['agent_1']['mean_f1'],
    'mean_similarity_agent_1': average['similarity_and_redundancy']['agent_1']['mean_similarity'],
    'mean_redundancy_agent_1': average['similarity_and_redundancy']['agent_1']['mean_redundancy'],
    'std_f1_agent_1': average['similarity_and_redundancy']['agent_1']['std_f1'],
    'std_similarity_agent_1': average['similarity_and_redundancy']['agent_1']['std_similarity'],
    'std_redundancy_agent_1': average['similarity_and_redundancy']['agent_1']['std_redundancy'],
    'initiate_collaboration': statistic['initiate_collaboration'],
    'respond_collaboration': statistic['respond_collaboration']
}])

    df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_csv(excel_path, index=False)
    print(f"Data for order '{order}' saved to {excel_path}.")


def boolean_argument(value):
    """Helper function to parse boolean arguments."""
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', '0', 'no'}:
        return False
    elif value.lower() in {'true', '1', 'yes'}:
        return True
    else:
        raise ValueError(f"Invalid boolean value: {value}")


if __name__ == '__main__':
    parser = ArgumentParser(description='Process evaluation results and update statistics data.')

    parser.add_argument('--model', type=str, default='gpt-3.5', help='Number of episodes')
    parser.add_argument('--order', type=str, default='AUTO', help='Task order name, "AUTO" represents automatic recognition.')
    args = parser.parse_args()
    variant = vars(args)

    start_time = time.time()
    for model in models:
        for order in orders:
            variant['model'] = model
            variant['order'] = order
            main(variant)
    end_time = time.time()
    print("\n======= Finished all =======\n")
    print(f"Cost time: {end_time - start_time:.3f}s\n")
