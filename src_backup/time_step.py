import os 
import json
import pandas as pd
from pathlib import Path
import numpy as np
import copy as cp

model_list = ["deepseek-ai/DeepSeek-V3"]
order_list = {"baked_bell_pepper":16,"baked_sweet_potato":16,"boiled_egg":16,"boiled_mushroom":16,"boiled_sweet_potato":16, #18
              "baked_potato_slices":23,"baked_pumpkin_slices":23,"boiled_corn_slices":23,"boiled_green_bean_slices":23,"boiled_potato_slices":23, #27
              "baked_bell_pepper_soup":35,"baked_carrot_soup":35,"baked_mushroom_soup":35,"baked_potato_soup":35,"baked_pumpkin_soup":35, #39
              "sliced_bell_pepper_and_corn_stew":32,"sliced_bell_pepper_and_lentil_stew":32,"sliced_eggplant_and_chickpea_stew":32,"sliced_pumpkin_and_chickpea_stew":32,"sliced_zucchini_and_chickpea_stew":32, #34
              "mashed_broccoli_and_bean_patty":55,"mashed_carrot_and_chickpea_patty":55,"mashed_cauliflower_and_lentil_patty":55,"mashed_potato_and_pea_patty":55,"mashed_sweet_potato_and_bean_patty":55,#62
              "potato_carrot_and_onion_patty":64,"romaine_lettuce_pea_and_tomato_patty":64,"sweet_potato_spinach_and_mushroom_patty":64,"taro_bean_and_bell_pepper_patty":64,"zucchini_green_pea_and_onion_patty":64} #73

statistic_dict = {}
for model in model_list:
    statistic_dict[model] = {}


def get_all_files(directory):
    file_paths = []  
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Construct the absolute path to the file and add it to the list
            file_paths.append(os.path.join(root, file))
    
    return file_paths

def find_average_success_time():
    for model in model_list:
        dir = f"./src/data/{model}/"
        if not os.path.exists(dir):
            print(f"{model} log not exist")
        files = get_all_files(dir)
        #print(f"{model} log file number:{len(files)}")
        for file in files:
            with open(file, 'r') as file_handler:
                data = json.load(file_handler)
                #Finished_order
                if data["total_order_finished"]!=[]:
                    order = data["total_order_finished"][0]
                    if order not in statistic_dict[model].keys():
                        statistic_dict[model][order] = {"success_time":[]}
                    statistic_dict[model][order]["success_time"].append(int(data["total_timestamp"][-1]))
        # Calculate average time
        for order in statistic_dict[model].keys():
            if len(statistic_dict[model][order]["success_time"])==0:
                statistic_dict[model][order]["average"]="NAN"
            else:
                statistic_dict[model][order]["average"] = sum(statistic_dict[model][order]["success_time"])/len(statistic_dict[model][order]["success_time"])
            temp = statistic_dict[model][order]["success_time"]
            del statistic_dict[model][order]["success_time"]
            statistic_dict[model][order]["success_time"] = temp
    df = pd.DataFrame(statistic_dict)
    excel_file_path = 'timesteps.xlsx'
    df.index = list(order_list.keys())
    df.to_excel(excel_file_path, index=False)

    print(f"JSON data has been written to {excel_file_path}")

def success_rate(multiple):
    for model in model_list:
        #initial
        for order in order_list.keys():
            statistic_dict[model][order] = {"success_rate":0,"finish_list":[]}
        
        dir = f"./src/data/{model}/"
        if not os.path.exists(dir):
            print(f"{model} log not exist")
        files = get_all_files(dir)
        for file in files:
            with open(file, 'r') as file_handler:
                data = json.load(file_handler)
                order = file.split("/")[-2]
                time_threshold = multiple*order_list[order]
                #Finished_order
                if data["total_order_finished"]!=[]:
                    if data["total_timestamp"][-1]<= time_threshold:
                        statistic_dict[model][order]['finish_list'].append(1) 
                    else:
                        statistic_dict[model][order]['finish_list'].append(0) 
                else:
                    statistic_dict[model][order]['finish_list'].append(0) 
        for order in order_list.keys():
            if len(statistic_dict[model][order]["finish_list"])==0:
                statistic_dict[model][order]["success_rate"]=0
            else:
                statistic_dict[model][order]['success_rate'] = sum(statistic_dict[model][order]["finish_list"])/len(statistic_dict[model][order]["finish_list"])

    shape = (len(list(order_list.keys())), len(model_list))
    output = np.zeros(shape)
    for i,order in enumerate(list(order_list.keys())):
        for j, model in  enumerate(model_list):
            output[i][j] = statistic_dict[model][order]['success_rate']
    df = pd.DataFrame(output)
    excel_file_path = f'success_rate_{multiple}.xlsx'
    df.index = list(order_list.keys())
    df.to_excel(excel_file_path, index=False)

    print(f"JSON data has been written to {excel_file_path}")

def truncate(multiple):
    for model in model_list:
        #initial
        for order in order_list.keys():
            statistic_dict[model][order] = {"success_rate":0,"finish_list":[]}
        dir = f"./src/data/{model}/"        
        if not os.path.exists(dir):
            print(f"{model} log not exist")        
        files = get_all_files(dir)
        for file in files:
            with open(file, 'r') as file_handler:
                data = json.load(file_handler)
                order = file.split("/")[-2]
            time_threshold = int(multiple*order_list[order])
            #Finished_order
            if not (data["total_order_finished"]!=[] and data["total_timestamp"][-1]<= time_threshold):
            #truncate log
                #part1: timestep
                data["total_timestamp"] = list(range(0,time_threshold))
                #part2: total_action_list
                for agent_index, agent_action_list in enumerate(data["total_action_list"]):
                    temp_action_list = []
                    for a in agent_action_list:
                        if a["timestamp"]<time_threshold:
                            temp_action_list.append(a)
                    data["total_action_list"][agent_index] = cp.copy(temp_action_list)
                #part3: content:
                data["content"]= data["content"][:time_threshold]
                #part4: finish order
                data["total_order_finished"]=[]
            #save file
            output_dir = f"./src/data/truncate_{str(multiple).replace('.','_')}/{model}/{order}/"
            if not os.path.exists(output_dir):
                print(f"Create truncation path:{output_dir}")
                os.makedirs(output_dir)
            with open(f"{output_dir}{file.split('/')[-1]}", 'w') as f:
                json.dump(data,f,indent=4)

truncate(1.5)

#find_average_success_time()
# for i in [1,1.2,1.5,2]:
#     success_rate(i)