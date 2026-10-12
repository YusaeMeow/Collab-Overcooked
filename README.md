<div align="center">
  <h1> Collab-Overcooked </h1>
  

</div>

We propose a new LLM-powered Multi-Agent System (LLM-MAS) benchmark, Collab-Overcooked, built on the popular Overcooked-AI game with more applicable and challenging tasks in interactive environments. Collab-Overcooked extends existing benchmarks from two novel perspectives. First, it provides a multi-agent framework supporting diverse tasks and objectives and encourages collaboration through natural language communication. Second, it introduces a spectrum of process-oriented evaluation metrics to assess the fine-grained collaboration capabilities of different LLM agents, a dimension often overlooked in prior work.

## ⚠️ Upcoming Code Restructure for Improved Clarity
We're preparing to restructure our code to significantly improve its readability and logical organization (expected to take 1-2 months). This update is designed to make it much simpler for researchers to follow and understand.

Important points for you:

- No impact on current experiments: This update will not affect any experimental results. If you're currently testing with Collab-Overcooked, you can continue using it without interruption.

- For incremental work: If you plan to build incrementally on Collab-Overcooked and have questions about the current code, please pay close attention to our upcoming updates.

We believe these changes will greatly enhance your experience working with our codebase!

## Getting Started

### Play with an agent teammate

The optional browser mode lets a human control either chef with the keyboard
while an agent operates the other chef. It uses the existing Python game engine
and does not require the benchmark's embedding models or API keys.

```bash
pip install -r requirements-human.txt
pip install -e lib/overcooked_ai
python src/human_play.py --order boiled_egg
```

Open <http://127.0.0.1:8080>, then press **Start / Resume**. Use **WASD / arrows**
to move, **Space** to interact, and **E** to start a loaded appliance. Select the
ingredient before using its dispenser. Items can be passed on shared counters;
there is no chat channel between the human and the agent.

The default teammate is a deterministic recipe baseline. A chat-completions
model server or custom HTTP policy can be selected without blocking keyboard
input. See [human-agent play](docs/human-play.md) for setup, protocol, recording,
and validation. The original `src/main.py` benchmark entry point is unchanged.

### Install
We recommend using the anaconda management environment. Python 3.8 is recommended for this project.  
- Install requirements
    - Directly do:
        ```
        conda create -n collab-overcooked python=3.8
        conda activate collab-overcooked

        pip install -r requirements.txt
        conda install mpi4py==3.1.4  # pip install often fails
        ```

- Install the game environment `overcooked_ai` locally.
    ```
    cd ./lib/overcooked_ai
    pip install -e .
    ```
    Notes: [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game Overcooked. We made certain modifications based on Overcooked-AI.

### Quick Test
The easiest way to test whether the environment is installed correctly is to use gpt-3.5-turbo to test after filling in the openai api secret key. 
- Fill in the OpenAI API key at `Collab-Overcooked/src/openai_key.txt`
- Run the following commands
  ```
  cd Collab-Overcooked/src
  python main.py --horizon 3 --order boiled_egg
  ```
If you can output the environmental visualization map normally, the agents' normal output content, and run through 3 time steps without any errors, then your environment and agent configuration are successful. This will take you about 1.5 minutes, depending on the speed of your connection to OpenAI.

### Configure the Open-source LLMs
We recommend using [vLLM](https://github.com/vllm-project/vllm) for local deployment of open-source LLMs. 

### Evaluation
The evaluation scripts are provided in the "Collab-Overcooked/src" folder. The evaluation process consists of three sequential scripts:
- **evaluation.py**: Evaluates the environment's output, calculates the metrics for each task, and stores the results in the corresponding task folder.
- **organize_result.py**: Summarizes the metrics of each task into the `statistics_data.csv` file.
- **convert_result.py**: Computes the metrics for each complexity level and stores the results in the `converted_data.csv` file.

Example commands:
```bash
cd Collab-Overcooked/src

# (Optional) Run an episode to generate experiment logs (saved under ./data by default)
python main.py --order boiled_egg --gpt_model gpt-4o 

# Evaluate a single task: reads logs from ./data and writes results to ./eval_result by default
python evaluation.py --test_mode fix_task --model gpt-4o --order boiled_egg

# Aggregate per-task results into a CSV, then compute per-level averages
python organize_result.py
python convert_result.py
```

The final evaluation results are stored in `converted_data.csv`, with the following key fields:
- **mean_f1_agent_*:** The F1 score for agent * computed using the TES function, considering both correctly matched actions and redundant actions.
- **mean_similarity_agent_*:** The similarity between the actions generated by agent * and the RATs (Reference Action Templates).
- **mean_redundancy_agent_*:** The redundancy between the actions generated by agent * and the RATs.
- **initiate_collaboration:** The ability of the LLM-MAS (Multi-Agent System) to initiate collaboration.
- **respond_collaboration:** The ability of the LLM-MAS to respond to collaboration.

## Modify the Environment
The environment settings and logic are stored in the `Collab-Overcooked/lib/overcooked_ai` folder. Layout files are stored in the `data/layouts` folder, while the environment logic is in the `mdp` folder. 

- To add more tasks (such as additional recipes or ingredients), you only need to modify the corresponding layout files.
- If you need to add new interactive elements, ensure that you update the environment logic accordingly.


## Reference
```bibtex
@inproceedings{sun-etal-2025-collab,
  title={Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents},
  author={Sun, Haochen and Zhang, Shuwen and Niu, Lujie and Ren, Lei and Xu, Hao and Fu, Hao and Zhao, Fangkun and Yuan, Caixia and Wang, Xiaojie},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year= {2025},
  url= {https://aclanthology.org/2025.emnlp-main.249/},
  doi= {10.18653/v1/2025.emnlp-main.249}
}


@inproceedings{zhang2024proagent,
  title={Proagent: building proactive cooperative agents with large language models},
  author={Zhang, Ceyao and Yang, Kaijie and Hu, Siyi and Wang, Zihao and Li, Guanghe and Sun, Yihang and Zhang, Cheng and Zhang, Zhaowei and Liu, Anji and Zhu, Song-Chun and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17591--17599},
  year={2024}
}

@inproceedings{carroll2019utility,
 title={On the Utility of Learning About Humans for Human-AI Coordination},
 author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
 booktitle={Advances in Neural Information Processing Systems},
 pages={},
 volume={32},
 year={2019},
}
```
