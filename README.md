# AI Double Agent Defenders

Official repository for **[Playing Along: Learning a Double-Agent Defender for Belief Steering via Theory of Mind](https://arxiv.org/abs/2604.11666)**

**[Hanqi Xiao](https://hanqixiao.github.io/)\*, [Vaidehi Patil](https://vaidehi99.github.io/#home)\*, [Zaid Khan](https://zaidkhan.me/), [Hyunji Lee](https://amy-hyunji.github.io/), [Elias Stengel-Eskin](https://esteng.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)**

![Overview](images/DA_Fig1_v14.png)

## Abstract

As large language models (LLMs) become the engine behind conversational systems, their ability to reason about the intentions and states of their dialogue partners (i.e., form and use a theory-of-mind, or ToM) becomes increasingly critical for safe interaction with potentially adversarial partners. We propose a novel privacy-themed ToM challenge, **ToM for Steering Beliefs (ToM-SB)**, in which a defender must act as a Double Agent to steer the beliefs of an attacker with partial prior knowledge within a shared universe. To succeed on ToM-SB, the defender must engage with and form a ToM of the attacker, with a goal of fooling the attacker into believing they have succeeded in extracting sensitive information. We find that strong frontier models like Gemini3-Pro and GPT-5.4 struggle on ToM-SB, often failing to fool attackers in hard scenarios with partial attacker prior knowledge, even when prompted to reason about the attacker's beliefs (ToM prompting). To close this gap, we train models on ToM-SB to act as AI Double Agents using reinforcement learning, testing both fooling and ToM rewards. Notably, we find a bidirectionally emergent relationship between ToM and attacker-fooling: rewarding fooling success alone improves ToM, and rewarding ToM alone improves fooling. Across four attackers with different strengths, six defender methods, and both in-distribution and out-of-distribution (OOD) evaluation, we find that gains in ToM and attacker-fooling are well-correlated, highlighting belief modeling as a key driver of success on ToM-SB. AI Double Agents that combine both ToM and fooling rewards yield the strongest fooling and ToM performance, outperforming Gemini3-Pro and GPT-5.4 with ToM prompting on hard scenarios. We also show that ToM-SB and AI Double Agents can be extended to stronger attackers, demonstrating generalization to OOD settings and the upgradability of our task.

**This repository is under construction. Code and documentation are being cleaned for public release.**

## Installation

### Requirements

- Python 3.12.2
- PyTorch compiled with CUDA

### Setup

```bash
python -m venv AIDoubleAgentDefenders_venv
source AIDoubleAgentDefenders_venv/bin/activate
pip install -r requirements.txt
```

### Shell Script Permissions
In case any scripts do not run, ensure to change their execution permissions to be permissive.
```bash
chmod +x config_launchers/*.sh datasets_directory/data_generation_scripts/*.sh
```

### Environment Variables

Create a `.env` file in the repository root:

```bash
# Required
GEMINI_KEY=<your Gemini API key>
OPENAI_KEY=<your OpenAI API key>
VENV_PATH=<path to your virtualenv /bin/activate script>
REPO_DIR=<absolute path to this repository>
RESULTS_DIR=<base directory for saving results>
LOG_DIRECTORY=<directory for log files>
AZURE_OPENAI_ENDPOINT=<your Azure OpenAI endpoint, required for gpt-* models>
WANDB_API_KEY=<your Weights & Biases API key>

# Optional — only needed if using Vertex AI auth for dataset generation (--use_vertex_ai)
GOOGLE_APPLICATION_CREDENTIALS=<path to your Google service account JSON>
GCP_PROJECT_ID=<your GCP project ID>
GCP_LOCATION=<your GCP location, defaults to us-east1>
```

---

## Training

All training jobs are launched via `shells_launcher.py`, which wraps a shell script in a subprocess and:
- Initializes a [Weights & Biases](https://wandb.ai/) run to track the job, you need to log in to a wandb account in your terminal or specify WANDB_API_KEY.
- Streams stdout to both the console and a log file (written to `LOG_DIRECTORY`)
- Logs errors and the script's exit code to W&B

During training, there is verbose output. Search for `Starting eval` in the log to jump between evaluation blocks that print performance summaries.

You can also directly run the .sh scripts without shells_launcher (you may have to handle saving the output logs yourself)

**ADA Fooling + ToM** (combined reward):
```bash
python shells_launcher.py -s config_launchers/train_against_main_attacker.sh -l Date_logname.txt -a "0,1,2,3" config_launchers/configs/train_traj_fool_and_PToM.yaml
```

**ADA Fooling** (fooling reward only):
```bash
python shells_launcher.py -s config_launchers/train_against_main_attacker.sh -l Date_logname.txt -a "0,1,2,3" config_launchers/configs/train_traj_fool.yaml
```

**ADA ToM** (ToM reward only):
```bash
python shells_launcher.py -s config_launchers/train_against_main_attacker.sh -l Date_logname.txt -a "0,1,2,3" config_launchers/configs/train_traj_PToM.yaml
```

### Evaluation

To evaluate a defender across multiple attacker prompts, use `launch_main_eval_ov4eval_multiattackeranalysis.sh`. It runs `main_training_script.py` in eval-only mode over 4 attacker prompt sets for each config yaml provided.

```bash
python shells_launcher.py -s config_launchers/launch_main_eval_ov4eval_multiattackeranalysis.sh \
    -l eval_logname.txt -a "0,1,2,3" 42 \
    config_launchers/configs/eval_gemini.yaml
```

Arguments after `-a` are: `CUDA_VISIBLE_DEVICES`, `torch_seed`, followed by one or more config yamls. Set `eval_only: true` in the yaml to skip training.

To evaluate a trained LoRA checkpoint, point `--checkpoints_dir` at the directory containing the saved adapter and set `--engine` to the adapter subdirectory name:

```yaml
ModelConfig:
  engine: "my_lora_model"
  checkpoints_dir: "/path/to/saved_checkpoints"
  eval_only: true
  do_run_first_eval: true
```

During training, checkpoints are saved to `model_savepath` (auto-computed from `RESULTS_DIR` and run name, or set explicitly via `--model_savepath`). It should be printed near the end of a successful training run log. 

---

## Dataset Generation

Two-stage pipeline in `datasets_directory/data_generation_scripts/`. By default authenticates with `GEMINI_KEY`; pass `--use_vertex_ai` to use Vertex AI instead (requires `GCP_PROJECT_ID` and optionally `GCP_LOCATION` in `.env`).

**1. Generate raw scenarios** (`dataset_generation.py`)

| Argument | Default | Description |
|---|---|---|
| `--mode {theme,balance,none}` | `theme` | `theme` — cycles through 15 built-in scenario themes (home address, corporate structure, medical provider, …). `balance` — cycles attacker knowledge level through 0/1/2 (deprecated; `transform_dataset` handles this now; this is how the original paper dataset was generated). `none` — uses the base prompt only. |
| `--num_attempts N` | `500` | Number of generation attempts. Not all will pass verification, so the number of output files will be less than this. |
| `--output_dir DIR` | `datasets_directory/raw_datasets` | Directory to write raw scenario JSONs to. |
| `--use_vertex_ai` | off | Use Vertex AI auth instead of `GEMINI_KEY`. |

```bash
python datasets_directory/data_generation_scripts/dataset_generation.py \
    --mode theme --num_attempts 500 --output_dir datasets_directory/raw_datasets
```

**2. Transform into training data** (`transform_dataset.py`)

Converts raw scenario JSONs into attacker/defender prompt pairs. For each scenario, produces 3 examples with attacker knowing 0, 1, or 2 layers of the hierarchy. Output is a shuffled JSON list.

| Argument | Default | Description |
|---|---|---|
| `--input_dir DIR` | (required) | Directory containing raw scenario JSONs from step 1. |
| `--output_file FILE` | (required) | Path for the output JSON file. |

```bash
python datasets_directory/data_generation_scripts/transform_dataset.py \
    --input_dir datasets_directory/raw_datasets \
    --output_file datasets_directory/transformed_dataset.json
```

See `datasets_directory/data_generation_scripts/test_pipeline.sh` for an example of how the two scripts can be composed.

**Example outputs** are included in the repo for reference:
- `datasets_directory/example_raw_dataset_generations/` — sample raw scenario JSONs from step 1.
- `datasets_directory/final_datasets/example_transformed_dataset.json` — sample transformed training data from step 2.

---

## Repository Structure

| Path | Description |
|------|-------------|
| `main_scripts/main_training_script.py` | Main training entry point |
| `utils/trainer.py` | Trainer implementations (GRPO, trajectory-wise) |
| `utils/training_utils.py` | Reward functions and logging utilities |
| `utils/defender.py` | Defender agent implementations |
| `utils/attacker.py` | Attacker agent implementations |
| `utils/model_utils.py` | Model loading and configuration |
| `utils/simple_generation_utils.py` | LLM generation helpers |
| `utils/dataset.py` | Dataset loading |
| `utils/prompts/` | Attacker prompt templates |
| `config_launchers/` | Shell scripts and YAML configs for training and evaluation |
| `datasets_directory/` | Dataset files and generation scripts |
