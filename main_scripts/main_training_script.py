import os
os.environ["WEAVE_DISABLED"] = "true"

import json
import argparse
import math
import wandb
from utils.model_utils import load_model
from transformers import AutoTokenizer
from dotenv import load_dotenv

from utils.model_utils import get_basemodel_loadstring
from utils.dataset import load_custom_dataset
from utils.attacker import Integrated_Attacker
from utils.defender import Integrated_Defender
from utils.rollout_utils import make_attacker, make_defender
from utils.trainer import BaseTrainer, TrainerArguments, TrajectorywiseGRPOTrainer
from utils.training_utils import Unified_Logger, is_gemini_model, make_fooling_reward, make_backward_ToM_reward, make_dummy_reward, make_format_rwd_reward, make_length_reward
from utils.training_utils import make_relative_time_logger, set_global_log_fn

from utils.simple_generation_utils import get_global_token_costs, reset_global_token_costs, is_gpt_model
from utils.config import parse_args_with_config, compute_model_savepath
load_dotenv()

from transformers import get_scheduler
from torch.optim import AdamW
import torch
import inspect

def main(args):
    if not args.model_savepath:
        args.model_savepath = compute_model_savepath(args, args.results_dir, args.runname_suffix)
        print(f"Auto-computed model_savepath: {args.model_savepath}")

    if args.use_logger:
        log = make_relative_time_logger(synchronize=args.use_synchronized_logger)
        set_global_log_fn(log)
    else:
        set_global_log_fn(lambda msg: None)
    # Reset token costs at start
    reset_global_token_costs()

    import torch
    if args.cuda_memory_snapshot_path:
        if not torch.cuda.is_available():
            raise ValueError("--cuda_memory_snapshot_path requires torch.cuda.is_available() == True")
        torch.cuda.memory._record_memory_history(max_entries=200000)
        print(f"Started torch CUDA memory history recording (max_entries=100000); will dump snapshot to: {args.cuda_memory_snapshot_path}")

    wandb_project = os.environ.get("WANDB_PROJECT", "da-project-tracker")
    wandb_run_name = os.environ.get("SHELLS_LAUNCHER_LOG_NAME", None)
    wandb_group = os.environ.get("WANDB_PARENT_RUN_ID", None)
    wandb_run = wandb.init(
        project=wandb_project,
        name=wandb_run_name + "_child",
        group=wandb_group,
        config=vars(args),
    )
    wandb.config.update(vars(args), allow_val_change=True)
    wandb_step_timing_counter = {"step": 0}
    
    # || Load Defender Model ||
    defender_client = None
    if args.defender_endpoint:
        from openai import OpenAI as OpenAI_Defender
        print(f"Connecting to remote vLLM defender server at: {args.defender_endpoint}")
        defender_client = OpenAI_Defender(
            api_key="EMPTY",
            base_url=args.defender_endpoint,
        )
        model = args.engine
        tokenizer = None
        print(f"Defender model connected to endpoint: {args.defender_endpoint}")
        print(f"Defender model identifier: {args.engine}")
    elif is_gpt_model(args.engine):
        from openai import AzureOpenAI as AzureOpenAI_Defender
        defender_client = AzureOpenAI_Defender(
            api_key=os.environ["OPENAI_KEY"],
            api_version=args.azure_openai_api_version,
            azure_endpoint=args.azure_openai_endpoint,
        )
        model = args.engine
        tokenizer = None
        print(f"Defender model using Azure OpenAI endpoint: {args.azure_openai_endpoint}")
        print(f"Defender model identifier: {args.engine}")
    elif is_gemini_model(args.engine):
        from openai import OpenAI as OpenAI_Defender
        defender_client = OpenAI_Defender(
            api_key=os.environ.get("GEMINI_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        model = args.engine
        tokenizer = None
        print(f"Defender model using Gemini API: {args.engine}")
    else:
        model_info = load_model(args.engine, args.checkpoints_dir, manual_precision=False, is_trainable=True)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]

        # || Add Lora ||
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
        
        # continue_training is now the default behavior when a checkpoint is provided
        continue_training = not args.stack_new_adapter

        target_modules_types = {
            "default": ["q_proj", "v_proj"],
            "mlp_down_only": ["down_proj"],
            "combined": ["q_proj", "v_proj", "down_proj"],
            "value_only": ["v_proj"],
            "q_k_only": ["q_proj", "k_proj"],
            "q_v": ["q_proj", "v_proj"],
            "k_o": ["k_proj", "o_proj"],
            "mlps_only": ["up_proj", "down_proj", "gate_proj"],
            "attentions_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "all-linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        }
        peft_config = LoraConfig(
            lora_alpha=args.alpha,
            lora_dropout=args.lora_dropout,
            r=args.rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules_types[args.target_modules],
        )
        
        if continue_training and isinstance(model, PeftModel):
            print("Continuing training with existing LoRA adapter.")
        else:
            if continue_training and args.checkpoints_dir:
                 raise Exception("Error: continue_training is enabled but no PeftModel found.")
            else:
                print("Adding new LoRA adapter layer.")
            model = get_peft_model(model, peft_config)

        print(f"Loaded Model:\n{model}")
        model.print_trainable_parameters()
        trainable_params_tuple = model.get_nb_trainable_parameters()
        print(f"{trainable_params_tuple=}")

    # ========== JUDGE MODEL (Remote Endpoint) ===========
    if args.judge_endpoint:
        # Use remote vLLM server via OpenAI-compatible API
        from openai import OpenAI
        print(f"Connecting to remote vLLM judge server at: {args.judge_endpoint}")
        judge_llm = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require authentication by default
            base_url=args.judge_endpoint,
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(
            get_basemodel_loadstring(args.judge_model),
            trust_remote_code=True
        )
        print(f"Judge model connected to endpoint: {args.judge_endpoint}")
        print(f"Judge model identifier: {args.judge_model}")
    elif is_gemini_model(args.judge_model):
        # Use Gemini API
        from openai import OpenAI
        judge_llm = OpenAI(
            api_key=os.environ.get("GEMINI_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        judge_tokenizer = None
        print(f"Judge model using Gemini API: {args.judge_model}")
    elif is_gpt_model(args.judge_model):
        from openai import AzureOpenAI
        judge_llm = AzureOpenAI(
            api_key=os.environ["OPENAI_KEY"],
            api_version=args.azure_openai_api_version,
            azure_endpoint=args.azure_openai_endpoint,
        )
        judge_tokenizer = None
        print(f"Judge model using Azure OpenAI endpoint: {args.azure_openai_endpoint}")
        print(f"Judge model identifier: {args.judge_model}")
    elif args.judge_model:
        raise ValueError("--Judge Model Specification Error")

    # ========== ATTACKER MODEL (Remote Endpoint) ==========
    attacker_client = None
    attacker_model_name = None
    if args.attacker_endpoint:
        from openai import OpenAI as OpenAI_Attacker
        print(f"Connecting to remote vLLM attacker server at: {args.attacker_endpoint}")
        attacker_client = OpenAI_Attacker(
            api_key="EMPTY",  # vLLM doesn't require authentication by default
            base_url=args.attacker_endpoint,
        )
        attacker_model_name = args.attacker_model
        attacker_tokenizer = None  # Not needed for remote endpoint
        print(f"Attacker model connected to endpoint: {args.attacker_endpoint}")
        print(f"Attacker model identifier: {args.attacker_model}")
    elif is_gemini_model(args.attacker_model):
        # Use Gemini API
        from openai import OpenAI as OpenAI_Attacker
        attacker_client = OpenAI_Attacker(
            api_key=os.environ.get("GEMINI_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        attacker_model_name = args.attacker_model
        attacker_tokenizer = None
        print(f"Attacker model using Gemini API: {args.attacker_model}")
    elif is_gpt_model(args.attacker_model):
        from openai import AzureOpenAI as AzureOpenAI_Attacker
        attacker_client = AzureOpenAI_Attacker(
            api_key=os.environ["OPENAI_KEY"],
            api_version=args.azure_openai_api_version,
            azure_endpoint=args.azure_openai_endpoint,
        )
        attacker_model_name = args.attacker_model
        attacker_tokenizer = None
        print(f"Attacker model using Azure OpenAI endpoint: {args.azure_openai_endpoint}")
        print(f"Attacker model identifier: {args.attacker_model}")
    elif args.attacker_model:
        raise ValueError("Attacker model specification error")

    # Create reward functions based on --reward_functions argument
    reward_funcs = {}
    requested_rewards = [r.strip() for r in args.reward_functions.split(",") if r.strip()]
    print(f"Requested reward functions: {requested_rewards}")

    # Fooling reward (requires both attacker and judge)
    if not ((args.attacker_endpoint or args.attacker_model) and (args.judge_endpoint or args.judge_model)):
        raise ValueError("fooling reward requires both attacker (--attacker_endpoint or --attacker_model) and judge (--judge_endpoint or --judge_model)")
    fooling_reward_fn = make_fooling_reward(
        attacker_client=attacker_client,
        attacker_model_name=attacker_model_name,
        judge_client=judge_llm,
        judge_model_name=args.judge_model,
        attacker_type=args.attacker_type,
        fooling_only=False,
        attacker_prompts_dir=args.attacker_prompts_dir,
        judge_prompt_version=args.judge_prompt_version
    )
    if "fooling" in requested_rewards:
        reward_funcs["fooling"] = fooling_reward_fn
        print("  Added: fooling reward")

    # Fooling only reward (no 0.0 penalties)
    fooling_only_reward_fn = make_fooling_reward(
        attacker_client=attacker_client,
        attacker_model_name=attacker_model_name,
        judge_client=judge_llm,
        judge_model_name=args.judge_model,
        attacker_type=args.attacker_type,
        fooling_only=True,
        attacker_prompts_dir=args.attacker_prompts_dir,
        judge_prompt_version=args.judge_prompt_version
    )
    if "fooling_only" in requested_rewards:
        reward_funcs["fooling_only"] = fooling_only_reward_fn
        print("  Added: fooling_only reward")

    # Backward ToM reward - belief only
    backward_ToM_belief_fn = make_backward_ToM_reward(
        attacker_client=attacker_client,
        attacker_model_name=attacker_model_name,
        judge_client=judge_llm,
        judge_model_name=args.judge_model,
        attacker_type=args.attacker_type,
        type="belief",
        attacker_prompts_dir=args.attacker_prompts_dir
    )
    if "backward_ToM_belief" in requested_rewards:
        reward_funcs["backward_ToM_belief"] = backward_ToM_belief_fn
        print("  Added: backward_ToM_belief reward")

    # Parser/format reward (no external deps; uses defender_instance from rollout)
    format_rwd_reward_fn = make_format_rwd_reward()
    if "format_rwd" in requested_rewards:
        reward_funcs["format_rwd"] = format_rwd_reward_fn
        print("  Added: format_rwd reward")

    # Length reward: 1.0 if <100 tokens, linear decay to 0.0 at 200 tokens
    length_reward_fn = make_length_reward()
    if "length_reward" in requested_rewards:
        reward_funcs["length_reward"] = length_reward_fn
        print("  Added: length_reward reward")

    dummy_reward_fn = make_dummy_reward()
    if "dummy" in requested_rewards:
        reward_funcs["dummy"] = dummy_reward_fn
        print("  Added: dummy reward")

    # Trajectory-level rewards: end-of-trajectory rewards interpreted by Trajectory.subrollout
    if "prior_knowledge_ToM_single_stage" in requested_rewards:
        raise ValueError("prior_knowledge_ToM_single_stage is deprecated and no longer supported. Use prior_knowledge_ToM instead.")
    known_trajectory_level_rewards = {"fooling_successful", "prior_knowledge_ToM"}
    trajectory_level_rewards = [r for r in requested_rewards if r in known_trajectory_level_rewards]
    for r in trajectory_level_rewards:
        print(f"  Added trajectory-level reward: {r}")
    print(f"Using {len(trajectory_level_rewards)} trajectory-level reward(s)")

    if requested_rewards and not reward_funcs and not trajectory_level_rewards:
        raise ValueError(f"No valid reward functions specified. Got: {args.reward_functions}. Options: fooling, fooling_only, backward_ToM_belief, format_rwd, length_reward, dummy, fooling_successful, prior_knowledge_ToM")

    print(f"Using {len(reward_funcs)} reward function(s)")


    # Build eval_reward_funcs: additional reward functions always monitored during eval
    all_reward_fn_map = {
        "fooling": fooling_reward_fn,
        "fooling_only": fooling_only_reward_fn,
        "backward_ToM_belief": backward_ToM_belief_fn,
        "format_rwd": format_rwd_reward_fn,
        "length_reward": length_reward_fn,
        "dummy": dummy_reward_fn,
    }
    requested_eval_rewards = [r.strip() for r in args.eval_reward_functions.split(",") if r.strip()]
    eval_reward_funcs = {}
    for r in requested_eval_rewards:
        if r in reward_funcs:
            continue  # already in training rewards, no need to duplicate
        eval_reward_funcs[r] = all_reward_fn_map[r]
        print(f"  Added eval-only reward: {r}")
    print(f"Using {len(eval_reward_funcs)} eval-only reward function(s)")

    # Build eval_trajectory_level_rewards: trajectory-level rewards always computed during eval
    eval_trajectory_level_rewards = [r.strip() for r in args.eval_trajectory_level_rewards.split(",") if r.strip()]
    for r in eval_trajectory_level_rewards:
        assert r in known_trajectory_level_rewards, f"Unknown eval trajectory-level reward: {r}. Options: {known_trajectory_level_rewards}"
        if r not in trajectory_level_rewards:
            print(f"  Added eval-only trajectory-level reward: {r}")
        else:
            print(f"  Eval trajectory-level reward {r} already in training trajectory_level_rewards, skipping")
    print(f"Using {len(eval_trajectory_level_rewards)} eval-only trajectory-level reward(s)")

    # Construct Dataset
    train_dataset = load_custom_dataset(
        args.dataset, 
        args.seed, 
        True, 
        args.train_end_p, 
        args.eval_start_p
    )
    eval_dataset = load_custom_dataset(
        args.dataset,
        args.seed,
        False,
        args.train_end_p,
        args.eval_start_p,
    )
    if args.train_skip_fraction > 0.0:
        skip_count = int(len(train_dataset) * args.train_skip_fraction)
        print(f"Warning:Error: (use continue training only in extreme cases) Skipping first {skip_count}/{len(train_dataset)} train examples (train_skip_fraction={args.train_skip_fraction})")
        train_dataset = train_dataset.select(range(skip_count, len(train_dataset)))

    print(f"Loaded {len(train_dataset)} train examples and {len(eval_dataset)} eval examples")
    wandb_run.log({
        "dataset/train_size": len(train_dataset),
        "dataset/eval_size": len(eval_dataset),
        "dataset/epochs": args.epochs,
    })
    trajectory_outputs = []

    if args.testing:
        train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(10, len(eval_dataset))))
        args.n_evals = 2
        print(f"Testing mode: reduced to {len(train_dataset)} train examples, {len(eval_dataset)} eval examples, and {args.n_evals} evals")

    total_train_trajectories = args.epochs * len(train_dataset)
    if args.n_evals < 0:
        raise ValueError("--n_evals must be >= 0 (but should ideally be >= 1, use 0 for evaluation only after training is done)")
    if args.n_evals > total_train_trajectories:
        raise ValueError(f"--n_evals ({args.n_evals}) cannot exceed total train trajectories ({total_train_trajectories})")
    eval_after_trajectory_counts = [
        math.ceil(total_train_trajectories * (i + 1) / (args.n_evals + 1)) for i in range(args.n_evals)
    ]

    # Initialize the Optimizer, Scheduler, etc.
    if defender_client == None:
        model_kwarg_keys = set(inspect.signature(model.forward).parameters.keys())
    else:
        model_kwarg_keys = set()

    if defender_client is None:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        num_trajectories = args.epochs * len(train_dataset)
        num_warmup_steps = int(args.lr_warmup_ratio * num_trajectories) if args.lr_scheduler != "constant" else 0
        scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_trajectories
        )
    else:
        optimizer = None
        scheduler = None

    # Set generation kwargs
    generate_kwargs = {
        "max_tokens": args.max_completion_length,  # For API and vLLM
    }
    if args.temperature is not None:
        generate_kwargs["temperature"] = args.temperature

    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
        print(f"Set torch manual seed: {args.torch_seed}")

    # Create attacker and defender instance 
    attacker: Integrated_Attacker = make_attacker(
        args.attacker_type,
        attacker_model_name,
        None,  # tokenizer not needed for remote endpoint
        "",
        client=attacker_client,
        generate_kwargs=generate_kwargs,
        attacker_prompts_dir=args.attacker_prompts_dir,
    )
    defender: Integrated_Defender = make_defender(
        args.defender_type,
        model,
        tokenizer,
        "",
        defender_lora_path=None,
        use_imitation_learning_steering=False,
        use_reasoning=args.use_reasoning,
        generate_kwargs=generate_kwargs,
        client=defender_client,
    )

    # Initialize Logger
    logger = Unified_Logger(wandb_run, wandb_step_timing_counter)

    trainer_arguments = TrainerArguments(
        # Trajectory Objects
        attacker = attacker,
        defender = defender,
        reward_funcs = reward_funcs,
        eval_reward_funcs = eval_reward_funcs,
        wandb_step_timing_counter = wandb_step_timing_counter,
        wandb_run = wandb_run,

        # Trajectory Configs
        max_iterations = args.max_iterations,
        num_generations = args.num_generations,
        enable_lock_on_generate = args.enable_lock_on_generate,
        max_format_retries = args.max_format_retries,

        # Training Objects
        model = model,
        tokenizer = tokenizer,
        scheduler = scheduler,
        optimizer = optimizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,

        # Training Configs
        epochs = args.epochs,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        max_grad_norm = args.max_grad_norm,
        loss_type = args.loss_type,
        max_completion_length = args.max_completion_length,
        model_kwarg_keys = model_kwarg_keys,
        eval_after_trajectory_counts = eval_after_trajectory_counts,
        total_train_trajectories = total_train_trajectories,
        n_evals = args.n_evals,
        raw_reward_function_names = requested_rewards,

        # Debug and Saving Configs
        cuda_memory_snapshot_path = args.cuda_memory_snapshot_path,
        model_savepath = args.model_savepath,
        model_name = args.engine,

        # Logger
        logger = logger,

        # Eval Configs
        eval_num_workers = args.eval_num_workers,
        eval_batch_size = args.eval_batch_size,
        debug_judge_prompts = args.debug_judge_prompts,
        attacker_type = args.attacker_type,
        defender_type = args.defender_type,
        dataset = args.dataset,
        attacker_model = args.attacker_model,
        defender_model = args.engine,

        # Prompt locking
        train_prompt_id = args.train_prompt_id,

        # Judge prompt version
        judge_prompt_version = args.judge_prompt_version,

        # Reward combination
        convex_joint = args.convex_joint,

        # Trajectory-level rewards
        trajectory_level_rewards = trajectory_level_rewards,
        eval_trajectory_level_rewards = eval_trajectory_level_rewards,
        judge_client = judge_llm,
        judge_model_name = args.judge_model,
    )

    # Pick Trainer.
    if args.convex_joint and args.training_strategy != "TrajectorywiseGRPO":
        raise ValueError(f"--convex_joint is only supported for TrajectorywiseGRPO, got: {args.training_strategy}")
    if args.training_strategy == "Stepwise":
        trainer = BaseTrainer(trainer_arguments)
    elif args.training_strategy == "TrajectorywiseGRPO":
        trainer = TrajectorywiseGRPOTrainer(trainer_arguments)
    else:
        raise Exception(f"Unknown training_strategy: {args.training_strategy}")

    if args.do_run_first_eval:
        trainer.run_eval(next_eval_idx=0, save_results=args.eval_only)
    if args.eval_only:
        trainer.print_total_logs()
        # Print final token usage before early return
        print(f"\n{'='*50}")
        print(f"Final Token Usage:")
        token_costs = get_global_token_costs()
        for key, value in token_costs.items():
            print(f"  {key}: {value}")
        print(f"{'='*50}\n")
        return
    trainer.run_train()
    trainer.run_eval(next_eval_idx=args.n_evals + 1, save_results=True)
    trainer.print_total_logs()
    print("Input args for above run:")
    print(f"[=]  attacker_prompts_dir: {args.attacker_prompts_dir}")
    print(json.dumps(vars(args), indent=2))

    if args.save_trajectory_outputs:
        save_dir = os.path.dirname(args.save_trajectory_outputs)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(args.save_trajectory_outputs, "w") as f:
            for o in trajectory_outputs:
                f.write(json.dumps(o) + "\n")
        print(f"Wrote trajectory_outputs to {args.save_trajectory_outputs}")

    # Save the final model
    if defender_client is None:
        print("\n" + "="*50)
        print("Saving final model")
        print("="*50)
        os.makedirs(args.model_savepath, exist_ok=True)
        if hasattr(model, "peft_config"):
            print("  Model has PEFT config")
            print(f"  Adapters: {list(model.peft_config.keys())}")
            print(f"  Active adapter: {getattr(model, 'active_adapter', None)}")
        print(f"Saving model to {args.model_savepath}")
        model.save_pretrained(args.model_savepath)
        print("Training completed!")
    
    # Print final token usage
    print(f"\n{'='*50}")
    print(f"Final Token Usage:")
    token_costs = get_global_token_costs()
    for key, value in token_costs.items():
        print(f"  {key}: {value}")
    print(f"{'='*50}\n")

    if args.cuda_memory_snapshot_path:
        torch.cuda.memory._dump_snapshot(args.cuda_memory_snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Wrote torch CUDA memory snapshot to {args.cuda_memory_snapshot_path}")

    wandb_run.finish()

            
            

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file (CLI args override config values)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to load")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset loading")
    parser.add_argument("--torch_seed", type=int, default=None, help="If set, call torch.manual_seed() with this value before attacker/defender construction")
    parser.add_argument("--train_end_p", type=float, default=0.75, help="Training split end percentage")
    parser.add_argument("--eval_start_p", type=float, default=0.75, help="Evaluation split start percentage")
    parser.add_argument("--train_skip_fraction", type=float, default=0.0, help="Skip this fraction of the training set (e.g., 0.5 to use only the second half)")
    parser.add_argument("--engine", type=str, default="Qwen3-8B", help="Model name/identifier")
    parser.add_argument("--checkpoints_dir", type=str, default="", help="Directory containing model checkpoints")
    parser.add_argument("--model_savepath", type=str, default="", help="Path to save trained model; auto-computed from other args if not set")
    parser.add_argument("--results_dir", type=str, default=os.environ["RESULTS_DIR"], help="Base results directory used when auto-computing model_savepath")
    parser.add_argument("--runname_suffix", type=str, default="run", help="Prefix string for the auto-computed run name")
    
    # LoRA hyperparameters
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout rate")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--target_modules", type=str, default="default", 
                        choices=["default", "mlp_down_only", "combined", "value_only", "q_k_only", 
                                "q_v", "k_o", "mlps_only", "attentions_only", "all-linear"],
                        help="Which modules to apply LoRA to")
    
    # Training Configs
    parser.add_argument("--training_strategy", type=str, help="What type of training strategy to use, will initiate a different trainer depending on the choice.", default="Stepwise")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=["constant", "linear", "cosine"], help="Learning rate scheduler type")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.0, help="Fraction of total steps used for linear warmup (applies to linear and cosine schedulers)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps, each step is defined as one loss computation (one loss computation may compute loss on num_generations generations)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping before optimizer step")

    # Judge model arguments (remote endpoint only)
    parser.add_argument("--judge_model", type=str, default="Qwen3-32B",
                        help="Model for judging information revelation")
    parser.add_argument("--judge_endpoint", type=str, default="",
                        help="OpenAI-compatible vLLM server endpoint (e.g., http://localhost:8000/v1)")
    
    # Attacker model arguments (remote endpoint only)
    parser.add_argument("--attacker_model", type=str, default="gemini-3-flash-preview",
                        help="Model name/identifier for attacker (e.g., Qwen3-8B, gemini-2.5-flash)")
    parser.add_argument("--attacker_endpoint", type=str, default="",
                        help="OpenAI-compatible vLLM server endpoint for attacker (e.g., http://localhost:8001/v1)")
    parser.add_argument("--defender_endpoint", type=str, default="",
                        help="OpenAI-compatible vLLM server endpoint for defender (e.g., http://localhost:8002/v1)")
    parser.add_argument("--azure_openai_endpoint", type=str, default=os.environ["AZURE_OPENAI_ENDPOINT"],
                        help="Azure OpenAI endpoint used when model starts with gpt-*")
    parser.add_argument("--azure_openai_api_version", type=str, default="2024-12-01-preview",
                        help="Azure OpenAI API version used when model starts with gpt-*")
    parser.add_argument("--attacker_type", type=str, default="verifying_attacker_swapable_prompt",
                        help="Type of attacker to use")
    parser.add_argument("--attacker_prompts_dir", type=str, default="utils/prompts",
                        help="Directory of prompt text files for verifying_attacker_swapable_prompt (files named with numeric id)")
    parser.add_argument("--defender_type", type=str, default="integrated_defender_with_reflection_v2",
                        help="Type of defender to use")
    parser.add_argument("--reward_functions", type=str, default="fooling_successful",
                        help="Comma-separated list of reward functions to use. Options: fooling, fooling_only, backward_ToM_belief, format_rwd, length_reward, dummy, fooling_successful, prior_knowledge_ToM")
    parser.add_argument("--eval_reward_functions", type=str, default="backward_ToM_belief",
                        help="Comma-separated list of additional reward functions always computed during eval (but not training). Same options as --reward_functions.")
    parser.add_argument("--eval_trajectory_level_rewards", type=str, default="prior_knowledge_ToM",
                        help="Comma-separated list of trajectory-level rewards always computed during eval. Options: fooling_successful, prior_knowledge_ToM")
    parser.add_argument("--use_reasoning", action="store_true",
                        help="Enable reasoning for defender")
    parser.add_argument("--enable_lock_on_generate", action="store_true",
                        help="Use a threading.Lock around HuggingFace offline generate (for parallel rollout trainers). Applies to the Trajectory class used by BaseTrainer and TrajectorywiseGRPOTrainer.")
    parser.add_argument("--max_format_retries", type=int, default=0,
                        help="Max retries per defender generation when format_rwd check fails (0 = no retries)")

    # Training hyperparameters
    parser.add_argument("--max_iterations", type=int, default=15, help="Maximum number of iterations per trajectory for multi-turn environments")
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Number of completions to sample per prompt for GRPO")
    parser.add_argument("--max_completion_length", type=int, default=20000,
                        help="Maximum length of generated completions")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature; if not set, model/API default is used")
    parser.add_argument(
        "--loss_type",
        type=str,
        required=True,
        help="Loss type to use for defender training. Supported: dr_grpo, grpo, dr_grpo_with_only_seqnorm, dr_grpo_with_tokennorm.",
    )
    parser.add_argument(
        "--use_logger",
        action="store_true",
        help="Enable relative-time logger (prints profiling timestamps).",
    )
    parser.add_argument(
        "--use_synchronized_logger",
        action="store_true",
        help="If set, logger calls torch.cuda.synchronize() before printing (more accurate GPU timings, slower).",
    )

    # Testing
    parser.add_argument("--eval_num_workers", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=6,
                        help="Number of eval jobs (datapoint x prompt combinations) to run in parallel via ThreadPoolExecutor")
    parser.add_argument("--testing", action="store_true",
                        help="Truncate dataset to 10 samples for quick testing")
    parser.add_argument("--n_evals", type=int, default=1,
                        help="Number of eval passes over eval dataset during training (includes final eval after all trajectories)")
    parser.add_argument("--stack_new_adapter", action="store_true",
                        help="Stack a new adapter instead of continuing training the existing one (default: continue training if checkpoint exists)")
    parser.add_argument("--save_trajectory_outputs", type=str, default="",
                        help="Write JSONL of train/eval trajectory outputs to this path")
    parser.add_argument("--cuda_memory_snapshot_path", type=str, default="",
                        help="If set, record CUDA memory history and dump a snapshot to this path for PyTorch memory visualization")
    parser.add_argument("--debug_judge_prompts", action="store_true",
                        help="Print judge prompt/response for the first example in each batch")
    parser.add_argument("--do_run_first_eval", type=int, default=0, choices=[0,1],
                        help="Run eval prior to training, 0 for no eval prior to train")
    parser.add_argument("--eval_only", action="store_true",
                        help="Run only the first eval (controlled by --do_run_first_eval) and skip training and the final eval")

    # Prompt locking
    parser.add_argument("--train_prompt_id", type=str, default=None, help="If set, lock training to this prompt_id on every fresh_start")

    # Judge prompt version
    parser.add_argument("--judge_prompt_version", type=str, default="v3", help="Judge prompt version for evaluate_attack_success_batch ('v1' or 'v2')")

    # Reward combination
    parser.add_argument("--convex_joint", action="store_true",
                        help="Take mean over reward types instead of sum when combining multiple reward functions")

    args = parse_args_with_config(parser)
    print("Parsed arguments:")
    print(json.dumps(vars(args), indent=2))
    main(args)