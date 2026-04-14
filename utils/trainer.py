from collections import defaultdict
from dataclasses import dataclass, fields
import gc
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from torch.nn import functional as F
from wandb import Run
from main_scripts.evaluate_outputs import evaluate_trajectories, print_evaluation_results, evaluate_multi_prompt_trajectories, print_multi_prompt_results
from utils.attacker import Integrated_Attacker
from utils.defender import Integrated_Defender
from utils.rollout_utils import evaluate_attack_success_batch
from utils.training_utils import Trajectory, Unified_Logger, log
from typing import Any, Literal
from contextlib import nullcontext

TokenTensor = torch.Tensor
BooleanTensor = torch.Tensor

def print_trajectory_like_setting_simplified_iterated(*, trajectory_output: dict, idx: int, label: str):
    # Reference: main_scripts/setting_simplified_iterated.py:249-257
    print(f"=-=-=-=-=-={label} Example {idx}=-=-=-=-=-=")  # Separator between conversations
    print("Attacker Signals: " + "->".join(trajectory_output["attacker_reflection_signals"]))
    print(f"Attacker Goal: {trajectory_output['attacker_target_information']}")
    print(f"Ground Truth: {trajectory_output['defender_private_information']}")
    for message in trajectory_output["conversation_histories"]:
        print(f"{message['role']}: {message['content']}")
    print(flush=True)

@dataclass
class TrainerArguments():
    """Central reference of trainer arguments, shared by BaseTrainer and TrajectorywiseGRPOTrainer."""
    # Trajectory Objects
    attacker: Integrated_Attacker = None # Since these are constructed as an object prior to passing to trainer, we can imbune them with history. 
    defender: Integrated_Defender = None
    reward_funcs: dict = None
    eval_reward_funcs: dict = None  # extra reward functions always computed during eval (merged with reward_funcs)
    wandb_step_timing_counter: dict = None
    wandb_run: Run = None

    # Trajectory Configs
    max_iterations: int = None
    num_generations: int = None
    enable_lock_on_generate: bool = False

    # Training Objects
    model: Any = None
    tokenizer: Any = None
    scheduler: Any = None
    optimizer: Any = None
    train_dataset: Any = None
    eval_dataset: Any = None

    # Training Configs
    epochs: int = None
    gradient_accumulation_steps: int = None
    max_grad_norm: float = 10.0
    loss_type: str = None
    max_completion_length: int = None
    model_kwarg_keys: set = None
    eval_after_trajectory_counts: int = None # How many trajectories before we trigger an evaluation event
    total_train_trajectories: int = None
    n_evals: int = None
    raw_reward_function_names: list[str] = None

    # Logging object
    logger: Unified_Logger = None

    # Debug and Saving Configs
    cuda_memory_snapshot_path: str = None
    model_savepath: str = None
    model_name: str = None

    # Eval Configs
    eval_num_workers: int = 1
    eval_batch_size: int = 1
    debug_judge_prompts: bool = None
    attacker_type: str = None
    defender_type: str = None
    dataset: str = None
    attacker_model: str = None
    defender_model: str = None

    # Format retry
    max_format_retries: int = 0

    # Prompt locking
    train_prompt_id: str = None  # If set, lock training to this prompt_id on every fresh_start; if None, attacker picks randomly as usual

    # Judge prompt version
    judge_prompt_version: str = "v1"

    # Reward combination
    convex_joint: bool = False  # If True, take mean over reward types instead of sum (only supported for TrajectorywiseGRPO)

    # Trajectory-level rewards (list of strings interpreted by Trajectory.subrollout at end of trajectory)
    trajectory_level_rewards: list = None
    eval_trajectory_level_rewards: list = None
    judge_client: Any = None
    judge_model_name: str = None


@dataclass
class LossConfigs:
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    token_level_normalize_type: Literal["token", "max_len", "sequence_only"] = "token"
    beta: float = None # KL regularization coef
    strip_pad_tokens: bool = True # Strip pad Tokens From the Loss so we don't tune to generate pad tokens (arbitrarily increasing loss for no reason [it's possible that this is only an issue if we use logprobs loss])


def compute_outcomes_by_null_count(information_dict_list):
    """Compute fooling/extraction/detected/baited rates bucketed by null count in attacker's Initial Private Beliefs.
    Uses the same outcome definitions as evaluate_outputs.evaluate_trajectories."""
    from main_scripts.evaluate_outputs import normalize_signals, first_terminal_type

    buckets = defaultdict(lambda: {"total": 0, "fooled": 0, "extracted": 0, "detected": 0, "baited": 0})
    for info_dict in information_dict_list:
        ati = info_dict["attacker_target_information"]
        m = re.search(r'Your Initial Private Beliefs.*?(\{.*?\})', ati, re.DOTALL)
        beliefs = json.loads(m.group(1))
        null_count = sum(1 for v in beliefs.values() if v is None)

        sigs = normalize_signals(list(info_dict.get("attacker_reflection_signals", [])))
        first = first_terminal_type(sigs)
        successful = info_dict.get("attacker_successful")

        b = buckets[null_count]
        b["total"] += 1
        if first == "attack successful" and successful is False:
            b["fooled"] += 1
        elif successful is True:
            b["extracted"] += 1
        elif first == "defender detected":
            b["detected"] += 1
        else:
            b["baited"] += 1

    lines = []
    for k in sorted(buckets):
        b = buckets[k]
        t = b["total"]
        pct = lambda v: f"{100*v/t:.1f}%" if t else "N/A"
        lines.append(
            f"  nulls={k} (n={t}): fooled={b['fooled']}({pct(b['fooled'])})  "
            f"extracted={b['extracted']}({pct(b['extracted'])})  "
            f"detected={b['detected']}({pct(b['detected'])})  "
            f"baited={b['baited']}({pct(b['baited'])})"
        )
    return "Outcomes by Attacker Prior Knowledge Null Count:\n" + "\n".join(lines)


class BaseTrainer():
    """A steplevel GRPO trainer that is the base trainer for TrajectorywiseGRPOTrainer."""
    def __init__(self, args: TrainerArguments):
        self.args = args
        self.loss_type = args.loss_type
        self.gradients_accumulated_count = 0
        self.logger = args.logger

        # Accumulate a log of args and eval results for final printout
        args_lines = ["===== TrainerArguments ====="]
        for f in fields(args):
            args_lines.append(f"  {f.name}: {getattr(args, f.name)}")
        self.total_logs = "\n".join(args_lines) + "\n"

    def print_total_logs(self):
        print("\n===== total_logs =====")
        print(self.total_logs)


    @torch.no_grad()
    def run_eval(self, trajectory_number=0, next_eval_idx=0, save_results: bool = False):
        args = self.args
        attacker, defender = args.attacker, args.defender

        has_multiple_prompts = hasattr(attacker, 'get_prompt_ids')
        prompt_ids = attacker.get_prompt_ids() if has_multiple_prompts else [None]

        # Build eval jobs dictionary: (sample_idx, prompt_id) -> job_info
        eval_jobs = {}
        for eval_sample_idx, eval_sample in enumerate(args.eval_dataset):
            for prompt_id in prompt_ids:
                eval_jobs[(eval_sample_idx, prompt_id)] = {
                    "eval_sample_idx": eval_sample_idx,
                    "prompt_id": prompt_id,
                    "attacker_target_information": eval_sample["attacker_target_information"],
                    "defender_private_information": eval_sample["defender_private_information"],
                }

        print(f"=-=-=-=-=-=Starting eval {next_eval_idx + 1}/{args.n_evals} after {trajectory_number}/{args.total_train_trajectories} train trajectories=-=-=-=-=-=")
        print(f"  {len(args.eval_dataset)} datapoints x {len(prompt_ids)} prompts = {len(eval_jobs)} eval jobs, eval_batch_size={args.eval_batch_size}")
        if hasattr(args.model, 'eval'):
            args.model.eval()

        def run_single_eval_job(job_key, job_info, job_flat_idx):
            eval_sample_idx = job_info["eval_sample_idx"]
            prompt_id = job_info["prompt_id"]

            local_attacker = attacker.copy()
            local_defender = defender.copy()

            local_attacker.update_attacker_state(attacker_target_information=job_info["attacker_target_information"], fresh_start=True, prompt_id=prompt_id)
            local_defender.update_defender_state(defender_private_information=job_info["defender_private_information"], fresh_start=True)

            eval_trajectory_id = args.total_train_trajectories + (next_eval_idx * len(eval_jobs)) + job_flat_idx
            eval_reward_funcs = {**(args.reward_funcs or {}), **(args.eval_reward_funcs or {})}
            eval_traj_level_rewards = list(args.trajectory_level_rewards or [])
            for r in (args.eval_trajectory_level_rewards or []):
                if r not in eval_traj_level_rewards:
                    eval_traj_level_rewards.append(r)
            eval_trajectory = Trajectory(
                attacker=local_attacker,
                defender=local_defender,
                defender_optimizer=args.optimizer,
                max_turns=args.max_iterations,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_generations=args.num_generations,
                reward_functions=eval_reward_funcs,
                trajectory_id=eval_trajectory_id,
                wandb_run=args.wandb_run,
                wandb_step_timing_counter=args.wandb_step_timing_counter,
                loss_type=args.loss_type,
                max_completion_length=args.max_completion_length,
                model_kwarg_keys=args.model_kwarg_keys,
                logger=args.logger,
                judge_prompt_version=args.judge_prompt_version,
                trajectory_level_rewards=eval_traj_level_rewards,
                judge_client=args.judge_client,
                judge_model_name=args.judge_model_name,
            )

            is_first_job = (eval_sample_idx == 0 and (prompt_id is None or prompt_id == prompt_ids[0]))
            eval_output = eval_trajectory.subrollout(eval_mode=True, debug_prompts=is_first_job)
            eval_output["eval_idx"] = next_eval_idx
            eval_output["prompt_id"] = prompt_id

            del eval_trajectory
            gc.collect()
            torch.cuda.empty_cache()

            return job_key, eval_output

        # Execute eval jobs in batches of eval_batch_size
        eval_results = {}
        job_items = list(eval_jobs.items())
        eval_batch_size = args.eval_batch_size

        for batch_start in range(0, len(job_items), eval_batch_size):
            batch = job_items[batch_start:batch_start + eval_batch_size]
            batch_with_idx = [(key, info, batch_start + i) for i, (key, info) in enumerate(batch)]

            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [executor.submit(run_single_eval_job, key, info, flat_idx) for key, info, flat_idx in batch_with_idx]
                for future in as_completed(futures):
                    result_key, result = future.result()
                    eval_results[result_key] = result

            if args.cuda_memory_snapshot_path:
                torch.cuda.memory._dump_snapshot(args.cuda_memory_snapshot_path)

        # Save model after each eval
        eval_checkpoint_dir = f"{args.model_savepath}/{args.model_name}_checkpoint-eval-{next_eval_idx}_lora_model"
        os.makedirs(eval_checkpoint_dir, exist_ok=True)
        print(f"Saving model checkpoint after eval {next_eval_idx} to {eval_checkpoint_dir}")
        if hasattr(args.model, "peft_config"):
            print("  Model has PEFT config, saving adapter")
            print(f"  Adapters: {list(args.model.peft_config.keys())}")
            print(f"  Active adapter: {getattr(args.model, 'active_adapter', None)}")
        if hasattr(args.model, 'save_pretrained'):
            args.model.save_pretrained(eval_checkpoint_dir)
            args.tokenizer.save_pretrained(eval_checkpoint_dir)
        print(f"Model checkpoint saved to {eval_checkpoint_dir}")

        try:
            # Multi-sample results first (before signal list mutation from appending extraction tokens)
            if has_multiple_prompts:
                multi_results = evaluate_multi_prompt_trajectories(eval_results)

            # Flatten results to information_dict_list for evaluate_trajectories
            information_dict_list = []
            for (eval_sample_idx, prompt_id) in sorted(eval_results.keys()):
                eval_output = eval_results[(eval_sample_idx, prompt_id)]
                information_dict = {**eval_output, "prompt_id": prompt_id, "eval_sample_idx": eval_sample_idx}
                information_dict["attacker_reflection_signals"] = list(information_dict["attacker_reflection_signals"]) + ["extractionSuccessful" if information_dict["attacker_successful"] else "extractionFailed"]
                information_dict_list.append(information_dict)

            # Save eval results to JSONL
            if save_results:
                eval_results_savepath = f"{args.model_savepath}/{args.model_name}_eval-{next_eval_idx}_results.jsonl"
                with open(eval_results_savepath, "w") as f:
                    for info_dict in information_dict_list:
                        save_dict = {
                            "eval_sample_idx": info_dict["eval_sample_idx"],
                            "prompt_id": info_dict["prompt_id"],
                            "attacker_reflection_signals": info_dict["attacker_reflection_signals"],
                            "attacker_successful": info_dict["attacker_successful"],
                            "defender_private_information": info_dict["defender_private_information"],
                            "attacker_output": info_dict["attacker_output"],
                            "attacker_target_information": info_dict["attacker_target_information"],
                            "conversation_histories": info_dict["conversation_histories"],
                            "attacker_conversation_history": info_dict["attacker_conversation_history"],
                        }
                        f.write(json.dumps(save_dict) + "\n")
                print(f"Eval results saved to {eval_results_savepath}")

            # Print Detailed Debug Information
            for idx, info_dict in enumerate(information_dict_list):
                prompt_filename = getattr(attacker, 'prompt_filenames', {}).get(info_dict.get('prompt_id'), '')
                filename_str = f", file={prompt_filename}" if prompt_filename else ""
                print(f"=-=-=-=-=-=Testing Example {idx} (prompt_id={info_dict.get('prompt_id')}{filename_str})=-=-=-=-=-=")
                print(f"[=]Defender Name: {args.defender_model}")
                print("[=]Attacker + Judge Signals: " + "->".join(info_dict["attacker_reflection_signals"]))
                print(f"[=]Attacker Context: {info_dict['attacker_target_information']}")
                print(f"[=]Defender Context: {info_dict['defender_private_information']}")
                print("="*10 + "conversation:")
                for message in info_dict["conversation_histories"]:
                    print(f"{message['role']}: {message['content']}\n")
                print("[=]Attacker + Judge Signals (repeated): " + "->".join(info_dict["attacker_reflection_signals"]))

            # Print standard evaluation statistics
            signals_printout = '\n'.join([f"id{idx}: " + "->".join(i["attacker_reflection_signals"]) for idx, i in enumerate(information_dict_list)])
            stat_line_1 = f"Results for {eval_checkpoint_dir}"
            stat_line_2 = f"Eval #{next_eval_idx} Results for | attacker {args.attacker_type} | defender {args.defender_type} | on {args.dataset} | {args.attacker_model=} | {args.defender_model=}"
            print(stat_line_1)
            print(stat_line_2)
            results = evaluate_trajectories(information_dict_list, print_trajectories=False)
            results_str = print_evaluation_results(results)
            self.total_logs += f"\n{stat_line_1}\n{stat_line_2}\n{results_str}"

            # Per-reward statistics from core_trajectory_rewards_per_turn
            reward_trajectory_means = defaultdict(list)
            reward_all_values = defaultdict(list)
            for info_dict in information_dict_list:
                traj_totals = defaultdict(float)
                traj_counts = defaultdict(int)
                for turn_reward_dict in info_dict["core_trajectory_rewards_per_turn"]:
                    if not turn_reward_dict:
                        continue
                    for reward_name, reward_value in turn_reward_dict.items():
                        traj_totals[reward_name] += reward_value
                        traj_counts[reward_name] += 1
                        reward_all_values[reward_name].append(reward_value)
                for reward_name in traj_totals:
                    reward_trajectory_means[reward_name].append(traj_totals[reward_name] / traj_counts[reward_name])

            all_reward_names = sorted(reward_trajectory_means.keys())
            reward_stats_lines = []
            for reward_name in all_reward_names:
                traj_mean = sum(reward_trajectory_means[reward_name]) / len(reward_trajectory_means[reward_name])
                instance_mean = sum(reward_all_values[reward_name]) / len(reward_all_values[reward_name])
                reward_stats_lines.append(f"  {reward_name}: traj_mean={traj_mean:.4f} (instance_mean={instance_mean:.4f})")
            reward_stats_str = "Core Trajectory Reward Statistics:\n" + "\n".join(reward_stats_lines)
            print(reward_stats_str)
            self.total_logs += f"\n{reward_stats_str}"

            if has_multiple_prompts:
                multi_str = print_multi_prompt_results(multi_results, prompt_filenames=getattr(attacker, 'prompt_filenames', None))
                self.total_logs += f"\n{multi_str}"

            try:
                null_str = compute_outcomes_by_null_count(information_dict_list)
                print(null_str)
                self.total_logs += f"\n{null_str}"
            except:
                print("Outcomes by null count computation failed")

            print(f"{signals_printout}")

        except:
            print("Eval print statistics failed")
            pass

    def _update_defender_model(self):
        """Apply optimizer step and zero gradients. Optimizer steps should actually be taken per x * loss.backward call, which is why log ratios exist and are saved"""
        args = self.args

        # Clip and log gradient magnitude
        if args.max_grad_norm != -1:
            preclip_total_norm = torch.nn.utils.clip_grad_norm_(args.defender.model.parameters(), max_norm=args.max_grad_norm)
            args.logger._log_grad_norm_to_wandb(preclip_total_norm.item(), split="train", step=None, trajectory_id=None)
            if preclip_total_norm > 1.0:
                params_w_grad = [p.grad for p in args.defender.model.parameters() if p.grad != None]
                print(f"Warning: scaled original grad_norm={preclip_total_norm.item():.3f} to grad_norm={torch.nn.utils.get_total_norm(params_w_grad).item():.3f}")
                for param_name, params in args.defender.model.named_parameters():
                    if params.grad != None:
                        print(f"Parameter {param_name} weight mean: {torch.mean(params).item()} abs grad mean: {torch.mean(torch.abs(params.grad)).item()}")
        else:
            params_w_grad = [p.grad for p in args.defender.model.parameters() if p.grad != None]
            args.logger._log_grad_norm_to_wandb(torch.nn.utils.get_total_norm(params_w_grad).item(), split="train", step=None, trajectory_id=None)

        # Step
        args.optimizer.step()
        args.optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def compute_per_token_logps_and_info(model, full_tokens, loss_mask, idx, device='cuda', debug=False, tokenizer=None, disable_forward_enable_grad=False):
        """Compute per_token_logps and other information. Disable_forward_enable_grad should be triggered when running this in no grad mode to prevent some extra overhead."""

        # Prepare Inputs to Model Forward
        input_ids = full_tokens.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        model_inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "use_cache": False,
        }

        # Forward pass with gradients
        forward_ctx = nullcontext() if disable_forward_enable_grad else torch.enable_grad()
        with forward_ctx:
            outputs = model(**model_inputs)
            logits = outputs.logits[0]
        del outputs

        # Obtain Per Token Logps (Shape of unmasked_len)
        # Produce tensors for losses
        labels = full_tokens

        # Shift for Loss Computation
        pre_mask_shift_logits = logits[:-1, :] # (unmasked_len, vocab)
        pre_mask_shift_labels = labels[1:]  # (unmasked_len,)
        cur_mask = loss_mask[idx]
        shift_mask = cur_mask[1:]

        # Mask Logits
        shift_logits = pre_mask_shift_logits[shift_mask]   # new contiguous tensor; IndexBackward only saves the mask, not logits data
        shift_labels = pre_mask_shift_labels[shift_mask] # (unmasked_len)
        if debug:
            print(
                f"{logits.shape=} {labels.shape=} {cur_mask.shape=}",
                f"{pre_mask_shift_logits.shape=} {pre_mask_shift_labels.shape=} {shift_mask.shape=}",
                f"{shift_logits.shape=} {shift_labels.shape=} {shift_mask.shape=}",
                sep="\n"
                )
        del logits, pre_mask_shift_logits  # (seq_len, vocab) freed before log_softmax; safe because shift_logits is a separate copy

        # Obtain logprobs 
        log_softmaxed_logits = torch.log_softmax(shift_logits, dim=-1) # (unmasked_len,vocab), 
        log_probability_indexes_to_gather = shift_labels.unsqueeze(-1) # (unmasked_len,1), 
        per_token_logps = torch.gather(log_softmaxed_logits, -1, log_probability_indexes_to_gather).squeeze(-1) # (unmasked_len)
        if debug:
            # Sanity check: cross_entropy(shift_logits) == -sum(per_token_logps)
            with torch.no_grad():
                sft_loss = F.cross_entropy(shift_logits.to(device), shift_labels.to(device), reduction='sum')
            print("Sanity check per_token_logps, next two items should be equal:", sft_loss, -torch.sum(per_token_logps))
            assert torch.allclose(sft_loss, -torch.sum(per_token_logps))
        del shift_logits  # LogSoftmaxBackward saves its output, not its input; safe to free input now

        # Debug generated probs for the label tokens, none of these should be < .001.
        if debug:
            if tokenizer:
                print(f"per_token_logps: {per_token_logps.tolist()} | words to tune on: {tokenizer.decode(shift_labels)=} | {tokenizer.decode(pre_mask_shift_labels)=}")
            print(
                f"{log_softmaxed_logits.shape=} {log_probability_indexes_to_gather.shape=} {per_token_logps.shape=}",
                )
            # Sanity Check Memory Use
            print(torch.cuda.memory_summary(abbreviated=False))

        return {
            "per_token_logps": per_token_logps,
            "shift_labels": shift_labels,
        }

    def generalized_grpo_like_loss(self, model, completions_in_tokenform: list[TokenTensor], rewards: dict[str, list[float]], loss_mask: list[BooleanTensor], loss_configurations: LossConfigs=None, model_kwarg_keys=None, debug=False):
        """
        Forward pass batch of prompt+completion sequences through model to get logits.
        The model is trained to predict everything that is not masked (it will also not be trained to predict the first token).

        This is similar to GPG rather than DR GRPO.
        GPG takes divdes the final loss by the total number of tokens rather than sequences (maybe prevents full rollouts with more tokens getting more gradients)
        Dr_GRPO actually normalize by max tokens rather than 1/o. (may not matter, in terms of numerical stability.)
        """
        args = self.args

        log("Begin GRPO like loss")
        # Compute Advantages
        first_key = next(iter(rewards))
        total_rewards = [0.0] * len(rewards[first_key])
        for completion_idx, _ in enumerate(rewards[first_key]):
            for reward_type in rewards:
                total_rewards[completion_idx] += rewards[reward_type][completion_idx]
        if len(total_rewards) != len(completions_in_tokenform):
            raise Exception("rewards and logits must have the same number of completions.")

        # DR_GRPO simple subtract by mean
        mean_reward = sum(total_rewards)/len(total_rewards)
        for idx, x in enumerate(total_rewards):
            total_rewards[idx] = x - mean_reward
        device = next(model.parameters()).device
        advantages = torch.tensor(total_rewards, dtype=torch.float32, device=device)
        n_completions = len(completions_in_tokenform)

        # Compute max length of non-masked tokens that we may obtain a loss from
        if loss_configurations.token_level_normalize_type == "max_len":
            token_lengths = [tokens[loss_mask[idx]].shape[-1] for idx, tokens in enumerate(completions_in_tokenform)]
            max_len = max(token_lengths)
            if debug:
                print(f"{token_lengths=}")

        log("Finish adv calculation, begin loops")

        # Compute KL if KL is required for this loss
        if loss_configurations.beta != None:
            model.disable_adapter_layers()
            try: 
                print("Active adapters pre KL", model.get_active_adapters())
            except:
                pass
            kl_ref_per_token_logps = [None] * len(completions_in_tokenform)
            with torch.no_grad():  # old_per_token_logps does not require grad, and if you do it with grad, you will run out of memory because torch autograd graphs will accumulate since backward is never called.
                for idx, full_tokens in enumerate(completions_in_tokenform):
                    ref_per_token_logps = self.compute_per_token_logps_and_info(model, full_tokens, loss_mask, idx, device, debug, args.defender.tokenizer, disable_forward_enable_grad=True)["per_token_logps"]
                    kl_ref_per_token_logps[idx] = ref_per_token_logps.detach()
            model.enable_adapter_layers()
            try: 
                print("Active adapters post KL", model.get_active_adapters())
            except:
                pass

        # Compute old_per_token_logps for log ratio if needed
        old_per_token_logps = [None] * len(completions_in_tokenform)
        if args.gradient_accumulation_steps == args.num_generations or args.num_generations == 1 or args.gradient_accumulation_steps % args.num_generations == 0:
            pass  # No need to compute old_per_token_logps since rollouts will always be on policy
        else:
            log("Begin model forward for log ratio")
            with torch.no_grad():  # old_per_token_logps does not require grad, and if you do it with grad, you will run out of memory because torch autograd graphs will accumulate since backward is never called.
                for idx, full_tokens in enumerate(completions_in_tokenform):
                    # Make a function, forward for logprobs, that forwards the logprobs to populate the old_per_token_logps list
                    orig_per_token_logps = self.compute_per_token_logps_and_info(model, full_tokens, loss_mask, idx, device, debug, args.defender.tokenizer, disable_forward_enable_grad=True)["per_token_logps"]
                    old_per_token_logps[idx] = orig_per_token_logps.detach()
            log("Finished model forward for log ratio")

        log("Begin main loss compute")

        total_loss_for_logging = 0.0
        for idx, full_tokens in enumerate(completions_in_tokenform):

            # Print some debug and skip completions where the tuned-on length exceeds 2000 tokens
            n_tuned_tokens = loss_mask[idx].sum().item()
            print(f"debug: in grpo_like_loss: Sequence Length {len(loss_mask[idx])}, Tuned on Sequence Length {n_tuned_tokens}")
            if n_tuned_tokens > 3000 or len(loss_mask[idx]) > 6000:
                print(f"Warning:Error: Skipping completion {idx}: tuned-on length {n_tuned_tokens} > 3000 or {len(loss_mask[idx])} > 6000 to prevent OOM")
                continue

            # The below is equivalent to 1 iteration of _compute_loss from trl

            # Compute per_token_logps
            per_token_logps_and_info = self.compute_per_token_logps_and_info(model, full_tokens, loss_mask, idx, device, debug, args.defender.tokenizer)
            per_token_logps = per_token_logps_and_info["per_token_logps"]
            shift_labels = per_token_logps_and_info["shift_labels"]
            old_per_token_logps[idx] = per_token_logps.detach() if old_per_token_logps[idx] is None else old_per_token_logps[idx]

            # Compute Log Ratio; should be 0 when num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
            log_ratio = per_token_logps - old_per_token_logps[idx]

            # Grab advantage for current sample
            adv = advantages[idx]

            # Compute Loss and clip loss; where did the -per_token_logps part of the loss go? coef_1 is just the probability ratio new/old, adv is just a scalar.
            coef_1 = torch.exp(log_ratio) # off-policy ratio
            coef_2 = torch.clamp(coef_1, 1 - loss_configurations.epsilon_low, 1 + loss_configurations.epsilon_high) # Clipped off-policy ratio
            per_token_loss1 = coef_1 * adv
            per_token_loss2 = coef_2 * adv
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2) # Full surrogate loss which will backpropagate to give us the -per_token_logps automatically. We must put a negative even though it is not part of the objective since optimizers assume we give a loss (which means it will try to optimize the weights to descent the gradient (decrease the value passed to the optimizer), when we actually want to follow the gradient and increase the value passed to the optimizer).

            # KL per token for stability (requires unaggregated per token loss)
            if loss_configurations.beta != None:
                per_token_kl = (
                    torch.exp(kl_ref_per_token_logps - per_token_logps) - (kl_ref_per_token_logps - per_token_logps) - 1
                )
                per_token_loss = per_token_loss + loss_configurations.beta * per_token_kl

            # Optional Second Loss Masking Stage
            if loss_configurations.strip_pad_tokens:
                second_stage_mask = torch.ones_like(per_token_loss, dtype=torch.bool)
                pad_id = args.defender.tokenizer.pad_token_id
                pad_positions = (shift_labels == pad_id)
                second_stage_mask = second_stage_mask & ~pad_positions.to(second_stage_mask.device)
                pad_stripped = torch.sum(pad_positions)
                per_token_loss = per_token_loss[second_stage_mask]
                if debug:
                    print(f"debug: stripped pad token {args.defender.tokenizer.pad_token_id}, {args.defender.tokenizer.decode(args.defender.tokenizer.pad_token_id)}. This many pad tokens were stripped: {pad_stripped.item()=}")
                    try:
                        print(f"debug: Labels to tune on after stripping pad token {args.defender.tokenizer.decode(shift_labels[second_stage_mask])}")
                    except:
                        pass
                del second_stage_mask, pad_positions
                        

            # Length Normalization
            if loss_configurations.token_level_normalize_type == "token":
                loss_i = (per_token_loss.sum(-1) / per_token_loss.shape[-1])
                debug and print(f"pre mean {loss_i=} {coef_1=} {coef_2=} {log_ratio=}")
            elif loss_configurations.token_level_normalize_type == "max_len":
                loss_i = (per_token_loss.sum(-1) / max_len)
                debug and print(f"pre mean {loss_i=} {coef_1=} {coef_2=} {log_ratio=}")
            elif loss_configurations.token_level_normalize_type == "sequence_only":
                loss_i = per_token_loss.sum(-1)
                debug and print(f"pre mean {loss_i=} {coef_1=} {coef_2=} {log_ratio=}")
            normalizer = args.gradient_accumulation_steps
            loss_i = loss_i / normalizer # for pariety in terms of the scale of the gradient per optimizer step.

            if debug:
                print(
                    f"{log_ratio.shape=} {coef_1.shape=} {coef_2.shape=} {per_token_loss.shape=}",
                    f"{loss_i.shape=} {per_token_loss.sum(-1).shape=} {per_token_loss.shape[-1]=}",
                    sep="\n"
                    )
                if loss_configurations.token_level_normalize_type == "max_len":
                    print(f"{max_len=}")

            # Backward pass
            loss_i.backward()
            self.gradients_accumulated_count += 1
            total_loss_for_logging += loss_i.detach().item()
            del per_token_logps_and_info, per_token_logps, shift_labels, log_ratio, coef_1, coef_2, per_token_loss1, per_token_loss2, per_token_loss, loss_i

            # Do optimization step if enough backward passes happened
            if self.gradients_accumulated_count % args.gradient_accumulation_steps == 0:
                self._update_defender_model()
                self.gradients_accumulated_count = 0
            
        log("Finish loss compute")

        if n_completions == 0:
            raise Exception("No completions provided; cannot compute DR-GRPO loss.")
        return total_loss_for_logging

    def compute_token_mask(self, model, tokenizer, prompt_messages: list[dict], completions_in_tokenform: list[TokenTensor], debug=False, loss_masking="last_completion_only"):
        """Computes Masks and passes along arguments. Prompt messages will be None when loss_masking = 'assistant_only'; does not support thinking mode templates."""

        def _qwen3_assistant_only_mask(tensor_of_full_tokens: torch.Tensor) -> torch.Tensor:
            """
            Returns a boolean mask of shape [seq_len] where True marks tokens belonging to assistant-role spans.
            Assumes ChatML-ish structure: <|im_start|>assistant ... <|im_end|>
            """
            # Assert Qwen3
            model_type = getattr(getattr(model, "config", None), "model_type", None)
            assert model_type == "qwen3", f"assistant_only expects Qwen3; got model.config.model_type={model_type!r}"

            # Hidden-error guard: ensure 1D
            if tensor_of_full_tokens.dim() != 1:
                raise Exception(f"Expected 1D token tensor, got shape={tuple(tensor_of_full_tokens.shape)}")

            seq_len = tensor_of_full_tokens.shape[0]
            cur_mask = torch.zeros(seq_len, dtype=torch.bool, device=tensor_of_full_tokens.device)

            im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")

            # If tokenizer doesn't know these, masking cannot be reliable.
            if im_start_id is None or im_start_id < 0 or im_end_id is None or im_end_id < 0:
                raise Exception("Tokenizer missing <|im_start|>/<|im_end|> ids; cannot build assistant_only mask reliably for Qwen3.")

            # Qwen chat template writes role label right after <|im_start|>
            assistant_role_ids = tokenizer.encode("assistant", add_special_tokens=False)

            # Scan for: im_start + assistant_role_ids, then mark until next im_end
            i = 0
            while i < seq_len:
                if int(tensor_of_full_tokens[i]) == int(im_start_id):
                    j = i + 1
                    # Check for assistant role label
                    ok = True
                    if j + len(assistant_role_ids) <= seq_len:
                        for k, rid in enumerate(assistant_role_ids):
                            if int(tensor_of_full_tokens[j + k]) != int(rid):
                                ok = False
                                break
                    else:
                        ok = False

                    if ok:
                        # Start marking after the role label.
                        # (We intentionally keep <|im_start|> and the role label itself masked out.)
                        start = j + len(assistant_role_ids)

                        # Often there's a newline right after the role; include it as assistant span
                        # (harmless either way; newline tokenization varies).
                        end = start
                        while end < seq_len and int(tensor_of_full_tokens[end]) != int(im_end_id):
                            end += 1

                        if start < end:
                            cur_mask[start:end] = True

                        # Jump to end marker (or seq end)
                        i = end + 1
                        continue
                i += 1

            return cur_mask

        def _mistral_assistant_only_mask(tensor_of_full_tokens: torch.Tensor) -> torch.Tensor:
            """
            Returns a boolean mask of shape [seq_len] where True marks assistant text spans.
            Assumes Mistral Instruct-ish structure: ... [/INST] assistant_text [INST] ...
            """
            model_type = getattr(getattr(model, "config", None), "model_type", None)
            assert model_type in {"mistral", "mistral3"}, f"assistant_only expects Mistral/Ministral; got model.config.model_type={model_type!r}"

            if tensor_of_full_tokens.dim() != 1:
                raise Exception(f"Expected 1D token tensor, got shape={tuple(tensor_of_full_tokens.shape)}")

            inst_start_ids = tokenizer.encode("[INST]", add_special_tokens=False)
            inst_end_ids = tokenizer.encode("[/INST]", add_special_tokens=False)
            if len(inst_start_ids) == 0 or len(inst_end_ids) == 0:
                raise Exception("Tokenizer missing [INST]/[/INST] tokenization; cannot build assistant_only mask reliably for Mistral/Ministral.")

            seq_len = tensor_of_full_tokens.shape[0]
            cur_mask = torch.zeros(seq_len, dtype=torch.bool, device=tensor_of_full_tokens.device)

            def _matches_at(position: int, pattern_ids: list[int]) -> bool:
                if position + len(pattern_ids) > seq_len:
                    return False
                for idx, pid in enumerate(pattern_ids):
                    if int(tensor_of_full_tokens[position + idx]) != int(pid):
                        return False
                return True

            i = 0
            while i < seq_len:
                if _matches_at(i, inst_end_ids):
                    start = i + len(inst_end_ids)
                    end = start
                    while end < seq_len and not _matches_at(end, inst_start_ids):
                        end += 1

                    if start < end:
                        cur_mask[start:end] = True
                    i = end
                    continue
                i += 1

            return cur_mask

        def _gemma3_assistant_only_mask(tensor_of_full_tokens: torch.Tensor) -> torch.Tensor:
            """
            Returns a boolean mask of shape [seq_len] where True marks tokens belonging to model-role spans.
            Assumes Gemma 3 structure: <start_of_turn>model\n...content...<end_of_turn>
            """
            model_type = getattr(getattr(model, "config", None), "model_type", None)
            assert model_type in {"gemma3", "gemma3_text"}, f"_gemma3_assistant_only_mask expects Gemma 3; got model.config.model_type={model_type!r}"

            if tensor_of_full_tokens.dim() != 1:
                raise Exception(f"Expected 1D token tensor, got shape={tuple(tensor_of_full_tokens.shape)}")

            seq_len = tensor_of_full_tokens.shape[0]
            cur_mask = torch.zeros(seq_len, dtype=torch.bool, device=tensor_of_full_tokens.device)

            start_of_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
            end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")

            if start_of_turn_id is None or end_of_turn_id is None:
                raise Exception("Tokenizer missing <start_of_turn>/<end_of_turn> ids; cannot build assistant_only mask for Gemma 3.")

            # Gemma 3 uses "model" as the assistant role label
            model_role_ids = tokenizer.encode("model", add_special_tokens=False)

            i = 0
            while i < seq_len:
                if int(tensor_of_full_tokens[i]) == int(start_of_turn_id):
                    j = i + 1
                    ok = True
                    if j + len(model_role_ids) <= seq_len:
                        for k, rid in enumerate(model_role_ids):
                            if int(tensor_of_full_tokens[j + k]) != int(rid):
                                ok = False
                                break
                    else:
                        ok = False

                    if ok:
                        # Start marking after the role label (skip <start_of_turn> and "model").
                        start = j + len(model_role_ids)

                        end = start
                        while end < seq_len and int(tensor_of_full_tokens[end]) != int(end_of_turn_id):
                            end += 1

                        if start < end:
                            cur_mask[start:end] = True

                        i = end + 1
                        continue
                i += 1

            return cur_mask

        if loss_masking == "last_completion_only":
            # Tokenize completions and prompts
            # Apply chat template to prompt
            prompt_str = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            # Tokenize prompt once to get its length
            prompt_tokens = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt")
            prompt_length = prompt_tokens.input_ids.shape[1]

        if debug:
            print(f"compute_token_mask: {len(completions_in_tokenform)} completions, loss_masking={loss_masking!r}")
            print(f"  Detokenized first completion: {tokenizer.decode(completions_in_tokenform[0], skip_special_tokens=False)}")

        masks = []
        for tensor_of_full_tokens in completions_in_tokenform:
            cur_mask = torch.zeros(tensor_of_full_tokens.shape[0], dtype=torch.bool)
            if loss_masking == "last_completion_only":
                cur_mask[prompt_length:] = True        # typically start at prompt_length (not prompt_length-1)
            elif loss_masking == "none":
                cur_mask[:] = True
            elif loss_masking == "assistant_only":
                model_type = getattr(getattr(model, "config", None), "model_type", None)
                if model_type == "qwen3":
                    cur_mask = _qwen3_assistant_only_mask(tensor_of_full_tokens)
                elif model_type in {"mistral", "mistral3"}:
                    cur_mask = _mistral_assistant_only_mask(tensor_of_full_tokens)
                elif model_type in {"gemma3", "gemma3_text"}:
                    cur_mask = _gemma3_assistant_only_mask(tensor_of_full_tokens)
                else:
                    raise Exception(f"assistant_only not implemented for model_type={model_type!r}")
                # Sanity check: must find at least some assistant tokens
                assert bool(cur_mask.any().item()), "assistant_only mask found zero assistant tokens; check that tokens include ChatML markers."
            else:
                raise Exception("Not implemented yet.")
            masks.append(cur_mask)

        if debug:
            print(f"debug: Full Tensors (of the last completion): {tokenizer.decode(tensor_of_full_tokens)}")
            print(f"debug: Words to tune on (of the last completion): {tokenizer.decode(tensor_of_full_tokens[cur_mask])}")

        return masks
        
    def _train_step_stepwise(self, reward: dict[list], defender_prompt, candidates, step=None, trajectory_id=None):
        args = self.args

        # Defender training step
        if args.loss_type is None:
            raise Exception("loss_type is None; pass loss_type explicitly to Trajectory(..., loss_type=...).")

        # Select the Correct Loss
        if args.loss_type == "dr_grpo":
            loss_mask = self.compute_token_mask(
                model=args.defender.model,
                tokenizer=args.defender.tokenizer,
                prompt_messages=defender_prompt,
                completions_in_tokenform=candidates["full_tokens"],
                debug=True if trajectory_id < 3 else False,
                loss_masking="last_completion_only"
            )

            loss_configurations = LossConfigs(epsilon_low=0.2, epsilon_high=0.2, token_level_normalize_type="max_len", beta=None)

            loss_value = self.generalized_grpo_like_loss(
                model=args.defender.model,
                completions_in_tokenform=candidates["full_tokens"],
                rewards=reward,
                loss_mask=loss_mask,
                loss_configurations=loss_configurations,
                model_kwarg_keys=args.model_kwarg_keys,
                debug=True if trajectory_id < 3 else False
            )
        elif args.loss_type == "grpo":
            loss_mask = self.compute_token_mask(
                model=args.defender.model,
                tokenizer=args.defender.tokenizer,
                prompt_messages=defender_prompt,
                completions_in_tokenform=candidates["full_tokens"],
                debug=True if trajectory_id < 3 else False,
                loss_masking="last_completion_only"
            )

            loss_configurations = LossConfigs(epsilon_low=0.2, epsilon_high=0.2, token_level_normalize_type="max_len", beta=0.04)  # beta set same as deepseek math

            loss_value = self.generalized_grpo_like_loss(
                model=args.defender.model,
                completions_in_tokenform=candidates["full_tokens"],
                rewards=reward,
                loss_mask=loss_mask,
                loss_configurations=loss_configurations,
                model_kwarg_keys=args.model_kwarg_keys,
                debug=True if trajectory_id < 3 else False
            )
        elif args.loss_type == "dr_grpo_with_only_seqnorm":
            loss_mask = self.compute_token_mask(
                model=args.defender.model,
                tokenizer=args.defender.tokenizer,
                prompt_messages=defender_prompt,
                completions_in_tokenform=candidates["full_tokens"],
                debug=True if trajectory_id < 3 else False,
                loss_masking="last_completion_only"
            )

            loss_configurations = LossConfigs(epsilon_low=0.2, epsilon_high=0.2, token_level_normalize_type="sequence_only", beta=None)

            loss_value = self.generalized_grpo_like_loss(
                model=args.defender.model,
                completions_in_tokenform=candidates["full_tokens"],
                rewards=reward,
                loss_mask=loss_mask,
                loss_configurations=loss_configurations,
                model_kwarg_keys=args.model_kwarg_keys,
                debug=True if trajectory_id < 3 else False
            )
        elif args.loss_type == "dr_grpo_with_tokennorm":
            loss_mask = self.compute_token_mask(
                model=args.defender.model,
                tokenizer=args.defender.tokenizer,
                prompt_messages=defender_prompt,
                completions_in_tokenform=candidates["full_tokens"],
                debug=True if trajectory_id < 3 else False,
                loss_masking="last_completion_only"
            )

            loss_configurations = LossConfigs(epsilon_low=0.2, epsilon_high=0.2, token_level_normalize_type="token", beta=None)

            loss_value = self.generalized_grpo_like_loss(
                model=args.defender.model,
                completions_in_tokenform=candidates["full_tokens"],
                rewards=reward,
                loss_mask=loss_mask,
                loss_configurations=loss_configurations,
                model_kwarg_keys=args.model_kwarg_keys,
                debug=True if trajectory_id < 3 else False
            )
        else:
            raise Exception(f"Unknown loss_type: {args.loss_type}")

        print(f"Defender loss: {loss_value}")
        args.logger._log_defender_loss_to_wandb(loss_value, split="train", step=step, trajectory_id=trajectory_id)
        
        if not args.loss_type in ["dr_grpo", "dr_grpo_with_only_seqnorm", "dr_grpo_with_tokennorm", "grpo"]:
            self.gradients_accumulated_count += 1
            if self.gradients_accumulated_count % args.gradient_accumulation_steps == 0:
                self._update_defender_model()
                self.gradients_accumulated_count = 0

    def run_train(self):
        """Run stepwise training loop: for each sample, create a Trajectory and run subrollout one step at a time, training after each step."""
        args = self.args
        attacker, defender = args.attacker, args.defender

        trajectory_number = 0
        next_eval_idx = 1
        for epoch in range(args.epochs):
            for sample in args.train_dataset:
                
                # Extract information from sample
                attacker_target_information = sample["attacker_target_information"]
                defender_private_information = sample["defender_private_information"]
                
                # Initialize Attacker and Defender State
                attacker.update_attacker_state(attacker_target_information=attacker_target_information, fresh_start=True, prompt_id=args.train_prompt_id)
                defender.update_defender_state(defender_private_information=defender_private_information, fresh_start=True)
                trajectory = Trajectory(
                    attacker=attacker,
                    defender=defender,
                    defender_optimizer=args.optimizer,
                    max_turns=args.max_iterations,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    num_generations=args.num_generations,
                    reward_functions=args.reward_funcs,
                    trajectory_id=trajectory_number,
                    wandb_run=args.wandb_run,
                    wandb_step_timing_counter=args.wandb_step_timing_counter,
                    loss_type=args.loss_type,
                    max_completion_length=args.max_completion_length,
                    model_kwarg_keys=args.model_kwarg_keys,
                    logger=args.logger,
                    judge_prompt_version=args.judge_prompt_version,
                    trajectory_level_rewards=args.trajectory_level_rewards,
                    judge_client=args.judge_client,
                    judge_model_name=args.judge_model_name,
                )

                stopped = False
                while not stopped:
                    training_output = trajectory.subrollout(steps=1, debug_prompts=trajectory_number < 3)
                    if training_output["latest_reward"] != None: # Check if we got new rewards this turn.
                        self._train_step_stepwise(training_output["latest_reward"], training_output["latest_defender_prompt"], training_output["latest_candidates"], step=trajectory.step - 1, trajectory_id=trajectory.trajectory_id)
                        del training_output["latest_candidates"]  # free full_tokens GPU tensors; no longer needed after train step
                    stopped = training_output["stopped"]

                # Logging
                if True: # or trajectory_number < 3:
                    print_trajectory_like_setting_simplified_iterated(
                        trajectory_output=training_output,
                        idx=trajectory_number,
                        label="Training",
                    )
                del trajectory
                gc.collect()
                torch.cuda.empty_cache()

                if args.cuda_memory_snapshot_path:
                    torch.cuda.memory._dump_snapshot(args.cuda_memory_snapshot_path)
                    print(f"CUDA memory snapshot dumped {args.cuda_memory_snapshot_path}", flush=True)
                
                print(f"Sample trajectory_number {trajectory_number} complete.", flush=True)
                trajectory_number += 1
                # Step Scheduler
                args.scheduler.step()

                # Trigger Eval
                if next_eval_idx <= len(args.eval_after_trajectory_counts) and trajectory_number == args.eval_after_trajectory_counts[next_eval_idx - 1]:      
                    self.run_eval(trajectory_number, next_eval_idx)
                    if hasattr(args.model, 'train'):
                        args.model.train()
                    next_eval_idx += 1



class TrajectorywiseGRPOTrainer(BaseTrainer):

    def _train_step_trajectory_level(self, reward: dict[list], defender_histories, completions_in_tokenform: list, step=None, trajectory_id=None, force_debug=False):
        """Format-failure debug printing (force_debug) is supported here for TrajectorywiseGRPOTrainer."""
        args = self.args

        # Defender training step
        if args.loss_type is None:
            raise Exception("loss_type is None; pass loss_type explicitly to Trajectory(..., loss_type=...).")

        reward_is_nonzero = sum(abs(v) for key in reward for v in reward[key]) > 0
        debug = trajectory_id < 3 or force_debug
        # Select the Correct Loss
        if args.loss_type == "dr_grpo":
            loss_mask = self.compute_token_mask(
                model=args.defender.model,
                tokenizer=args.defender.tokenizer,
                prompt_messages=defender_histories,
                completions_in_tokenform=completions_in_tokenform,
                debug=debug,
                loss_masking="assistant_only"
            )
            loss_configurations = LossConfigs(epsilon_low=0.2, epsilon_high=0.2, token_level_normalize_type="max_len", beta=None)
            loss_value = self.generalized_grpo_like_loss(
                model=args.defender.model,
                completions_in_tokenform=completions_in_tokenform,
                rewards=reward,
                loss_mask=loss_mask,
                loss_configurations=loss_configurations,
                model_kwarg_keys=args.model_kwarg_keys,
                debug=debug
            )
        else:
            raise Exception(f"Unknown loss_type: {args.loss_type}")

        print(f"Defender loss: {loss_value}")
        args.logger._log_defender_loss_to_wandb(loss_value, split="train", step=step, trajectory_id=trajectory_id)
        
        if not args.loss_type in ["dr_grpo", "dr_grpo_with_only_seqnorm", "dr_grpo_with_tokennorm", "grpo"]:
            self.gradients_accumulated_count += 1
            if self.gradients_accumulated_count % args.gradient_accumulation_steps == 0:
                self._update_defender_model()
                self.gradients_accumulated_count = 0


    def run_train(self):
        args = self.args
        attacker, defender = args.attacker, args.defender

        steps_per_policy_sync = -1
        generator_lora_name = ""
        trained_lora_name = ""
        trajectory_number = 0
        next_eval_idx = 1
        for epoch in range(args.epochs):
            for sample in args.train_dataset:
                
                # Extract information from sample
                attacker_target_information = sample["attacker_target_information"]
                defender_private_information = sample["defender_private_information"]


                # Parallel execution
                def rollout_single_trajectory(attacker_target_information, defender_private_information):
                    # Initialize Attacker and Defender State
                    cur_attacker = attacker.copy()
                    cur_defender = defender.copy()
                    cur_attacker.update_attacker_state(attacker_target_information=attacker_target_information, fresh_start=True, prompt_id=args.train_prompt_id)
                    cur_defender.update_defender_state(defender_private_information=defender_private_information, fresh_start=True)

                    trajectory = Trajectory(
                        attacker=cur_attacker,
                        defender=cur_defender,
                        defender_optimizer=args.optimizer,
                        max_turns=args.max_iterations,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        num_generations=1,
                        reward_functions=args.reward_funcs,
                        trajectory_id=trajectory_number,
                        wandb_run=args.wandb_run,
                        wandb_step_timing_counter=args.wandb_step_timing_counter,
                        loss_type=args.loss_type,
                        max_completion_length=args.max_completion_length,
                        model_kwarg_keys=args.model_kwarg_keys,
                        logger=args.logger,
                        enable_lock_on_generate=args.enable_lock_on_generate,
                        judge_prompt_version=args.judge_prompt_version,
                        trajectory_level_rewards=args.trajectory_level_rewards,
                        judge_client=args.judge_client,
                        judge_model_name=args.judge_model_name,
                        max_format_retries=args.max_format_retries,
                    )

                    training_output = trajectory.subrollout(steps=-1, num_generations_override=1, compute_core_trajectory_rewards=True, debug_prompts=debug_prompts)

                    return training_output
                
                # Launch num_generations trajectories
                debug_prompts = trajectory_number < 10
                rewards = {"total_return": [0.0] * args.num_generations}
                per_generation_reward_means = [None] * args.num_generations
                histories = [None] * args.num_generations
                completions_in_tokenform = [None] * args.num_generations
                nonzero_format_rwd_trajectory_count = 0
                all_training_outputs = [None] * args.num_generations
                with ThreadPoolExecutor(max_workers=args.num_generations) as executor:
                    futures = [executor.submit(rollout_single_trajectory, attacker_target_information, defender_private_information) for _ in range(args.num_generations)]
                    training_output_for_logging = None
                    for idx, future in enumerate(futures):
                        # Gather results
                        training_output = future.result()
                        all_training_outputs[idx] = training_output

                        # Save for Logging
                        if idx == 0:
                            training_output_for_logging = training_output
                            print(f"debug: {training_output_for_logging['core_trajectory_rewards_per_turn']=}")
                        
                        # Save completions for loss computation
                        histories[idx] = training_output["defender_conversation_history"]
                        completions_in_tokenform[idx] = training_output["latest_picked_defender_response_full_tokens"]
                        
                        # Gather rewards for loss computation
                        reward_sums_this_generation = {}
                        reward_turns_count_per_type = {}
                        has_zero_fmt_reward = False
                        for reward_dict in training_output["core_trajectory_rewards_per_turn"]:
                            if not reward_dict:  # Skip if no rewards this turn to avoid issues when we only use final reward.
                                continue
                            if "format_rwd" in reward_dict and reward_dict["format_rwd"] == 0:
                                has_zero_fmt_reward = True
                            for reward_name in reward_dict:
                                if reward_name == "format_rwd":
                                    continue
                                reward_sums_this_generation.setdefault(reward_name, 0.0)
                                reward_sums_this_generation[reward_name] += reward_dict[reward_name]
                                reward_turns_count_per_type.setdefault(reward_name, 0)
                                reward_turns_count_per_type[reward_name] += 1
                        per_generation_reward_means[idx] = {
                            reward_name: reward_sums_this_generation[reward_name] / reward_turns_count_per_type[reward_name]
                            for reward_name in reward_sums_this_generation
                        }
                        rewards["total_return"][idx] = sum(per_generation_reward_means[idx].values())
                        if args.convex_joint:
                            rewards["total_return"][idx] /= len(reward_sums_this_generation)
                        if has_zero_fmt_reward:
                            rewards["total_return"][idx] = 0.0
                        else:
                            nonzero_format_rwd_trajectory_count += 1

                # Logging for rewards
                all_reward_names = sorted({
                    reward_name
                    for generation_reward_means in per_generation_reward_means
                    for reward_name in generation_reward_means
                })
                for reward_name in all_reward_names:
                    mean_reward_component = sum(
                        generation_reward_means.get(reward_name, 0.0)
                        for generation_reward_means in per_generation_reward_means
                    ) / len(per_generation_reward_means)
                    print(f"Trajectory mean {reward_name}: {mean_reward_component}")
                    args.logger._wandb_log({
                        f"train/trajectory_mean_reward_component/{reward_name}": mean_reward_component,
                        "train/trajectory/id": trajectory_number,
                    })
                mean_total_return = sum(rewards["total_return"]) / len(rewards["total_return"])
                print(f"Trajectory mean total_return: {mean_total_return}")
                args.logger._wandb_log({
                    "train/trajectory_mean_total_return": mean_total_return,
                    "train/trajectory/id": trajectory_number,
                })
                print(f"Trajectory nonzero format_rwd count: {nonzero_format_rwd_trajectory_count}")
                args.logger._wandb_log({
                    "train/trajectory_nonzero_format_rwd_count": nonzero_format_rwd_trajectory_count,
                    "train/trajectory/id": trajectory_number,
                })

                # Format failure: print all parallel trajectories with per-step rewards
                format_failure_debug = nonzero_format_rwd_trajectory_count < args.num_generations
                if format_failure_debug:
                    print(f"=== FORMAT FAILURE DEBUG (trajectory {trajectory_number}): {nonzero_format_rwd_trajectory_count}/{args.num_generations} had valid format ===")
                    for gen_idx, t_output in enumerate(all_training_outputs):
                        print(f"--- Generation {gen_idx} | final total_return (passed to training): {rewards['total_return'][gen_idx]} ---")
                        for turn_idx, reward_dict in enumerate(t_output["core_trajectory_rewards_per_turn"]):
                            if not reward_dict:
                                print(f"  Turn {turn_idx}: (no rewards)")
                                continue
                            print(f"  Turn {turn_idx}: {reward_dict}")
                        for msg in t_output["conversation_histories"]:
                            print(f"  {msg['role']}: {msg['content']}")
                    print(f"=== END FORMAT FAILURE DEBUG ===")
                
                self._train_step_trajectory_level(
                    rewards,
                    histories,
                    completions_in_tokenform=completions_in_tokenform,
                    step=training_output_for_logging["step"],
                    trajectory_id=training_output_for_logging["trajectory_id"],
                    force_debug=format_failure_debug,
                )
                del completions_in_tokenform, histories  # free full_tokens GPU tensors; no longer needed after train step

                # Logging (use first rollout's output as representative; no single trajectory object to delete here)
                if True: # or trajectory_number < 3:
                    print_trajectory_like_setting_simplified_iterated(
                        trajectory_output=training_output_for_logging,
                        idx=trajectory_number,
                        label="Training",
                    )
                gc.collect()
                torch.cuda.empty_cache()

                if args.cuda_memory_snapshot_path:
                    torch.cuda.memory._dump_snapshot(args.cuda_memory_snapshot_path)
                    print(f"CUDA memory snapshot dumped {args.cuda_memory_snapshot_path}", flush=True)
                
                print(f"Sample trajectory_number {trajectory_number} complete.", flush=True)
                trajectory_number += 1

                if next_eval_idx <= len(args.eval_after_trajectory_counts) and trajectory_number == args.eval_after_trajectory_counts[next_eval_idx - 1]:      
                    self.run_eval(trajectory_number, next_eval_idx)
                    if hasattr(args.model, 'train'):
                        args.model.train()
                    next_eval_idx += 1


