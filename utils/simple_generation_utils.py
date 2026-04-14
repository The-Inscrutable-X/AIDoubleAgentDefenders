from openai import OpenAI
import torch
from pydantic import BaseModel, Field
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

global_token_costs = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

def update_global_token_costs(usage):
    for i in usage:
        if i in global_token_costs:
            global_token_costs[i] += usage[i]

def get_global_token_costs():
    return global_token_costs

def reset_global_token_costs():
    for i in global_token_costs:
        global_token_costs[i] = 0

def is_gpt_model(model_name: str) -> bool:
    if model_name is None:
        return False
    return model_name.lower().startswith("gpt")

def _max_tokens_kwarg(model_name, value):
    """GPT models on Azure require max_completion_tokens; others use max_tokens."""
    if is_gpt_model(model_name):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}

def generate_openai(
    client: OpenAI,
    model_name: str,
    prompt: str | list,
    generate_kwargs: dict = None,
    apply_chat_template: bool = False,
    verbose = False,
    system_message = None,
    output_json_schema = None
):
    """Generation from openai compatible API"""
    # Set default generate_kwargs if None
    if output_json_schema:
        schema_dict = output_json_schema.model_json_schema()["properties"]
        schema_name = getattr(output_json_schema, "__name__", "output")
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": output_json_schema.__name__,
                "schema": output_json_schema.model_json_schema(),  # pydantic v2
            },
        }
    else:
        response_format = None

    if generate_kwargs is None:
        generate_kwargs = {}
    
    temperature = generate_kwargs.get("temperature", 0.7)
    top_p = generate_kwargs.get("top_p", 1.0)
    max_tokens = generate_kwargs.get("max_tokens", generate_kwargs.get("max_new_tokens", 30000))

    if apply_chat_template:
        if system_message:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [
                {"role": "user", "content": prompt}
            ]
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            **_max_tokens_kwarg(model_name, max_tokens),
            response_format=response_format
        )
        verbose and print(chat_response)
        # Track token usage
        if hasattr(chat_response, 'usage') and chat_response.usage:
            update_global_token_costs({
                "prompt_tokens": chat_response.usage.prompt_tokens,
                "completion_tokens": chat_response.usage.completion_tokens,
                "total_tokens": chat_response.usage.total_tokens
            })
        generated_text = chat_response.choices[0].message.content
    else:
        verbose and print(f"Pinging API with {model_name=} {prompt=}")
        if isinstance(prompt, list):
            if len(prompt) == 0:
                raise ValueError("Messages list cannot be empty")
            for i, msg in enumerate(prompt):
                if not isinstance(msg, dict):
                    raise TypeError(f"Message at index {i} must be a dict, got {type(msg)}")
                if "role" not in msg:
                    raise ValueError(f"Message at index {i} missing 'role' key")
                if "content" not in msg:
                    raise ValueError(f"Message at index {i} missing 'content' key")
                if not isinstance(msg["content"], str) or len(msg["content"].strip()) == 0:
                    raise ValueError(f"Message at index {i} has empty or invalid 'content'")
        else:
            raise TypeError(f"Expected messages to be a list when apply_chat_template=False, got {type(prompt)}")
        
        # Ensure there's at least one user message (required by some APIs like Gemini)
        messages = prompt.copy()
        has_user_message = any(msg.get("role") == "user" for msg in messages)
        if not has_user_message:
            # If there's only a system message, add a default user message
            messages.append({"role": "user", "content": "Please respond."})
        
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            **_max_tokens_kwarg(model_name, max_tokens),
            response_format=response_format
        )
        verbose and print(chat_response)
        # Track token usage
        if hasattr(chat_response, 'usage') and chat_response.usage:
            update_global_token_costs({
                "prompt_tokens": chat_response.usage.prompt_tokens,
                "completion_tokens": chat_response.usage.completion_tokens,
                "total_tokens": chat_response.usage.total_tokens
            })
        generated_text = chat_response.choices[0].message.content
    return generated_text


def generate_offline(
    model, tokenizer, input_prompt, generate_kwargs=None, use_vllm=True, apply_chat_template=False,
    system_message=None, verbose=True, use_reasoning=False, using_unsloth=False,
    batched_eval=False, lora_request=None, input_prompt_is_messages=True, defender_lora_path=None,
    num_generations=1, return_logits=False, skip_special_tokens=True, output_json_schema=None, return_tokens=False
):
    """
    Generate text using either local model or vLLM.
    generate_kwargs specify args for huggingface generate, which defaults to values set here.
    If batched_eval is True, input_prompt should be a list of strings.
    return_tokens: Will return all tokens including the input tokens
    """
    # Spawn default generate_kwargs if None
    if generate_kwargs is None:
        generate_kwargs = {}
    # Set default values for sampling_kwargs individually, which are the arguments actually passed in to generating functions
    sampling_kwargs = dict(
            temperature=generate_kwargs.get('temperature', 0.7),
            top_p=generate_kwargs.get('top_p', 0.8),
            repetition_penalty=generate_kwargs.get('repetition_penalty', 1.05),
            max_tokens=generate_kwargs.get('max_tokens', generate_kwargs.get('max_new_tokens', 20000))
        )
    if output_json_schema:
        sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=output_json_schema.model_json_schema())
    
    # Create LoRA request if defender_lora_path is provided
    if defender_lora_path and use_vllm and not lora_request:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("defender_lora", 1, defender_lora_path)
    
    # If batched_eval, input_prompt is a list; else, it's a string
    if batched_eval:
        if not use_vllm:
            raise NotImplementedError("Batched eval is only implemented for vLLM backend.")
        if return_logits:
            raise NotImplementedError("return_logits is not supported for vLLM backend.")
        if num_generations != 1:
            raise NotImplementedError("num_generations is not supported for vLLM backend.")
        # Apply chat template if requested
        prompts = input_prompt
        if input_prompt_is_messages:
            prompts_with_template = []
            for prompt in prompts:
                prompt_with_template = tokenizer.apply_chat_template(
                    input_prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=use_reasoning
                )
                prompts_with_template.append(prompt_with_template)
            prompts = prompts_with_template
        elif apply_chat_template:
            prompts_with_template = []
            for prompt in prompts:
                if system_message:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                prompt_with_template = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=use_reasoning
                )
                prompts_with_template.append(prompt_with_template)
            prompts = prompts_with_template
        # Prepare sampling params for vllm
        sampling_kwargs = dict(
            temperature=generate_kwargs.get('temperature', 0.7),
            top_p=generate_kwargs.get('top_p', 0.8),
            repetition_penalty=generate_kwargs.get('repetition_penalty', 1.05),
            max_tokens=generate_kwargs.get('max_tokens', generate_kwargs.get('max_new_tokens', 20000))
        )
        if output_json_schema:
            sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=output_json_schema.model_json_schema())
        sampling_params = SamplingParams(**sampling_kwargs)
        # Generate in batch
        if using_unsloth:
            print(f"Generating with lora_request {lora_request}")
            outputs = model.fast_generate(prompts, sampling_params, lora_request=lora_request)
        else:
            if lora_request:
                outputs = model.generate(prompts, sampling_params, lora_request=lora_request)
            else:
                outputs = model.generate(prompts, sampling_params)
        # Extract generated text, strip prompt if present
        responses = []
        for prompt, output in zip(prompts, outputs):
            generated_text = output.outputs[0].text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]
            responses.append(generated_text)
        return responses
    else: # None Batched Generation
        # Apply chat template if requested
        original_input_prompt = input_prompt
        if input_prompt_is_messages:
            input_prompt = tokenizer.apply_chat_template(
                input_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=use_reasoning
            )
        elif apply_chat_template:
            if system_message:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": input_prompt}
                ]
            else:
                messages = [
                    {"role": "user", "content": input_prompt}
                ]
            
            input_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=use_reasoning
            )
        
        # VLLM no batched eval
        if use_vllm:
            if return_logits:
                raise NotImplementedError("return_logits is not supported for vLLM backend.")
            if num_generations != 1:
                raise NotImplementedError("num_generations is not supported for vLLM backend.")

            # Convert generate_kwargs to SamplingParams for vlllm
            sampling_kwargs = dict(
                temperature=generate_kwargs.get('temperature', 0.7),
                top_p=generate_kwargs.get('top_p', 0.8),
                repetition_penalty=generate_kwargs.get('repetition_penalty', 1.05),
                max_tokens=generate_kwargs.get('max_tokens', generate_kwargs.get('max_new_tokens', 20000))
            )
            if output_json_schema:
                sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=output_json_schema.model_json_schema())
            sampling_params = SamplingParams(**sampling_kwargs)
            
            # Generate using vLLM
            verbose and print(f"\nInput Prompt:\n{input_prompt}")
            if using_unsloth:
                outputs = model.fast_generate([input_prompt], sampling_params, lora_request=lora_request)
            else:
                if lora_request:
                    outputs = model.generate([input_prompt], sampling_params, lora_request=lora_request)
                else:
                    outputs = model.generate([input_prompt], sampling_params)
            
            # Extract generated text
            response = []
            for output in outputs:
                generated_text = output.outputs[0].text
                if generated_text.startswith(input_prompt):
                    generated_text = generated_text[len(input_prompt):]
                response.append(generated_text)
                verbose and print(f"\nGenerate Outputs:\n{generated_text}")
            
            if len(response) > 1:
                return response
            return response[0]
        
        else: # None Batched Huggingface generation 
            if output_json_schema is not None:
                raise NotImplementedError("output_json_schema is not supported for non-vLLM (HuggingFace) backend.")

            # Make inputs, tokenize directly from the initial openai format to be safe?
            if input_prompt_is_messages:
                direct_input_prompt = tokenizer.apply_chat_template(
                    original_input_prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=use_reasoning,
                    return_tensors="pt",
                    return_dict=True
                )
            else:
                raise Exception("Not implemented for huggingface generation.")

            input_prompt = direct_input_prompt
            input_ids = input_prompt.input_ids.cuda()
            attention_mask = input_prompt.attention_mask.cuda()

            # Make generate args
            local_generate_kwargs = sampling_kwargs.copy()
            local_generate_kwargs["do_sample"] = True
            local_generate_kwargs["num_return_sequences"] = num_generations
            local_generate_kwargs["return_dict_in_generate"] = return_logits
            local_generate_kwargs["output_logits"] = return_logits  # Equal to logits, but after postprocessing
            if "max_tokens" in local_generate_kwargs:  # Translate API kwargs to HF kwargs
                local_generate_kwargs["max_new_tokens"] = local_generate_kwargs.pop("max_tokens")
                local_generate_kwargs["max_length"] = None  # override model's generation_config.max_length to avoid conflict

            # Generate
            output_ids = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **local_generate_kwargs
            )

            # Collect Results
            response = []
            response_tokens = []
            if return_logits:
                sequences = output_ids.sequences
            else:
                sequences = output_ids

            for i in range(sequences.shape[0]):
                if return_tokens:
                    response_tokens.append(sequences[i])
                response.append(
                    tokenizer.decode(
                        sequences[i][input_ids.shape[1] :],
                        skip_special_tokens=skip_special_tokens,
                        ignore_tokenization_space=True,
                    )
                )
            

            # Return
            if return_logits:
                logits = torch.stack(output_ids.logits, dim=1)
                return {"completions": response, "logits": logits}
            elif return_tokens:
                return {"completions": response, "full_tokens": response_tokens}
            if len(response) > 1:
                return response
            return response[0]
    