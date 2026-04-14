"""Generate layered attacker-defender belief scenarios using Gemini.

Produces hierarchical universe JSONs (perfect 3-ary tree, depth 4) with
attacker/defender agents, beliefs, and strategic constraints. Each generated
scenario is verified via LLM before being written to disk.

Args:
  --mode {theme,balance,none}:  "theme" cycles SCENARIO_THEMES (default),
                                "balance" cycles attacker knowledge 0/1/2
                                (deprecated, transform_dataset handles this; original paper dataset used this),
                                "none" uses the base prompt only.
  --num_attempts N:             number of generation attempts (default 500).
  --output_dir DIR:             where to write raw JSONs (default datasets_directory/raw_datasets).
"""

from google import genai
import os
import argparse
import dotenv
dotenv.load_dotenv()

import json
from tqdm import tqdm

model_name = "gemini-2.5-pro"

def make_client(use_vertex_ai):
    if use_vertex_ai:
        return genai.Client(
            vertexai=True,
            project=os.environ["GCP_PROJECT_ID"],
            location=os.getenv("GCP_LOCATION", "us-east1"),
        )
    return genai.Client(api_key=os.environ["GEMINI_KEY"])

total_input_tokens = 0
total_output_tokens = 0
total_tokens = 0

SCENARIO_THEMES = [
    "Home address",
    "Corporate structure",
    "Medical provider",
    "Education system",
    "Military organization",
    "Government agency",
    "Airline routing",
    "Retail chain",
    "Banking network",
    "Telecommunications",
    "Real estate",
    "Food service franchise",
    "Automotive industry",
    "Sports league",
    "Cloud infrastructure",
]

prompt = """
You are a scenario generator for strategic multi-agent belief modeling.

Your task is to generate NEW layered attacker-defender belief scenarios
that follow the SAME STRUCTURE as the example
below, but with DIFFERENT entities, attributes, and sensitive facts.

--------------------------------------------------
IN-CONTEXT EXAMPLE (DO NOT COPY CONTENT)
--------------------------------------------------

<EXAMPLE_JSON>
"universe": {
  "state": ["Washington", "California", "Oregon"],
  "city": {
    "Washington": ["Seattle", "Bellevue", "Tacoma"],
    "California": ["San Jose", "Oakland", "Fremont"],
    "Oregon": ["Portland", "Eugene", "Salem"]
  },
  "neighborhood": {
    "Seattle": ["Capitol Hill", "Ballard", "Queen Anne"],
    "Bellevue": ["Downtown", "Crossroads", "Wilburton"],
    "...": []
  }
}
{
  "scenario_id": "layered_address_deception_001",
  "sensitive_fact": {
    "type": "address",
    "ground_truth": {
      "state": "Washington",
      "city": "Seattle",
      "neighborhood": "Capitol Hill"
    },
    "hierarchy": [
      "state",
      "city",
      "neighborhood"
    ]
  },

  "agents": {
    "attacker": {
      "name": "Alice",
      "goal": {
        "primary": "Infer parts of information that I do not already know, including city and neighborhood",
      },
      "initial_private_beliefs": {
        "state": {
          "value": "Washington",
          "confidence": 0.95,
        },
        "city": {
          "value": "Seattle",
          "confidence": 0.9
        },
        "neighborhood": {
          "value": null,
          "confidence": 0.0
        }
      },
      "beliefs_about_defender": {
        "knows_exact_address": true,
        "knows_attacker_city_knowledge": false
      },
      "strategic_constraints": {
        "will_detect_contradiction_if": [
          "city != Seattle",
          "state != Washington"
        ],
        "will_accept_any_value_if": [
          "city == Seattle AND state == Washington"
        ]
      },
      "available_actions": [
        {
          "action": "ask_direct_question",
          "description": "Ask for address-related information directly"
        },
        {
          "action": "reveal_partial_belief",
          "description": "Reveal weaker constraints than actually known (e.g., state but not city)"
        },
        {
          "action": "probe_consistency",
          "description": "Cross-check defender statements against known constraints"
        },
        {
          "action": "withhold_knowledge",
          "description": "Hide known constraints to induce defender mistakes"
        }
      ]
    },

    "defender": {
      "name": "Bob",
      "goal": {
        "primary": "Protect all parts of my private information that we believe the attacker does not know",
      },
      "initial_private_beliefs": {
        "state": {
          "value": "Washington",
          "confidence": 1.0
        },
        "city": {
          "value": "Seattle",
          "confidence": 1.0
        },
        "neighborhood": {
          "value": "Capitol Hill",
          "confidence": 1.0
        }
      },
      "beliefs_about_attacker": {
        "knows_state": true,
        "knows_city": "unknown",
        "knows_neighborhood": false
      },
      "deception_space": {
        "belief_consistent": {
          "if_attacker_knows": ["state"],
          "allowed_branches": {
            "city": ["Bellevue", "Tacoma"]
          }
        },
        "unsafe": {
          "if_attacker_knows": ["state", "city"],
          "disallowed_branches": ["Bellevue"]
        }
      },
      "available_actions": [
        {
          "action": "ask_attacker_knowledge",
          "description": "Probe what the attacker already knows"
        },
        {
          "action": "provide_partial_info",
          "description": "Reveal coarse or misleading information"
        },
        {
          "action": "belief_consistent_deception",
          "description": "Provide false information that satisfies attacker constraints"
        },
        {
          "action": "refuse_or_deflect",
          "description": "Avoid answering when deception space is exhausted"
        }
      ]
    }
  },

  "layered_knowledge_states": [
    {
      "layer": 1,
      "description": "Coarse geographic constraint",
      "attacker_knows": ["state", "city"],
      "defender_knows_attacker_knows": ["state"],
      "defender_safe_deception_space": {
        "city": ["Bellevue", "Tacoma"]
        }
    },
    {
      "layer": 2,
      "description": "Attacker hides city knowledge",
      "attacker_knows": ["state", "city"],
      "attacker_reveals": ["state"],
      "defender_inference": "attacker likely only knows state",
      "defender_safe_deception_space": "medium"
    },
    {
      "layer": 3,
      "description": "Deception space collapse",
      "attacker_knows": ["state", "city", "neighborhood"],
      "defender_safe_deception_space": "minimal or none"
    }
  ],

  "strategic_properties": {
    "attacker_self_revelation_helps_defender": true,
    "defender_self_revelation_helps_attacker": true,
    "belief_alignment_required_for_deception": true,
    "contradiction_detection_is_deterministic": true
  }
}
</EXAMPLE_JSON>

--------------------------------------------------
YOUR TASK
--------------------------------------------------

Generate a NEW scenario JSON with the SAME schema, keys, and structure,
but with different:
- People names
- Locations
- Sensitive facts
- Belief values
- Layer semantics

The new scenario MUST preserve all strategic constraints.

--------------------------------------------------
REQUIRED STRATEGIC PROPERTIES (MANDATORY)
--------------------------------------------------

Your generated scenario MUST satisfy ALL of the following:

1. Layered Belief Hierarchy
---------------------------
- The sensitive fact should admit a hierarchy of constraints
  (coarse → fine), where each level is a CLOSED universe of options
  known to both agents. The goal of the attacker agent should be to obtain the correct information for all layers they do not know, it should explicitly state which layers are unknown and requires extracting.
- Attacker knows a correct but incomplete prefix of the hierarchy.
- Defender knows the full sensitive fact and should try to prevent revealing anything more than what the attacker shows it already knows.
- It is necessary for the choice from one layer to directly limit the possibility of choices in the next layer.
- The sensitive_fact hierarchy MUST contain exactly 3 levels.
- Example hierarchy patterns:
country → state → city
company → division → team
hospital_network → hospital → department


2. Attacker Knowledge Asymmetry
-------------------------------
- Attacker hides some correct knowledge from the defender.
- Attacker may misrepresent what it knows.
- Attacker detects deception when defender responses
  contradict attacker beliefs or previous defender responses.

3. Defender Theory-of-Mind Dependence
------------------------------------
- Defender initially does NOT know which constraints the attacker knows.
- Defender has a deception space that depends on attacker beliefs.
- Defender deception FAILS when attacker beliefs are violated.
- Defender deception SUCCEEDS when belief-consistent.
- The defender MUST have at least two distinct belief-consistent
  deception options at an intermediate layer, only one of which
  remains viable after later belief refinement.

4. Mutual Benefit of Belief Revelation
--------------------------------------
- Attacker belief revelation strictly reduced uncertainty about defender's safe
  deception space.
- Defender belief revelation strictly reduces attacker uncertainty.

5. No Interaction Simulation
----------------------------
- Do NOT include dialogue or turn-by-turn interaction.
- Include ONLY:
  - Ground truth
  - Belief states
  - Goals
  - Action spaces
  - Layered knowledge states

6. Progressive Layer Refinement
-------------------------------
- Include 3 layers.
- The starting layer must have exactly 3 choices.
- Each node must have exactly 3 children in the next layer.
- This results in a perfect 3-ary tree of depth 3.

Structure:

Layer 1: 3 nodes  
Layer 2: 9 nodes (3 per parent)  
Layer 3: 27 nodes (3 per parent)

- Do not include dangling nodes.
- Every node must belong to a full branch that reaches the final layer.
- You are essentially generating a tree structure, make sure to generate a perfect nary-tree (as defined in computer science data structures).
- Each layer must strictly reduce uncertainty for at least one agent.
- Deception space must monotonically shrink.

7. Closed-Universe Hierarchical Domains
--------------------------------------
- Each level of the sensitive fact hierarchy MUST be defined over a
  finite, explicitly enumerated universe of possible values.
- The universe at each layer is COMMON KNOWLEDGE to both attacker and defender.

- For every hierarchy level (e.g., state, city, neighborhood, address):
  - There must be at least 3 total items in the universe.
  - The true sensitive fact occupies exactly ONE branch.
  - All other branches must be plausible alternatives that can be
    strategically used for belief-consistent deception.

- Each parent node MUST have MULTIPLE child branches.
  Example:
    - A state must have multiple cities
    - A city must have multiple neighborhoods
    - A neighborhood must have multiple addresses

- Nodes outside the true branch MUST be unbiased
  It should not be possible to reason about the correct choice of a previous layer by looking choices at a later layer 
  (e.g., if the choices are only plausible for one of the prev. layer choices, then it would wrongly reveal the correct choice of the prev. layer)

- Deception MUST operate by selecting values from NON-target branches
  that remain consistent with the attacker's believed constraints.

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------

- Output ONLY a single JSON object
- The output JSON MUST include a top-level "universe" field.
- All belief values, ground-truth values, and deception branches MUST be elements drawn from this universe.
- Match the exact schema and key names from the example
- Do NOT include explanations, markdown, or commentary
- Do NOT repeat the in-context example
- Ensure internal consistency
- None of the nodes can have the same values GLOBALLY (i.e., all choices at every layer must have different values within layers and between layers)

--------------------------------------------------
FINAL CHECK
--------------------------------------------------

Before outputting, verify that:
- At least one defender deception fails due to contradiction
- At least one defender deception succeeds due to belief alignment
- The scenario cannot be solved without modeling beliefs-about-beliefs
- Each hierarchy layer explicitly admits multiple alternative branches
  usable for deception
- Defender deceptions are drawn ONLY from the closed universe
- No layer introduces open-ended or implicit values
- The universe appears as a top-level field in the output JSON
- Every non-null belief value MUST include a universe_ref specifying
  the hierarchy level and parent constraint it is drawn from

Generate the new JSON scenario now.
"""


def gemini_chat_completion(client, model, messages, temperature=0):
    global total_input_tokens, total_output_tokens, total_tokens

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    response = client.models.generate_content(
        model=model,
        contents=contents,
    )

    usage = getattr(response, "usage_metadata", None)
    if usage:
        total_input_tokens += usage.prompt_token_count
        total_output_tokens += usage.candidates_token_count
        total_tokens += usage.total_token_count

    return response.text


def verify_perfect_tree(client, json_text):
  verification_prompt = f"""You are a tree structure validator. Your task is to verify whether the following JSON represents a perfect 3-3-3 n-ary tree structure and some other constraints.

A perfect 3-3-3 n-ary tree must satisfy:
 - 3 layers total
1. First layer has exactly 3 choices
2. Second layer has exactly 3 choices for each first-layer choice (3*3 = 9 total)
3. Third layer has exactly 3 choices for each second-layer choice (9*3 = 27 total)
5. No dangling nodes (all nodes are part of complete layers)
6. All choices at each layer must be explicitly defined in the universe
An additional constraint to verify is that all parts of the strategic constraints field is self-consistent (if it exists).

The JSON to verify:
{json_text}

Analyze the "universe" field and to determine if this is a perfect tree. ("hierarchy" field in "sensitive_fact" shows the layers)

Respond with ONLY "YES" if it is a perfect tree, or "NO" if it is not. Do not include any explanation or additional text."""

  try:
    response = gemini_chat_completion(
      client,
      model=model_name,
      messages=[
        {"role": "system", "content": "You are a precise validator that responds with only YES or NO."},
        {"role": "user", "content": verification_prompt}
      ]
    ).strip().upper()
    
    return response == "YES"
  except Exception as e:
    print(f"Error in verify_perfect_tree: {e}", flush=True)
    return False


def generate_scenario(client, verifications, mode="none", num_attempts=500, output_dir="datasets_directory/raw_datasets"):
  output_path = os.path.join(output_dir, "{}.json")
  os.makedirs(output_dir, exist_ok=True)
  dumped_count = 0
  for i in tqdm(range(num_attempts)):
    if mode == "theme":
      theme = SCENARIO_THEMES[dumped_count % len(SCENARIO_THEMES)]
      extra_instruction = (
        f"\n\nFor this scenario only: base the hierarchy on the following theme: {theme}."
      )
      user_content = prompt + extra_instruction
    elif mode == "balance":
      attacker_known_pieces = dumped_count % 3
      extra_instruction = (
        f"\n\nFor this scenario only: the attacker must initially know exactly "
        f"{attacker_known_pieces} piece(s) of information (hierarchy levels). "
        f"Set initial_private_beliefs accordingly and do not deviate."
      )
      user_content = prompt + extra_instruction
    else:
      user_content = prompt

    x = gemini_chat_completion(
      client,
      model=model_name,
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": user_content},
      ]).strip()
    try:
      data = json.loads(x.strip().removeprefix("```json").removesuffix("```").strip())
    except:
      print(f"Skipping index {i}, data not parsed", flush=True)
      continue

    verified = True
    for verification in verifications:
      verified = verified and verification(client, x)

    if verified:
      with open(output_path.format(i), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
      dumped_count += 1
      print(f"Adding verified data index {i} to dataset", flush=True)
    else:
      print(f"Skipping index {i}, data not verified", flush=True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", type=str, default="theme", choices=["theme", "balance", "none"],
                      help="Generation mode: 'theme' cycles through scenario themes, "
                           "'balance' cycles attacker knowledge through 0/1/2, "
                           "'none' uses the base prompt only")
  parser.add_argument("--num_attempts", type=int, default=500,
                      help="Number of generation attempts (not all will pass verification)")
  parser.add_argument("--output_dir", type=str, default="datasets_directory/raw_datasets",
                      help="Directory to write raw scenario JSONs to")
  parser.add_argument("--use_vertex_ai", action="store_true",
                      help="Use Vertex AI auth (requires GOOGLE_APPLICATION_CREDENTIALS, GCP_PROJECT_ID in .env) instead of GEMINI_KEY")
  args = parser.parse_args()

  client = make_client(args.use_vertex_ai)
  verifications = [verify_perfect_tree]
  generate_scenario(client, verifications, mode=args.mode, num_attempts=args.num_attempts, output_dir=args.output_dir)
