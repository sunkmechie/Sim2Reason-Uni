import re, copy, time, json, math
import os, pathlib, ipdb
import numpy as np
import requests

from typing import List, Dict, Union

from io import BytesIO
from PIL import Image

from collections import defaultdict
from sympy import sympify, Symbol

from sympy.parsing.latex import parse_latex

from llm.utils import math_utils
from llm.utils.math_utils import extract_answer
from llm.utils.unicode_to_latex import my_unicode_to_latex

from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, LatexNormalizationConfig

import multiprocessing

from functools import wraps

class TimeoutException(Exception):
    pass

def timeout(seconds=2, error_message="Function timed out"):
    import signal
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def _handle_timeout(signum, frame):
                raise TimeoutException(error_message)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Always disable the alarm
        return wrapper
    return decorator

def count_tokens_from_msg(messages: List[Dict[str, Union[str, list]]], model: str = "gpt-3.5-turbo") -> int:
    """Count tokens for special message structure."""
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    tokens_per_message = 3
    num_tokens = 0
    
    for message in messages:
        # Base message tokens
        num_tokens += tokens_per_message
        
        # Count role
        num_tokens += len(encoding.encode(message['role']))
        
        # Handle content based on structure
        content = message['content']
        if content:  # Non-empty list
            text = content[0]['text']
            num_tokens += len(encoding.encode(text))
    
    # Add completion format overhead
    num_tokens += 3
    
    return num_tokens

class DummyResponse:
    content = "Failed to do API call."
    tool_calls = None

# PARENT = pathlib.Path(__file__).parent.parent.relative_to(pathlib.Path.cwd())
PARENT = pathlib.Path(__file__).parent.parent.resolve()
UTILS_DIR = pathlib.Path(__file__).parent
st = ipdb.set_trace

cost_dict = json.load(open(os.path.join(UTILS_DIR, "cost.json"), 'r'))
openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

try:
    openrouter_headers = {
        "Authorization": f"Bearer {os.environ['OR_TOKEN']}",
        "Content-Type": "application/json"
    }
except:
    openrouter_headers = {
        "Authorization": f"Bearer <insert>",
        "Content-Type": "application/json"
    }

ltx_cfg = LatexExtractionConfig(
    boxed_match_priority=0,
    normalization_config=LatexNormalizationConfig(
        units=True,        
    ),
)
expr_cfg = ExprExtractionConfig()
math_verify_extraction_config = [
    ltx_cfg,
    expr_cfg,
]

enclose = lambda x: "\\boxed{" + x + "}"

import importlib

def _get_class_constructor(class_name: str):
    """
    Dynamically imports a class from a string path.
    e.g., "my_module.my_submodule.MyClass" -> <class 'my_module.my_submodule.MyClass'>
    """
    try:
        module_path, class_name = class_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Could not import class '{class_name}' from module '{module_path}': {e}")

# from run_gpt import LLM
def do_openrouter_req(model, messages, include_reasoning=False):
    payload = {
        "model": model,
        "messages": messages,
        "include_reasoning": include_reasoning
    }
    return requests.post(openrouter_url, headers=openrouter_headers, data=json.dumps(payload))

def get_costs(id_gen):
    response = requests.get(f'https://openrouter.ai/api/v1/generation?id={id_gen}', headers=openrouter_headers)
    stats = response.json()
    return stats

def call_gemini(prompts):
    from google import genai
    API_KEY=""

    client = genai.Client(api_key=API_KEY)
    responses = []

    total_input_tokens, total_output_tokens = 0, 0

    for prompt in prompts:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt[-1]["content"]
        )

        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
        else:
            input_tokens = output_tokens = 0

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        responses.append(response.text)

    total_cost = (total_input_tokens / 1e6) * 0.3 + \
                     (total_output_tokens / 1e6) * 2.5

    return responses, total_cost

def openai_call(model_name, prompt_ex, stats, client, openrouter=False, n=1):
    error_generating =  0
    if "qwen3-235b" in model_name:
        extra_body = {
            "provider": {
                "order": ["together"],
                "allow_fallbacks": False
            }
        }

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages= prompt_ex,
            n=n,
        )    
    except json.decoder.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        response_text = ''
        
        error_generating = 1
        return [response_text], stats, error_generating
    except Exception as e:
        print(f"Error: {e}")
        response_text = ''
        error_generating = 2
        return [response_text], stats, error_generating
    # st()
    response_texts = []
    for choice_n in range(len(response.choices)):
        try:
            response_text = response.choices[choice_n].message.content
            if response_text is None:
                response_text = ''
        except Exception as e:
            print(f"Error: {e}")
            response_text = ''
        response_texts.append(response_text)
        stats['tokens/valid_end'].append(response.choices[choice_n].finish_reason == "stop")
    stats['tokens/input'].append(response.usage.prompt_tokens)
    stats['tokens/output'].append(response.usage.completion_tokens)
    print("completion tokens: ", response.usage.completion_tokens)
    # st()
    try:
        if response.usage.prompt_tokens_details is not None and  hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
            stats['tokens/cached'].append(response.usage.prompt_tokens_details.cached_tokens)
        else:
            stats['tokens/cached'].append(0)
    except Exception as e:
        print(f"Error: {e}")
        stats['tokens/cached'].append(0)

    if openrouter:
        retries = 0
        max_retries = 3
        while True:
            cost_structure = get_costs(response.id)
            if isinstance(cost_structure, dict) and 'error' in cost_structure:
                if retries >= max_retries:
                    cost_structure = {'data': {'total_cost': 0}}
                    break
                time.sleep(1)
                retries += 1
                continue
            break
        
        stats['cost'].append(cost_structure['data']['total_cost'])
    else:
        if stats['tokens/cached'][-1] is not None and stats['tokens/cached'][-1] > 0:
            stats['cost'].append((cost_dict[model_name]['input'] * response.usage.prompt_tokens / 1e6) + (cost_dict[model_name]['output'] * response.usage.completion_tokens / 1e6 )+ (cost_dict[model_name]['cached'] * stats['tokens/cached'][-1] / 1e6))
        else:
            stats['cost'].append((cost_dict[model_name]['input'] * response.usage.prompt_tokens / 1e6) + (cost_dict[model_name]['output'] * response.usage.completion_tokens / 1e6 ))

    
    return response_texts, stats, error_generating

def call_llm_api(cfg, model_name, llm_prompt, cost_count=False, n=1):
    import httpx
    from openai import OpenAI, AsyncOpenAI, AzureOpenAI

    if cfg.use_openrouter:
        model_name = model_name.lower()
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ['OR_TOKEN'],
            )        
        all_response_text = []
        all_error_generating = []
        stats = defaultdict(list)
        # st()
        for prompt_ex in llm_prompt:
            if model_name == "openai/gpt-oss-120b":
                prompt_ex = np.concatenate([[{"role": "system", "content": "Reasoning: high"}], prompt_ex])
            response_text, stats, error_generating = openai_call(model_name, prompt_ex, stats, client, openrouter=True)
            # st()
            all_response_text.append(response_text)
            all_error_generating.append(error_generating)
    elif cfg.use_azure:        
        # Create a custom http client to avoid proxy issues
        http_client = httpx.Client()
        
        client = AzureOpenAI(
            api_version="2025-01-01-preview",
            api_key="",
            azure_endpoint='',
            http_client=http_client
        )
        all_response_text = []
        all_error_generating = []
        stats = defaultdict(list)
        for prompt_ex in llm_prompt:
            response_text, stats, error_generating = openai_call(model_name, prompt_ex, stats, client)
            all_response_text.append(response_text)
            all_error_generating.append(error_generating)
    else:
        
        client = OpenAI(
            organization=os.getenv('OPENAI_ORG_ID'),
            api_key=os.getenv('PHO_OPENAI_API_KEY')
        )
        
        all_response_text = []
        all_error_generating = []
        stats = defaultdict(list)
        for prompt_ex in llm_prompt:
            response_text, stats, error_generating = openai_call(model_name, prompt_ex, stats, client, n=n)
            all_response_text.append(response_text)
            all_error_generating.append(error_generating)
    return all_response_text, stats, all_error_generating
    
def call_llm(cfg, model_name, llm_prompt):
    from openai import OpenAI, AsyncOpenAI, AzureOpenAI
    if cfg.use_openrouter:
        client = OpenRouterAPI(model_name)
        response = client.send_request(llm_prompt)
        response_text = response['choices'][0]['message']['content']
    else:
        client = OpenAI(
            organization=os.getenv('OPENAI_ORG_ID'),
            api_key=os.getenv('PHO_OPENAI_API_KEY')
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=llm_prompt)            
        response_text = response.choices[0].message.content
    return response_text

def openai_call_with_tools(model_name, prompt_ex, stats, client, openrouter=False, n=1, tools=None, tool_choice="none"):
    """
    Makes a call to the OpenAI or compatible API, with added support for tools.
    """
    try:
        if tool_choice != "none":
            response = client.chat.completions.create(
                model=model_name,
                messages=prompt_ex,
                n=n,
                tools=tools,
                tool_choice=tool_choice,
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=prompt_ex,
                n=n,
            )
    except Exception as e:
        print(f"An error occurred during the API call: {e}")
        # Return a structure that mimics a failed response to avoid crashing the calling code
        
        stats["tokens/input"].append(0)
        stats["tokens/output"].append(0)
        stats["tokens/cached"].append(0)
        stats["cost"].append(0)
        return [DummyResponse], stats

    response_messages = [choice.message for choice in response.choices]

    stats['tokens/input'].append(response.usage.prompt_tokens)
    stats['tokens/output'].append(response.usage.completion_tokens)
    if hasattr(response.usage, 'cached_tokens') and response.usage.cached_tokens is not None:
        stats['tokens/cached'].append(response.usage.cached_tokens)
    else:
        stats['tokens/cached'].append(0)

    if openrouter:
        retries = 0
        max_retries = 3
        while retries < max_retries:
            cost_structure = get_costs(response.id)
            if isinstance(cost_structure, dict) and 'error' in cost_structure:
                time.sleep(1)
                retries += 1
            else:
                stats['cost'].append(cost_structure['data']['total_cost'])
                break
        if retries == max_retries:
            stats['cost'].append(0) # Default cost if fetching fails
    else:
        input_cost = cost_dict.get(model_name, {}).get('input', 0)
        output_cost = cost_dict.get(model_name, {}).get('output', 0)
        cached_cost = cost_dict.get(model_name, {}).get('cached', 0)
        
        cost = (input_cost * response.usage.prompt_tokens / 1e6) + \
               (output_cost * response.usage.completion_tokens / 1e6)
        if stats['tokens/cached'] and stats['tokens/cached'][-1] > 0:
            cost += (cached_cost * stats['tokens/cached'][-1] / 1e6)
        stats['cost'].append(cost)

    return response_messages, stats

def call_llm_api_with_tools(cfg, model_name, llm_prompt, cost_count=False, tool_cfg=None, n=1):
    """
    Calls the LLM API with support for tool calling, handling the conversation flow.
    """
    import asyncio, httpx
    from openai import OpenAI, AsyncOpenAI, AzureOpenAI
    from verl.tools.schemas import OpenAIFunctionToolSchema
    if cfg.use_openrouter:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get('OR_TOKEN'),
        )
    elif cfg.use_azure:
        http_client = httpx.Client()
        client = AzureOpenAI(
            api_version="2025-01-01-preview",
            api_key=os.environ.get('AZURE_API_KEY'),
            azure_endpoint=os.environ.get('AZURE_ENDPOINT'),
            http_client=http_client
        )
    else:
        client = OpenAI(
            organization=os.getenv('OPENAI_ORG_ID'),
            api_key=os.getenv('PHO_OPENAI_API_KEY')
        )

    # --- Tool Initialization ---
    # We will instantiate all tools once and store them in a dictionary
    # mapping the function name to the tool instance for easy lookup.
    tools = {}
    tool_schema = None
    if tool_cfg and "tools" in tool_cfg:
        tool_schema = [tool["tool_schema"] for tool in tool_cfg["tools"]]
        for tool_config in tool_cfg["tools"]:
            try:
                class_constructor = _get_class_constructor(tool_config["class_name"])
                instance = class_constructor(config=tool_config["config"], tool_schema=OpenAIFunctionToolSchema.model_validate(tool_config["tool_schema"]))
                function_name = tool_config["tool_schema"]["function"]["name"]
                tools[function_name] = instance
            except Exception as e:
                print(f"Error instantiating tool {tool_config.get('class_name', 'N/A')}: {e}")
    # --- End Tool Initialization ---

    all_final_responses = []
    stats = defaultdict(list)
    max_tool_calls = 3

    for prompt_ex in llm_prompt:
        conversation_history = prompt_ex.copy()
        
        for _ in range(max_tool_calls):
            response_messages, stats = openai_call_with_tools(
                model_name,
                conversation_history,
                stats,
                client,
                openrouter=cfg.use_openrouter,
                n=n,
                tools=tool_schema,
                tool_choice="auto"
            )

            response_message = response_messages[0]

            if response_message.tool_calls:
                conversation_history.append(response_message)
                
                for tool_call in response_message.tool_calls:
                    # =================================================================
                    # START: Tool Execution Logic
                    # =================================================================
                    function_name = tool_call.function.name
                    
                    if function_name in tools:
                        try:
                            tool_instance = tools[function_name]
                            function_args = json.loads(tool_call.function.arguments)

                            if function_name == "code_interpreter":
                                # Add print to last statment if not present
                                code = function_args.get("code", "")
                                lines = code.split("\n")[::-1]
                                for _ in range(len(lines)):
                                    line = lines[_]
                                    line = line.strip()
                                    if line and not line.startswith("#"):
                                        if not line.startswith("print("):
                                            code += f"\nprint({line})"
                                        break    
                                function_args["code"] = code
                            
                            # Execute the tool method
                            # The return is (ToolResponse, float, dict), we need the first element.
                            # Define a small async wrapper to call the method
                            async def run_tool():
                                return await tool_instance.execute(
                                    instance_id=tool_call.id,
                                    parameters=function_args
                                )

                            tool_response_obj, _, _ = asyncio.run(run_tool())
                            
                            # Serialize the response object to a string (JSON is a good choice)
                            # to pass back to the LLM. This assumes the ToolResponse object
                            # can be represented as a string or has a .content or .to_dict() method.
                            # if hasattr(tool_response_obj, 'to_dict'):
                            #     function_response = json.dumps(tool_response_obj.to_dict())
                            # elif hasattr(tool_response_obj, 'content'):
                            #     function_response = str(tool_response_obj.content)
                            # else:
                            #     function_response = str(tool_response_obj)
                            function_response = tool_response_obj.text

                        except Exception as e:
                            print(f"Error executing tool {function_name}: {e}")
                            function_response = f"Error: An exception occurred while executing the tool '{function_name}'."
                    else:
                        function_response = f"Error: Tool with name '{function_name}' not found or configured."
                    # =================================================================
                    # END: Tool Execution Logic
                    # =================================================================

                    conversation_history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })
            else:
                all_final_responses.append([response_message.content])
                break # No more tool calls, so we exit the loop
        else:
            conversation_history += [
                {
                    "role": "user",
                    "content": "Please provide the final answer in \boxed{} without using any more tools.",
                }
            ]
            final_response_messages, stats = openai_call_with_tools(
                model_name,
                conversation_history,
                stats,
                client,
                openrouter=cfg.use_openrouter,
                n=n, 
            )
            all_final_responses.append([msg.content for msg in final_response_messages])

    print(f"{len(all_final_responses)} responses collected from LLM.")

    return all_final_responses, stats

def reformat_llm_prompt(llm_prompt):
    for prompt_item in llm_prompt:
        prompt_item['content'] = prompt_item['content'][0]['text']
    return llm_prompt

def call_model(llm, sampling_params, llm_prompt, cfg=None):
    import vllm, torch
    # conversations = [messages for _ in range(cfg.num_generations_per_problem)]
    outputs = llm.chat(messages=llm_prompt, sampling_params=sampling_params, use_tqdm=True)
    stats = defaultdict(list)
    for output in outputs:
        stats['tokens/input'].append(len(output.prompt_token_ids))
        for response in output.outputs:
            stats['tokens/output'].append(len(response.token_ids))
            valid_end = response.finish_reason == "stop"
            stats['tokens/valid_end'].append(valid_end)
    return outputs, stats

async def async_openai_call(client, model_name: str, messages: List[Dict]):
    """
    Make an asynchronous API call to OpenAI.
    
    Args:
        client: AsyncOpenAI client instance
        model_name: Name of the model to use
        messages: List of message dictionaries for the conversation
        
    Returns:
        Response from the API call
    """
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        
        return response
    except Exception as e:
        print(f"Error making OpenAI API call: {str(e)}")
        raise

async def batch_openai_calls(model_name: str, llm_prompt: List[List[Dict]]):
    """
    Make multiple async API calls to OpenAI in parallel.
    
    Args:
        model_name: Name of the model to use
        llm_prompt: List of message lists, where each inner list contains message dicts
        
    Returns:
        List of responses from the API calls
    """
    import asyncio
    from openai import OpenAI, AsyncOpenAI, AzureOpenAI

    async with AsyncOpenAI(
        organization=os.environ["OPENAI_ORG_ID"], 
        api_key=os.environ["PHO_OPENAI_API_KEY"]
    ) as client:  # ✅ Use `async with` to properly close the client

        tasks = [
            async_openai_call(client, model_name, messages) 
            for messages in llm_prompt
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return responses

def llm_check(llm_prompt, cfg, **kwargs):
    # st()
    if cfg.check_locally:
        model_response = call_model(kwargs['llm'], kwargs['sampling_params'], llm_prompt)
        response_text = model_response[0].outputs[0].text
    else:
        response_text = call_llm(cfg, cfg.verify_model_name, llm_prompt)
        
    generated_answer = extract_answer(response_text, 'physics')
    
    
    print("Response Text Answer: ", generated_answer)
    
    # Check if the response contains a confirmation of equivalence
    return "yes" in generated_answer.lower(), response_text

def check_value_with_units(gpt_answer, expected_answer, tolerance=0.01):
    """Check numeric values with units for equality."""
    
    gpt_value, gpt_unit = parse_value_with_units(gpt_answer)
    expected_value, expected_unit = parse_value_with_units(expected_answer)
    
    if gpt_value is None or expected_value is None:
        return False

    if gpt_unit == expected_unit and abs(gpt_value - expected_value) <= tolerance:
        return True
    
    return False

def solve_and_simplify(eq_str):
    """Solve an equation for a common variable and simplify."""
    transformations = (standard_transformations + (implicit_multiplication_application,))
    try:
        left, right = eq_str.split('=')
        expr_left = parse_expr(left, transformations=transformations)
        expr_right = parse_expr(right, transformations=transformations)
        equation = Eq(expr_left, expr_right)
        simplified_forms = [simplify(equation)]
        for var in equation.free_symbols:
            solved = solve(equation, var)
            if solved:
                simplified_forms.append(simplify(Eq(var, solved[0])))
        return simplified_forms
    except Exception as e:
        print(f"Error parsing equation '{eq_str}': {e}")
        return []

def check_equations(gpt_equation, expected_equation):
    if '=' not in gpt_equation or '=' not in expected_equation:
        return False
    """Check if two equations are equivalent."""
    gpt_simplified = solve_and_simplify(gpt_equation)
    expected_simplified = solve_and_simplify(expected_equation)
    return any(g == e for g in gpt_simplified for e in expected_simplified)

def check_float_answer(gpt_answer, expected_answer, precision=1, mode = 'legacy', max_relative_error = 1e-2, relative_to_absolute_threshold = 1e-2) -> bool:
    """Check if two floating point numbers are equivalent."""
    if mode == 'legacy':
        return bool(round(abs(round(gpt_answer, precision) - round(expected_answer, precision)), precision) <= 10**(-precision))
    else:
        # This finds relative error, and returns true if error < 5%.
        cond = (abs(expected_answer) <= relative_to_absolute_threshold or abs(gpt_answer) <= relative_to_absolute_threshold)
        if relative_to_absolute_threshold == 0.0: 
            cond = (abs(expected_answer) <= relative_to_absolute_threshold and abs(gpt_answer) <= relative_to_absolute_threshold)
        if cond:
            return bool(abs(abs(gpt_answer) - abs(expected_answer)) <= max_relative_error)
        else:
            return bool(abs(abs(gpt_answer) - abs(expected_answer)) / abs(expected_answer) <= max_relative_error)

def check_answer_heuristic(gpt_answer, expected_answer, precision=1, mode='relative', threshold = 1e-2) -> tuple[bool, str]:
    """Check answers based on their type."""
    # print("Model answer:", gpt_answer, "Expected answer:", expected_answer)
    # st()
    try:
        if check_float_answer(float(gpt_answer), float(expected_answer), precision, mode, relative_to_absolute_threshold= threshold):
            return True, 'Float Answer Check'
    except:
        pass
        
    # Check if answers are numerical values with units
    if check_value_with_units(gpt_answer, expected_answer):
        return True, 'Heuristic Value Check'

    if math_utils.is_equiv(gpt_answer, expected_answer, remove_unit_bool=True):
        return True, 'Math Utils Heuristic Value Check'

    # # Check if answers are equations
    if check_equations(gpt_answer, expected_answer):
        return True, 'Heuristic Equation Check'
    
    return False, 'Heuristic Check Failed'

def check_answer_gpt(gpt_answer, expected_answer, cfg, **kwargs):
    instruction = f"Predicted Answer: {gpt_answer}\n\nExpected Answer: {expected_answer}"    
    verify_intro = open(f'{PARENT}/prompts/verify.txt', 'r').read()
    
    llm_prompt = construct_prompt(
        image=None,  # No image input, set to None
        intro=verify_intro, 
        instruction=instruction,
        message=[]  # Empty initial message list
    )


    # For unstructured answers or if previous checks failed
    return llm_check(llm_prompt, cfg, **kwargs)

def pil_to_b64(img: Image.Image) -> str:
    import base64
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64

def construct_prompt(image=None, intro=None, examples=None, instruction=None, message=None):
    if message is None:
        message = []
    if intro is not None:
        message.extend([
            {
                "role": "system", 
                "content": [{"type": "text", "text": intro}],
            }
        ])
    
    if examples is not None:
        for example in examples:
            message.extend([
                {
                    "role": example[1], 
                    "content": [{"type": "text", "text": example[0]}],
                }
            ])
        
    content = []
    if image is not None:
        if (isinstance(image, list) and len(image) > 0 and
            isinstance(image[0], (list, tuple)) and len(image[0]) == 2):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": "**Reference Figures**:\n"
                    },
                ]
            )
            for (img_ind, img) in image:
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": f"Reference Figure {img_ind}."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(img), "detail": "high"},
                        }, 
                    ]
                )
        elif isinstance(image, Image.Image):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": "**Reference Figure **:\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(image), "detail": "high"},
                    }, 
                ]
            )
        else:
            raise NotImplementedError

    if instruction is not None:
        current_prompt = f"{instruction}"
        content = [{"type": "text", "text": current_prompt}] + content
        message.append({"role": "user", "content": content})
    
    return message

class OpenRouterAPI:
    def __init__(self, model_version):
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = "<insert key>" 
        self.model_version = model_version

    def send_request(self, llm_prompt):
        """Sends the LLM prompt to OpenRouter API and returns the response."""
        print(f'Sending request to: {self.openrouter_url}')
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_version,
            "messages": llm_prompt
        }

        try:
            response = requests.post(
                url=self.openrouter_url,
                headers=headers,
                data=json.dumps(payload)
            )

            if response.status_code == 200:
                print("Request was successful!")
                return response.json() 
            else:
                print(f"Failed with status code: {response.status_code}")
                print("Response:", response.text)
                return None
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None

def get_equations_from_response(response):
    equations_in_dollars = re.findall(r'\$\$(.*?)\$\$', response, re.DOTALL)    # Find all equations in $$ format
    parsed_equations = []
    
    if equations_in_dollars:
        for eq in equations_in_dollars:
            clean_eq = eq.replace("\n", " ").strip()    # Remove any newline characters
            clean_eq = clean_eq.replace("\\approx", "=")    # Replace \approx with = for sympy parsing
            clean_eq = re.sub(r'(\\,|\\quad).*$', '', clean_eq)    # Remove content after `\quad` or `\,`
            parsed_equations.append(parse_latex(clean_eq))
    else:
        print("No equations found in $$ format")
    print("Parsed equations: ", len(parsed_equations))

    return parsed_equations

def parse_example_text(example):
    # Parse example file into user and assistant text
    example_texts = []
    current_role = None
    current_text = []
    
    for line in example.split('\n'):
        if line.startswith('[USER]'):
            if current_role == 'assistant' and current_text:
                example_texts.append(('\n'.join(current_text), 'assistant'))
                current_text = []
            current_role = 'user'
        elif line.startswith('[ASSISTANT]'):
            if current_role == 'user' and current_text:
                example_texts.append(('\n'.join(current_text), 'user'))
                current_text = []
            current_role = 'assistant'
        elif line.strip() and current_role:
            current_text.append(line)
    
    if current_text:
        example_texts.append(('\n'.join(current_text), current_role))
    return example_texts

def parse_value_with_units(answer):
    """Parse numeric value and units from a string."""
    match = re.fullmatch(r"([+-]?\d*\.?\d+)\s*([a-zA-Z²/]+)", answer.strip())  
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        return value, unit
    return None, None

def extract_final_symbolic_answer(llm_response: str, not_equation = False):
    """
    Extracts the final symbolic equation from an LLM response.
    
    Parameters:
        llm_response (str): The full LLM response containing LaTeX equations.
        not_equation (bool): If True, the function will allow the answer to be a non-equation.
    Returns:
        sympy expression: Parsed symbolic equation.
    """
    # Step 1: Locate boxed equation or last displayed equation
    # boxed_match = re.findall(r'\boxed{(.+?)}', llm_response, re.DOTALL)
    boxed_match = re.findall(r'\\boxed{(.+)}', llm_response, re.DOTALL) # \\ instead of \. We remove '?' because it may cause problems while dealing with equations like \\boxed{\\frac{1}{2} m_6 g^2 \\sin^2(\\theta_6) t^2}.
    if boxed_match:
        equation_text = boxed_match[-1]  # Get the last boxed 

        vanilla_match = None
        opening = equation_text.find('{')
        closing = equation_text.find('}')
        if closing != -1 and (closing < opening or opening == -1):
            vanilla_match = equation_text[:closing]
        
        displayed_match = re.findall(r'\\\[(.*?)\\\]', equation_text, re.DOTALL)
        if not displayed_match:
            displayed_match = re.findall(r'\$\$(.*?)\$\$', equation_text, re.DOTALL) # support for $$ ... $$
        inline_match = re.findall(r'\\\((.*?)\\\)', equation_text, re.DOTALL)
        if not inline_match:
            inline_match = re.findall(r'\$(.*?)\$', equation_text, re.DOTALL) # support for $ ... $
            
        if vanilla_match:
            equation_text = vanilla_match
        if displayed_match:
            equation_text = displayed_match[-1]
        elif inline_match:
            equation_text = inline_match[-1]
    else:
        # If no \boxed{} found, get the last displayed equation
        displayed_match = re.findall(r'\\\[(.*?)\\\]', llm_response, re.DOTALL)
        if not displayed_match:
            displayed_match = re.findall(r'\$\$(.*?)\$\$', llm_response, re.DOTALL) # support for $$ ... $$
        inline_match = re.findall(r'\\\((.*?)\\\)', llm_response, re.DOTALL)
        if not inline_match:
            inline_match = re.findall(r'\$(.*?)\$', llm_response, re.DOTALL) # support for $ ... $
        
        if displayed_match:
            equation_text = displayed_match[-1]
        elif inline_match:
            equation_text = inline_match[-1]
        else:
            raise ValueError("No recognizable equation found in response.")

    # Step 2: Convert LaTeX syntax to a valid symbolic format

    # Convert \frac{A}{B} properly
    def frac_replace(match):
        num = match.group(1).strip()
        den = match.group(2).strip()
        return f"({num}) / ({den})"

    equation_text = re.sub(r'\\frac{([^{}]+)}{([^{}]+)}', frac_replace, equation_text)

    # Replace multiplication symbols
    equation_text = equation_text.replace('\\cdot', '*')
    equation_text = equation_text.replace('\\times', '*')

    # Convert square roots
    equation_text = re.sub(r'\\sqrt{([^{}]+)}', r'(\1)**(1/2)', equation_text)

    # Remove LaTeX formatting elements that are not needed
    equation_text = re.sub(r'\\left|\\right', '', equation_text)  # Remove LaTeX grouping
    equation_text = re.sub(r'\\text{.*?}', '', equation_text)  # Remove \text{} blocks

    # Ensure implicit multiplication is explicitly represented
    # List of known LaTeX math function names (expandable)
    math_functions = ['sin', 'cos', 'tan', 'csc', 'sec', 'cot',
                    'arcsin', 'arccos', 'arctan',
                    'sinh', 'cosh', 'tanh',
                    'log', 'ln', 'exp']

    # Add * before opening parentheses unless it's a known function
    # Regex to match a word followed by a parenthesis
    def insert_mul_before_paren(match):
        token = match.group(1)
        if token in math_functions:
            return f'\\{token}('  # Keep as function call
        else:
            return f'{token} * ('

    equation_text = re.sub(r'\\?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', insert_mul_before_paren, equation_text)
    # equation_text = re.sub(r'(\b[a-zA-Z0-9_\^]+)\s*\(', r'\1 * (', equation_text)  # Add * before opening parentheses
    equation_text = re.sub(r'\)\s*(\b[a-zA-Z0-9_\^]+)', r') * \1', equation_text)  # Add * after closing parentheses
    equation_text = re.sub(r'(\)|\b[a-zA-Z0-9_\^]+)\s+(\(|\b[a-zA-Z0-9_\^]+)', r'\1 * \2', equation_text) # Add * between space-separated variables

    equation_text = equation_text.strip()  # Remove extra spaces

    # Step 3: Extract the final equation of the form LHS = RHS
    out = tuple(equation_text.split("="))
    if len(out) == 1: 
        if not_equation:
            out = ("none", equation_text) # ignore the right hand side in this case
        else:
            print("Equation does not have an equals sign.")
            return
    if len(out) != 2:
        print("Equation has some weird formatting")
        return
    lhs, rhs = out

    left_brackets = '{([<'
    right_brackets = '})]>'

    # Remove text before unclosed left brackets in LHS
    while any(c in lhs for c in left_brackets):
        found_unclosed = False
        for i, char in enumerate(lhs):
            if char in left_brackets:
                # Check if this opening bracket has a matching closing bracket
                stack = [char]
                for j in range(i + 1, len(lhs)):
                    if lhs[j] in right_brackets:
                        if stack:
                            stack.pop()
                    elif lhs[j] in left_brackets:
                        stack.append(lhs[j])
                if stack:  # If stack is not empty, we found an unclosed bracket
                    lhs = lhs[i+1:]
                    found_unclosed = True
                    break
        if not found_unclosed:  # If no unclosed brackets found, exit loop
            break
    
    rhs = rhs[::-1]  # Reverse the RHS string
    # Remove text before unclosed left brackets in LHS
    while any(c in rhs for c in right_brackets):
        found_unclosed = False
        for i, char in enumerate(rhs):
            if char in right_brackets:
                # Check if this opening bracket has a matching closing bracket
                stack = [char]
                for j in range(i + 1, len(rhs)):
                    if rhs[j] in left_brackets:
                        if stack:
                            stack.pop()
                    elif rhs[j] in right_brackets:
                        stack.append(rhs[j])
                if stack:  # If stack is not empty, we found an unclosed bracket
                    rhs = rhs[i+1:]
                    found_unclosed = True
                    break
        if not found_unclosed:  # If no unclosed brackets found, exit loop
            break

    rhs = rhs[::-1]  # Reverse the RHS string back to original order

    # Step 4: Convert to sympy expression
    try:
        sympy_expr = parse_latex(rhs.strip())   # use parse_latex instead of sympify to avoid issues with latex formatting
        return sympy_expr
    except Exception as e:
        # raise ValueError(f"Error parsing equation into sympy: {e}")
        print(f"Error parsing equation into sympy: {e}")

def fast_latex_symbol(latex_str):
    # Assumes format like '\theta_1' or 'm_1'
    if latex_str.startswith('\\'):
        base, _, sub = latex_str[1:].partition('_')
        return Symbol(f"{base}_{{{sub}}}")
    elif '_' in latex_str:
        base, _, sub = latex_str.partition('_')
        return Symbol(f"{base}_{{{sub}}}")
    else:
        return Symbol(latex_str)

def convert_sym_mapping_to_latex(sym_mapping, mode = "fast"):
    """
    Converts a dictionary of symbolic expressions to a LaTeX formatted string.
    
    Parameters:
        sym_mapping (dict): A dictionary where keys are variable names and values are float.
        
    Returns:
        new_sym_mapping: LaTeX formatted keys representing the original keys.
    """
    new_sym_mapping = {}

    for k in sym_mapping:
        split = k.split('_')
        if len(split) == 1: new_sym_mapping[k] = sym_mapping[k]
        elif len(split) == 2:
            latex_name = my_unicode_to_latex.get(split[0], split[0])
            if mode != "legacy": latex_name = latex_name.strip('\\') # latex_name = latex_name[1:] # Remove the leading backslash for non-legacy mode
            new_sym_mapping[latex_name + '_' + split[1]] = sym_mapping[k]
        else: raise ValueError(f"Unexpected variable name: {k}, didn't expect more than 1 underscore in variable name.")
    
    if mode != "legacy" and mode != "fast": 
        # Add .lcase version of the keys
        new_sym_mapping.update({k.lower(): v for k, v in new_sym_mapping.items()})
        return new_sym_mapping
    if mode != "fast":
        return {parse_latex(k):v for k, v in new_sym_mapping.items()}
    return {fast_latex_symbol(k):v for k, v in new_sym_mapping.items()}
    
def extract_final_numerical_answer(response):
    """
    Extracts the final numerical answer (number and unit) from an LLM's response,
    using both boxed notation and contextual hints.
    
    Prioritizes answers near "Final Answer", "Hence", "Therefore", etc.
    
    Parameters:
        response (str): The LLM's response text.
    
    Returns:
        Tuple[float, str] | None: The final (number, unit) pair, or None if not found.
    """

    # Find all instances of \boxed{...}
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = list(re.finditer(boxed_pattern, response))

    if not boxed_matches:
        return None  # No boxed answers found

    # Define regex patterns for number and unit extraction
    number_pattern = r'([-+]?\d*\.\d+|\d+)'
    unit_pattern = r'\\text\{([^}]+)\}'

    # Define contextual keywords
    context_keywords = ["Final Answer", "Hence", "Therefore", "Thus"]

    # Find the last occurrence of a contextual hint
    last_context_pos = -1
    for keyword in context_keywords:
        match = re.search(rf'\b{keyword}\b', response, re.IGNORECASE)
        if match:
            last_context_pos = max(last_context_pos, match.start())  # Track furthest occurrence

    final_answer = None
    selected_distance = float('inf')  # Track distance from context

    for match in boxed_matches:
        content = match.group(1)
        num_match = re.search(number_pattern, content)
        if num_match:
            number = float(num_match.group(0))
            unit_match = re.search(unit_pattern, content)
            unit = unit_match.group(1).strip() if unit_match else ""

            # Check distance from last contextual hint
            distance = abs(match.start() - last_context_pos) if last_context_pos != -1 else float('inf')

            # Prioritize answer closest to the last contextual hint
            if last_context_pos != -1 and distance < selected_distance:
                final_answer = (number, unit)
                selected_distance = distance

    # If no context-based answer was found, return the last boxed one
    if final_answer is None:
        last_match = boxed_matches[-1].group(1)
        num_match = re.search(number_pattern, last_match)
        if num_match:
            number = float(num_match.group(0))
            unit_match = re.search(unit_pattern, last_match)
            unit = unit_match.group(1).strip() if unit_match else ""
            final_answer = (number, unit)

    return final_answer

def extract_bracket_content(text, prefix = "\\boxed", bracket_type = "{"):
    BRACKET_MAP = {
        '{': '}',
        '(': ')',
        '[': ']',
        '<': '>',
    }
    results = []
    i = 0
    while i < len(text):
        start = text.find(prefix+bracket_type, i)
        if start == -1:
            break
        i = start + len(prefix+bracket_type)
        stack = 1
        content_start = i
        while i < len(text) and stack > 0:
            if text[i] == bracket_type:
                stack += 1
            elif text[i] == BRACKET_MAP[bracket_type]:
                stack -= 1
            i += 1
        if stack == 0:
            results.append(text[content_start:i-1])  # exclude final '}'
        else:
            break  # unmatched
    return results

def extract_final_symbolic_answer_new(llm_response: str):
    """
    Extracts the final symbolic equation from an LLM response.
    
    Parameters:
        llm_response (str): The full LLM response containing LaTeX equations.
        not_equation (bool): If True, the function will allow the answer to be a non-equation.
    Returns:
        sympy expression: Parsed symbolic equation.
    """
    # we have to use the following version of math-verify and antlr4-python3-runtime for math-verify to work
    # st()
    # try:
    #     assert version("math-verify") == "0.7.0"
    # except PackageNotFoundError:
    #     raise ImportError("Please install math-verify==0.7.0")
    # try:
    #     assert version("antlr4-python3-runtime") == "4.11.0"
    # except PackageNotFoundError:
    #     raise ImportError("Please install antlr4-python3-runtime==4.11.0")
    # try:
    #     assert version("latex2sympy2_extended") == "1.0.9"
    # except PackageNotFoundError:
    #     raise ImportError("Please install latex2sympy2_extended==1.0.9")

    # Step 1: Locate boxed equation or last displayed equation
    boxed_match = extract_bracket_content(llm_response, prefix = "\\boxed", bracket_type = "{")
    
    if boxed_match:
        equation_text = boxed_match[-1]  # Get the last boxed         
    else: # Fallback
        # If no \boxed{} found, get the last displayed equation
        displayed_match = re.findall(r'\\\[(.*?)\\\]', llm_response, re.DOTALL)
        if not displayed_match:
            displayed_match = re.findall(r'\$\$(.*?)\$\$', llm_response, re.DOTALL) # support for $$ ... $$
        inline_match = re.findall(r'\\\((.*?)\\\)', llm_response, re.DOTALL)
        if not inline_match:
            inline_match = re.findall(r'\$(.*?)\$', llm_response, re.DOTALL) # support for $ ... $
        
        if displayed_match:
            equation_text = displayed_match[-1]
        elif inline_match:
            equation_text = inline_match[-1]
        else:
            raise ValueError("No recognizable equation found in response.")

    equation_text = equation_text.strip()  # Remove extra spaces

    # Step 2: Extract the final equation of the form LHS = RHS
    out = tuple(equation_text.split("="))

    if len(out) <= 1: 
        return parse(enclose(out[0]), extraction_config=math_verify_extraction_config)[0]
    if len(out) != 2:
        print("Equation has some weird formatting")
        out = out[0], out[-1]
    lhs, rhs = out

    try:
        return parse(enclose(rhs.strip()), extraction_config=math_verify_extraction_config)[0]
    except Exception as e:
        return sympify("0")

def my_subs(expr, sub_dict):
    """
    Substitute variables in a sympy expression with values from a dictionary.
    (Apparently, sympy's subs method isn't robust enough to work with string keys, so we implement our own version.)
    
    Parameters:
        expr (sympy expression): The expression to substitute.
        sub_dict (dict): Dictionary where keys are variable names and values are their replacements.
        
    Returns:
        sympy expression: The expression after substitution.
    """
    sp_sub_dict = {}
    for s in expr.free_symbols:
        if s.name in sub_dict:
            sp_sub_dict[s] = sub_dict[s.name]
    expr = expr.subs(sp_sub_dict)
    return expr

def relative_precision(pred, gt, max_relative_error=0.01) -> bool:
    try:
        pred = pred[0] if isinstance(pred, list) else pred
        gt = gt[0] if isinstance(gt, list) else gt

        pred_val = float(pred)
        gt_val = float(gt)

        if gt_val == 0 or pred_val == 0:
            return bool(pred_val == gt_val)
        else:
            relative_error = abs(pred_val - gt_val) / abs(gt_val)
            return bool(relative_error < max_relative_error)
    except:
        return False  # fallback if parsing fails

def convert_to_sci_if_needed(x: float):
    """Return (mantissa, exponent) if x is very small or very large; else just x"""
    if x == 0:
        return 0, 0  # mantissa=0, exponent=0
    if abs(x) >= 1000 or abs(x) < 0.001:
        exponent = math.floor(math.log10(abs(x)))
        mantissa = x / (10 ** exponent)
        return mantissa, exponent
    return x, 0

import signal



def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def _verify_worker(parsed_solution, gt_math_verify, result_queue):
    """Worker function for verification"""
    try:
        result = verify(parsed_solution, gt_math_verify)
        result_queue.put(('success', result))
    except Exception as e:
        result_queue.put(('error', e))

def verify_with_timeout(parsed_solution, gt_math_verify, timeout_seconds=5):
    """Wrapper for verify function with timeout to prevent hanging"""
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass  # Already set
    
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_verify_worker, args=(parsed_solution, gt_math_verify, result_queue))
    
    process.start()
    process.join(timeout_seconds)
    
    if process.is_alive():
        print(f"DEBUG: verify timed out after {timeout_seconds} seconds")
        process.terminate()
        process.join(1)  # Wait a bit for clean termination
        if process.is_alive():
            process.kill()  # Force kill if needed
        return False
    
    try:
        status, result = result_queue.get_nowait()
        if status == 'success':
            return result
        else:
            # Handle the exception that occurred in the worker process
            raise result
    except:
        return False


def _convert_float_worker(x, result_queue):
    """Worker function for float conversion"""
    try:
        value = x[0] if isinstance(x, list) else x
        result = float(value)
        result_queue.put(('success', result))
    except Exception:
        result_queue.put(('error', None))

def try_convert_float_with_timeout(x, timeout_seconds=2):
    """Try to convert x to float with process timeout"""
    # print("DEBUG: About to call try_convert_float", x, timeout_seconds)
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass  # Already set
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_convert_float_worker, args=(x, result_queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1)  # Wait a bit for clean termination
        if process.is_alive():
            process.kill()  # Force kill if needed
        return None
    
    try:
        status, result = result_queue.get_nowait()
        return result if status == 'success' else None
    except:
        return None


# @timeout(seconds=0.1)
def try_convert_float(x):
    """Try to convert x to float; return None if fails"""
    try:
        return float(x[0] if isinstance(x, list) else x)
    except:
        return None

def safe_try_convert_float(x, timeout_secs=1):
    try:
        return timeout(timeout_secs)(try_convert_float)(x)
    except TimeoutException:
        print("DEBUG: Timeout during float conversion.")
        return None
    except Exception as e:
        print(f"DEBUG: Exception in try_convert_float: {e}")
        return None

def math_verify_numerical_answer_simple_relative(pred_answer, expected_answer, use_relative_precision=False, max_relative_error=0.01, mode='relative', relative_to_absolute_threshold=1e-2, data_source = None, force_config_for_validation = True, config=None):
    """
    This function can deal with both numerical and symbolic answers if 
    the expected answer has been post-processed so there is no unit for the numerical answer and the symbolic answer is in latex format.

    pred_answer: str, the answer from the LLM
    expected_answer: str, 
                    - numerical answer: the number without the unit, e.g. "1.23"
                    - symbolic answer: the equation in latex format (e.g. "(a + \\frac{1}{2} \\cdot b \\cdot d) \\cdot d"), this function will wrap the equation in "$$".
    max_relative_error: float, the maximum relative error allowed
    """
    debug_print = config.debug_print
    if debug_print:
        print("DEBUG: About to call math_verify_numerical_answer_simple_relative", pred_answer, expected_answer, data_source, force_config_for_validation)
    if "validation" in data_source and force_config_for_validation:
        use_relative_precision = False
        max_relative_error=1e-2 
        mode='relative'
        relative_to_absolute_threshold=1e-2
    else:
        use_relative_precision = True
        max_relative_error = max_relative_error if max_relative_error is not None else 0.05
        mode = mode if mode is not None else 'relative'
        relative_to_absolute_threshold = relative_to_absolute_threshold if relative_to_absolute_threshold is not None else 1e-2
    if debug_print:
        print("DEBUG: About to parse pred_answer")
    parsed_solution = parse(str(pred_answer), math_verify_extraction_config)
    if debug_print:
        print("DEBUG: parsed_solution", parsed_solution)
    pred_val = safe_try_convert_float(parsed_solution)
    if debug_print:
        print("DEBUG: pred_val", pred_val)
    gt_math_verify = parse(f"${str(expected_answer)}$", math_verify_extraction_config)
    if debug_print:
        print("DEBUG: gt_math_verify", gt_math_verify)
    gt_val = safe_try_convert_float(gt_math_verify)
    if debug_print:
        print("DEBUG: gt_val", gt_val)

    if pred_val is not None and gt_val is not None:
        # if relative_precision(pred_val, gt_val, max_relative_error):
        if debug_print:
            print("DEBUG: About to call check_float_answer", pred_val, gt_val, use_relative_precision, mode, max_relative_error, relative_to_absolute_threshold)
            print("DEBUG: expected answer", expected_answer)
            # print("DEBUG: pred_answer", pred_answer)
            print("DEBUG: parsed_solution", parsed_solution)
            # st()
        if use_relative_precision and check_float_answer(pred_val, gt_val, mode=mode, max_relative_error=max_relative_error, relative_to_absolute_threshold=relative_to_absolute_threshold):
            if debug_print:
                print("DEBUG: check_float_answer done")
            return True, parsed_solution
        elif config and config.conservative_reward:
            return False, parsed_solution
        else:
            try:
                if debug_print:
                    print("DEBUG: About to call verify", parsed_solution, gt_math_verify)
                result = verify(parsed_solution, gt_math_verify)
                if debug_print:
                    print("DEBUG: verify done", result)
                return result, parsed_solution
            except Exception as e:
                if debug_print:
                    print("DEBUG: Exception in verify", e)
                return False, str(e)
    else:
        try:
            if debug_print:
                print("DEBUG: Both none verify")
            result = verify(parsed_solution, gt_math_verify)
            if debug_print:
                print("DEBUG: verify done", result)
            return result, parsed_solution
        except Exception as e:
            if debug_print:
                print("DEBUG: Exception in verify", e)
            return False, str(e)
    
def math_verify_numerical_answer(pred_answer, expected_answer, max_relative_error=0.01, max_decimal_error=0.01):
    parsed_solution = parse(str(pred_answer))
    gt_math_verify = parse(f"${str(expected_answer)}$")
    pred_val = safe_try_convert_float(parsed_solution)
    gt_val = safe_try_convert_float(gt_math_verify)

    if pred_val is not None and gt_val is not None:
        if pred_val == 0 or gt_val == 0:
            return relative_precision(pred_val, gt_val, max_relative_error)
        
        # use scientific notation for values far from 1
        pred_m, pred_e = convert_to_sci_if_needed(pred_val)
        gt_m, gt_e = convert_to_sci_if_needed(gt_val)
        
        # If exponents differ, compare full scientific numbers
        if pred_e != gt_e:
            return relative_precision(parsed_solution, gt_math_verify, max_relative_error)
        else:
            # Exponents match → compare mantissas with absolute error, e.g. 2.345e-3 and 2.34e-3 are considered equivalent
            return abs(pred_m - gt_m) < max_decimal_error
    else:
        # fallback symbolic/structural verification
        try:
            return verify(parsed_solution, gt_math_verify)
        except:
            return False
        