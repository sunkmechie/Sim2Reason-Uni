from evals.OlympiadBench.inference.code.evaluators.custom_llm_evaluator import CustomLLM_Evaluator
from evals.OlympiadBench.inference.judge import MathJudger, extract_answer

import json, os

from llm.utils import basic_utils

def get_prompts(cfg, evaluator=None):
    all_llm_prompts = {}
    all_llm_questions = {}
    all_save_result_dirs = {}

    all_is_chinese = {}
    all_is_deepseek = {}

    for val_data_file in cfg.data.val_files:
        dataset_name = val_data_file.split('/')[-1].replace('.json','')
        if "TP" in dataset_name:
            print("Warning: Theorem proving problems cannot currently be automatically evaluated. Skipping!")
            continue

        is_chinese = 'zh' in val_data_file
        is_deepseek = 'deepseek' in cfg.model_name
        
        save_result_dir = f"evals/OlympiadBench/generations/{dataset_name}/{cfg.model_name.replace('/','_')}"
        with open(val_data_file, 'r', encoding='utf-8') as f:
            json_dataset = json.load(f)
        
        os.makedirs(save_result_dir, exist_ok=True)
        
        llm_prompts, questions = evaluator.get_input(json_dataset_path=val_data_file, json_dataset=json_dataset, save_result_dir=save_result_dir)

        all_llm_prompts[dataset_name] = llm_prompts
        all_llm_questions[dataset_name] = questions
        all_save_result_dirs[dataset_name] = save_result_dir

        all_is_chinese[dataset_name] = is_chinese
        all_is_deepseek[dataset_name] = is_deepseek

    return all_llm_prompts, all_llm_questions, all_save_result_dirs, all_is_chinese, all_is_deepseek

def generate_responses(cfg, llm_prompts, llm, sampling_params):
    output_texts = []
    if cfg.solve_locally:
        all_model_outputs, token_stats = basic_utils.call_model(llm, sampling_params, llm_prompts, cfg)
        for model_outputs in all_model_outputs:
            for output in model_outputs.outputs:
                output_texts.append(output.text)
    else:
        all_model_outputs, token_stats, error_generating = basic_utils.call_llm_api(cfg, cfg.model_name, llm_prompts, n=cfg.actor_rollout_ref.rollout.val_kwargs.n)
        for model_outputs in all_model_outputs:
            for output_text in model_outputs:
                output_texts.append(output_text)

    return output_texts, token_stats

def evaluate_responses(cfg, dataset_name, token_stats, save_result_dir, judger, is_chinese, is_deepseek, logger):
    # evaluator.evaluate saves json files named i_to_j.json which we need to access

    results = os.listdir(save_result_dir)
    available_id_list = []	# deduplication
    merged_results = []
    correct_num = 0
    for res in results:
        if res.split('.')[-1] != 'json' or "token_stats.json" in res or "merged_results.json" in res: # only json files
            continue
        
        res_path = os.path.join(save_result_dir, res)
        with open(res_path, 'r', encoding='utf-8') as f:
            res_data = json.load(f)
        
        for id, question in enumerate(res_data):
            model_ans = question['model_output'][cfg.model_name]['raw_output']
            if (len(model_ans)>0 and model_ans != '<Inappropriate content in response>' and model_ans!='<No response>' and ('code:' not in model_ans or 'message:' not in model_ans)):
                if question['id'] in available_id_list:	# 重复数据
                    continue
                else:
                    available_id_list.append(question['id'])
            model_answer = question['model_output'][cfg.model_name]['raw_output']
            model_answer = extract_answer(is_chinese, model_answer, is_deepseek)

            answer_type = question['answer_type']
            if 'Tuple' in answer_type: # 目前可机评的数据中 没有 need_human_evaluate
                judge_result = judger.judge(model_answer, question['final_answer'][0])
            else:
                if question['error']:
                    if ',' in question['error']:
                        precisions = question['error'].split(',')
                        precisions = [float(p) if p else 1e-8 for p in precisions]
                        judge_result = judger.judge(model_answer, question['final_answer'][0], precisions)
                    else:
                        precision = float(question['error'])
                        judge_result = judger.judge(model_answer, question['final_answer'][0], precision)
                else:
                    judge_result = judger.judge(model_answer, question['final_answer'][0])

            if judge_result:
                correct_num += 1 # 貌似也没用到
            res_data[id]['model_output'][cfg.model_name]['answer'] = model_answer
            res_data[id]['model_output'][cfg.model_name]['correctness'] = judge_result

        merged_results += res_data # 保留所有的处理结果
    
    acc = 0.0 if len(available_id_list) ==0 else correct_num/len(available_id_list)
    response_length = 0.0 if len(token_stats["tokens/output"]) == 0 else sum(token_stats["tokens/output"]) / len(token_stats["tokens/output"])
    valid_end = 0.0 if len(token_stats["tokens/valid_end"]) == 0 else sum(token_stats["tokens/valid_end"]) / len(token_stats["tokens/valid_end"])
    
    print(f"Dataset: {dataset_name}, Model: {cfg.model_name}, Total: {len(available_id_list)}, Correct: {correct_num}, Accuracy: {acc}, Response length: {response_length}, Valid end rate: {valid_end}")

    with open(os.path.join(save_result_dir, 'merged_results.json'), 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=4)

    with open(os.path.join(save_result_dir, 'token_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(token_stats, f, ensure_ascii=False, indent=4)

    logger.log({f'{dataset_name}/acc': acc, f'{dataset_name}/response_length': response_length, f'{dataset_name}/valid_end': valid_end}, step=0)

def benchmark(cfg, llm, sampling_params, logger):
    judger = MathJudger()

    evaluator = CustomLLM_Evaluator(cfg)

    all_llm_prompts, all_llm_questions, all_save_result_dirs, all_is_chinese, all_is_deepseek = get_prompts(cfg, evaluator)

    for val_data_file in cfg.data.val_files:
        dataset_name = val_data_file.split('/')[-1].replace('.json','')
        
        if "MM_" in dataset_name or "TP_" in dataset_name:
            print("Warning: Multimodal problems and Theorem proving problems cannot currently be automatically evaluated. Skipping!")
            continue

        llm_prompts = all_llm_prompts[dataset_name]
        questions = all_llm_questions[dataset_name]
        save_result_dir = all_save_result_dirs[dataset_name]
        is_chinese = all_is_chinese[dataset_name]
        is_deepseek = all_is_deepseek[dataset_name]

        output_texts, token_stats = generate_responses(cfg, llm_prompts, llm, sampling_params)

        evaluator.evaluate(json_dataset_path=val_data_file, save_result_dir=save_result_dir, questions=questions, response_texts=output_texts)

        evaluate_responses(cfg, dataset_name, token_stats, save_result_dir, judger, is_chinese, is_deepseek, logger)

    logger.finish()

'''    
def benchmark_old():
    judger = MathJudger()

    for val_data_file in cfg.data.val_files:
        dataset_name = val_data_file.split('/')[-1].replace('.json','')
        
        if "TP" in dataset_name:
            print("Warning: Theorem proving problems cannot currently be automatically evaluated. Skipping!")
            continue

        model_name = cfg.model_name.replace('/','_')
        save_result_dir = f"evals/OlympiadBench/generations/{dataset_name}/{model_name}"
        is_chinese = 'zh' in val_data_file
        is_deepseek = 'deepseek' in cfg.model_name
        correct_num = 0

        with open(val_data_file, 'r', encoding='utf-8') as f:
            json_dataset = json.load(f)
        
        os.makedirs(save_result_dir, exist_ok=True)
        
        llm_prompts, questions = evaluator.get_input(json_dataset_path=val_data_file, json_dataset=json_dataset, save_result_dir=save_result_dir)

        output_texts = []
        if cfg.solve_locally:
            all_model_outputs, token_stats = basic_utils.call_model(llm, sampling_params, llm_prompts, cfg)
            for model_outputs in all_model_outputs:
                for output in model_outputs.outputs:
                    output_texts.append(output.text)
        else:
            all_model_outputs, token_stats, error_generating = basic_utils.call_llm_api(cfg, cfg.model_name, llm_prompts, n=cfg.actor_rollout_ref.rollout.val_kwargs.n)
            for model_outputs in all_model_outputs:
                for output_text in model_outputs:
                    output_texts.append(output_text)

        if cfg.solve_locally and rank == 0:
            # evaluator.evaluate(json_dataset_path=val_data_file, json_dataset=json_dataset, save_result_dir=save_result_dir, questions=questions, response_texts=output_texts)

            results = os.listdir(save_result_dir)
            available_id_list = []	# deduplication
            merged_results = []
            for res in results:
                if res.split('.')[-1] != 'json':
                    continue
                if "token_stats.json" in res or "merged_results.json" in res:
                    continue
                res_path = os.path.join(save_result_dir, res)
                with open(res_path, 'r', encoding='utf-8') as f:
                    res_data = json.load(f)
                for id, question in enumerate(res_data):
                    print("question:", question)
                    model_ans = question['model_output'][cfg.model_name]['raw_output']
                    if (len(model_ans)>0 and model_ans != '<Inappropriate content in response>' and model_ans!='<No response>' and ('code:' not in model_ans or 'message:' not in model_ans)):
                        if question['id'] in available_id_list:	# 重复数据
                            continue
                        else:
                            available_id_list.append(question['id'])
                    model_answer = question['model_output'][cfg.model_name]['raw_output']
                    model_answer = extract_answer(is_chinese, model_answer, is_deepseek)

                    answer_type = question['answer_type']
                    if 'Tuple' in answer_type: # 目前可机评的数据中 没有 need_human_evaluate
                        judge_result = judger.judge(model_answer, question['final_answer'][0])
                    else:
                        if question['error']:
                            if ',' in question['error']:
                                precisions = question['error'].split(',')
                                precisions = [float(p) if p else 1e-8 for p in precisions]
                                judge_result = judger.judge(model_answer, question['final_answer'][0], precisions)
                            else:
                                precision = float(question['error'])
                                judge_result = judger.judge(model_answer, question['final_answer'][0], precision)
                        else:
                            judge_result = judger.judge(model_answer, question['final_answer'][0])

                    if judge_result:
                        correct_num += 1 # 貌似也没用到
                    res_data[id]['model_output'][cfg.model_name]['answer'] = model_answer
                    res_data[id]['model_output'][cfg.model_name]['correctness'] = judge_result

                merged_results += res_data # 保留所有的处理结果

            print(f"Dataset: {dataset_name}, Model: {cfg.model_name}, Total: {len(available_id_list)}, Correct: {correct_num}, Accuracy: {0.0 if len(available_id_list) ==0 else correct_num/len(available_id_list)}")

            with open(os.path.join(save_result_dir, 'merged_results.json'), 'w', encoding='utf-8') as f:
                json.dump(merged_results, f, ensure_ascii=False, indent=4)

            with open(os.path.join(save_result_dir, 'token_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(token_stats, f, ensure_ascii=False, indent=4)
'''
# pass