from evals.OlympiadBench.inference.code.evaluators.evaluator import Evaluator
import json, os

class CustomLLM_Evaluator(Evaluator):
    def __init__(self, cfg):
        model_name = cfg.model_name
        super(CustomLLM_Evaluator, self).__init__(cfg, model_name)
        self.cfg = cfg

    def make_input(self, prompt, question_content):
        if self.is_chinese:
            subject = '数学' if self.is_math else '物理'
            question_message = prompt + '\n' + question_content
            messages = [
                {
                    'role': 'system',
                    'content': f'你是一个中文人工智能助手。请根据要求，完成下面的{subject}竞赛题目。'
                },
                {
                    'role': 'user',
                    'content': question_message
                }
            ]
        else:
            subject = 'Math' if self.is_math else 'Physics'
            question_message = prompt + '\n' + question_content
            messages = [
                {
                    'role': 'system',
                    'content': f'You are an AI assistant. Please answer the following {subject} competition questions as required.'
                },
                {
                    'role': 'user',
                    'content': question_message
                }
            ]
        return messages

    def get_input(self, json_dataset_path, json_dataset, save_result_dir):
        self.is_theorem_proving = 'TP' in json_dataset_path
        self.is_math = False # We are only dealing with physics problems here
        self.is_chinese = 'zh' in json_dataset_path

        llm_prompts = []
        questions = []
        for idx, question in enumerate(json_dataset):

            if self.cfg.get("only_mechanics", False) and question["subfield"] != "Mechanics":
                continue
            prompt = self.make_prompt(question)
            
            if 'context' in question and question['context']:
                input_msg = self.make_input(prompt, question['context'] + '\n' + question['question'])
            else:
                input_msg = self.make_input(prompt, question['question'])
            
            llm_prompts.append(input_msg)
            questions.append(question)

        return llm_prompts, questions
    
    def evaluate(self, json_dataset_path, save_result_dir, questions, response_texts):
        model_name = self.cfg.model_name  # Always use cfg.model_name

        # remove all .json files in save_result_dir
        if not os.path.exists(save_result_dir):
            os.makedirs(save_result_dir)
        else:
            for file in os.listdir(save_result_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(save_result_dir, file))

        # Save results in batches of 100
        temp_result = []
        for idx, question in enumerate(questions):
            answer = response_texts[idx] if response_texts[idx] else ""
            if 'model_output' not in question:
                question['model_output'] = {model_name: {'raw_output': answer}}
            else:
                question['model_output'][model_name] = {'raw_output': answer}
            temp_result.append(question)

            # Save every 100 results
            if idx % 100 == 99:
                save_start_id = idx - 99
                save_path = os.path.join(save_result_dir, f'{save_start_id}_to_{idx}.json')
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(temp_result, f, ensure_ascii=False, indent=4)
                temp_result = []

        # Save remaining results
        if temp_result:
            save_start_id = 100 * int(idx / 100)
            save_path = os.path.join(save_result_dir, f'{save_start_id}_to_{idx}.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(temp_result, f, ensure_ascii=False, indent=4)

        print(f'Evaluation finished for {json_dataset_path}.')