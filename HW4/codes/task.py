from api import get_model_response
import concurrent.futures
import json
import re
from utils.math_opensource import compute_scores
from tqdm import tqdm
from collections import defaultdict

class Math:
    def __init__(self, dataset, task, model, client):
        file_path = f"data/{dataset}.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

        if dataset != 'math500':
            for item in self.data:
                answer = item.get('answer', '')
                match = re.search(r'#### (.*)', answer)
                if match:
                    item['final'] = match.group(1).strip()
                    item['answer'] = re.sub(r'####\s*' + re.escape(match.group(1)), '', answer).strip()
                else:
                    item['final'] = ''

        self.client = client
        self.model = model
        self.model_name = self.model.split('/')[-1]
        self.dataset = dataset
        self.task = task

    def show(self, id):
        item = self.data[id]
        for key, value in item.items():
            print(f"{key.upper()}: {value}")

    def generate(self, prompt_func, id):
        field = 'problem' if self.dataset == 'math500' else 'question'
        question = self.data[id][field]

        if prompt_func.__name__ == 'zero_shot' or prompt_func.__name__ == 'cot' or prompt_func.__name__ == 'few_shot':
            prompt = prompt_func(question, self.dataset)
            print(prompt)
            response = get_model_response(self.client, self.model, prompt)
        else:
            response = prompt_func(self.client, self.model, question)

        return {
            "id": id,
            "prompt": prompt,
            "response": response,
            "answer": self.data[id].get('answer' if self.dataset == 'math500' else 'final', ''),
            "task": f"{self.dataset}box"
        }

    def generate_all(self, prompt_func, id, limit=None, output_path=None, max_workers=16):
        if id != 'all':
            count = 1
        else:
            count = len(self.data) if limit is None else min(limit, len(self.data))
        results = [None] * count
        
        def safe_generate(i, single=False):
            try:
                result = self.generate(prompt_func, i)
                if single is False:
                    results[i] = result
                else:
                    results[0] = result
            except Exception as e:
                print(f"[ID {i}] Generation failed: {e}")

        if id == 'all':
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(executor.map(safe_generate, range(count)), total=count, desc="Generating"))
        else:
            safe_generate(id, single=True)

        final_results = [r for r in results if r is not None]

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in final_results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        return final_results

    def evaluate(self, data, program_id):
        scores = compute_scores(data, f"./results/{self.model_name}_{program_id}_eval.json")
        print("final score: ", scores)
        print("Evaluation completed.")
        return scores
     
    def test(self, prompt, program_id='all', limit=None):
        results = self.generate_all(prompt, program_id, limit=limit, output_path=f"./results/{self.model_name}_{program_id}_generated.jsonl")
        level_results = defaultdict(list)

        for item, result in zip(self.data, results):
            level = item.get("level", "unknown")
            level_results[level].append(result)
            
        for level, group in level_results.items():
            score = compute_scores(group, f"./results/{self.model_name}_{program_id}_level_{level}_eval.json")
            print(f"Level: {level}, Count: {len(group)}, Score: {score}")
                
        final_score = self.evaluate(results, program_id)
        return final_score