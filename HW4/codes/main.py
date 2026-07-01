from openai import OpenAI
from task import Math
from prompt import get_prompt
import argparse
import scale

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_url', type=str, required=True, help='Server Address')
	parser.add_argument('--api_key', type=str, required=True, help='OpenAI API Key')
	parser.add_argument('--model', type=str, required=True, help='Model Name')
	parser.add_argument('--dataset', choices=['math500', 'gsm8k'], help='Dataset')
	parser.add_argument('--task', choices=['zero-shot', 'few-shot', 'scaling'], help='Task')
	args = parser.parse_args()

	client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    
	if args.dataset == 'math500' or args.dataset == 'gsm8k': # Math problem

		task = Math(args.dataset, args.task, args.model, client)
		if args.task == 'scaling': # Inference time scaling
			# TODO START
			# You can use task.test(scaling, i) to test the i-th problems, or just task.test(scaling) to test on the whole dataset.
			# If you just need to partially test the dataset, use task.test(prompt, limit=num) to test for the first num problems of the dataset.
			# If you want to check the output of your model, use task.test(scaling, i, output=True)
			# [bonus] You can use your own inference-time scaling strategies, if you implement them in scale.py

			scaling = scale.majority_voting
			task.test(
				scaling, 
				program_id='all'
			)
			
			# TODO END

		else: # Prompt strategy
			prompt = get_prompt(args.dataset, args.task)
			# TODO START
			# You can use task.test(prompt, i) to test the i-th problems, or just task.test(prompt) to test on the whole dataset.
			# If you just need to partially test the dataset, use task.test(prompt, limit=num) to test for the first num problems of the dataset.
			# If you want to check the output of your model, use task.test(prompt, i, output=True)
			task.test(
				prompt, 
				program_id='all'
			)
			# TODO END
	
	else:
		pass

if __name__ == "__main__":
	main()