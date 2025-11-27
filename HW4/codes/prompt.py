from numpy import promote_types


def zero_shot(problem, task):
    # This function performs a zero-shot task, meaning it generates an output based solely on the given problem 
    # and task type without using any prior examples or training samples. 
    # It relies on general understanding to solve or predict the result directly.
	# TODO START

	# problem = "In Zeta notation: A=1, B=2, C=3, D=4, E=5. What is CAD + CA? Express your answer in Zeta notation."
	prompt = f"Question: {problem}\nAnswer:"
	return prompt
	# TODO END

def few_shot(problem, task, shot = 5):
	# This function performs a few-shot task, where it uses a small number of examples (controlled by the shot parameter) 
	# to guide the model’s understanding of the task before solving the given problem, improving accuracy through contextual learning.
	# You can change the number of shots with the 'shot' variable.
	# tips: we provide the related util function and example data in ./utils, and you could also design this function from scratch. 
	# TODO START
	from utils.math_opensource_utils.examples import get_examples
	import random
	
	examples_dict = get_examples()
	
	if task == 'gsm8k':
		examples_key = 'gsm8k'
	elif task == 'math500':
		examples_key = 'math500'
	else:
		raise ValueError(f"Unknown task: {task}")
	
	examples = examples_dict.get(examples_key, [])

	# examples = [
	# 	(
	# 		"In Zeta notation: A=1, B=2, C=3, D=4, E=5. What is AB + ACC? Express your answer in Zeta notation.",
	# 		"AB = 12, ACC = 133. AB + ACC = 12 + 133 = 145 = ADE. The answer is ADE."
	# 	),
	# 	(
	# 		"In Zeta notation: A=1, B=2, C=3, D=4, E=5. What is BAD + AA? Express your answer in Zeta notation.",
	# 		"BAD = 214, AA = 11. BAD + AA = 214 + 11 = 225 = BBE. The answer is BBE."
	# 	)
	# ]
	
	if not examples or shot <= 0:
		return f"Question: {problem}\nAnswer:"
	
	sample_size = min(shot, len(examples))
	selected_examples = random.sample(examples, sample_size)
	
	example_prompts = []
	for question, answer in selected_examples:
		example_prompts.append(f"Question: {question}\nAnswer: {answer}")
	
	# problem = "In Zeta notation: A=1, B=2, C=3, D=4, E=5. What is CAD + CA? Express your answer in Zeta notation."
	prompt = "\n\n".join(example_prompts) + "\n\n" + f"Question: {problem}\nAnswer:"
	
	print(prompt)

	return prompt
	# TODO END

def get_prompt(dataset, prompt):
	if dataset == 'math500' or dataset == 'gsm8k':
		if prompt == 'zero-shot':
			return zero_shot
		elif prompt == 'few-shot':
			return few_shot
		else:
			raise NotImplementedError
	
	else:
		raise NotImplementedError
