from api import get_model_response

def majority_voting(client, model, problem, sample = 10):
	# TODO START
	# To get model's response for the prompt, use get_model_response(client, model, prompt).
	# You can change the number of shots with the 'sample' variable.
	
	from collections import Counter
	from utils.math_opensource_utils.parser import extract_answer, strip_string
	
	if sample <= 0:
		sample = 1
	
	prompt_template = "Question: {}\nAnswer:"
	extracted_answers = []
	
	for _ in range(sample):
		prompt = prompt_template.format(problem)
		response = get_model_response(client, model, prompt)
		if not response:
			continue
		
		# Use 'math500box' as default - it handles both \boxed{} and "answer is" patterns
		answer = extract_answer(response, 'math500box')
		answer = strip_string(answer)
		if not answer:
			answer = response.strip()
		
		extracted_answers.append(answer)
	
	if not extracted_answers:
		result = ""
		return result
	
	vote_counter = Counter(extracted_answers)
	final_answer = vote_counter.most_common(1)[0][0]
	
	result = f"The answer is {final_answer}"
	print(f"Majority Voting Result: {final_answer}")
	# TODO END

	return result

# TODO START
# [bonus] Implement other inference-time scaling methods below. You should keep the same interface with 'majority_voting' function.

# TODO END