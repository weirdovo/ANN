from api import get_model_response

def majority_voting(client, model, problem, sample = 5):
	# TODO START
	# To get model's response for the prompt, use get_model_response(client, model, prompt).
	# You can change the number of shots with the 'sample' variable.

	prompt = problem
	result = get_model_response(client, model, prompt)

	# TODO END

	return result

# TODO START
# [bonus] Implement other inference-time scaling methods below. You should keep the same interface with 'majority_voting' function.

# TODO END