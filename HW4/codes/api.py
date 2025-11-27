from openai import OpenAI

def get_model_response(client, model, prompt, max_retries=10):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                top_p=0.95,
                temperature=0.0
            )

            if response and response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                print(f"Attempt {attempt + 1}: Empty response, retrying...")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed due to error: {e}.")
            
    print("Failed to get model response after all attempts.")
    return None
