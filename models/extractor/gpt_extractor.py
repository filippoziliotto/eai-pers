import sys
import os
import json

# Add base path to PYTHONPATH
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
base_path = '/Users/filippoziliotto/Desktop/Repos/eai-pers/'

sys.path.append(base_path)

# Import the function to load episodes
from dataset.load_maps import load_episodes

# Import openai and its ChatCompletion resource, then set the API key
import openai
openai.api_key = os.environ['OPENAI_API_KEY']

# Function to call GPT-4o-mini using the new OpenAI API
def call_gpt(prompt: str, description: str) -> str:
    """
    Calls the GPT model with the given prompt and description using the new API and returns the response.
    
    Parameters:
        prompt (str): The input prompt for GPT.
        description (str): Additional description or context for the prompt.
    
    Returns:
        str: The GPT response.
    """
    try:
        # Combine the prompt and description for better context
        full_input = f"{prompt}{description}"
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an helpful assistant."},
                {"role": "user", "content": full_input}
            ],
            # temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error: {e}"

# Load episodes from the specified data split
data_dir = "data"
split = "val"
episodes = load_episodes(data_dir, split)

# Process episodes: For each episode, read the prompt and call the GPT model
for i, episode in enumerate(episodes):
    
    # Read the prompt from a text file
    with open('models/extractor/prompt.txt', 'r') as f:
        prompt = f.read()
    
    output = call_gpt(prompt, episode['summary'])
    episode['summary_extraction'] = output

    print(f"Processed episode {i}")
    # break  # Remove or adjust the break if you want to process all episodes

# Save the extracted episodes to a JSON file
output_file = os.path.join(data_dir, split, "filtered_episodes.json")
with open(output_file, "w") as f:
    json.dump(episodes, f, indent=2)

print(f"Extracted episodes saved to: {output_file}")
