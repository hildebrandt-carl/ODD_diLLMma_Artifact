import os
import glob
import time
import base64
import requests
import datetime
import argparse
from tqdm import tqdm


# Decare the dataset directory
DATASET_DIRECTORY = "../1_Datasets/Data"
API_KEY_DIRECTORY = "./API_Keys"

# Set the number of retries
MAX_RETRY = 10

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# roughly $0.002 per 1000 tokens
def create_request(image_path, prompt, api_key_file, max_tokens=25000):

    # Read the API key from the file
    with open(api_key_file, 'r') as file:
        API_KEY = file.read().strip()

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": 
                    {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }],
        "max_tokens": 4096
    }

    return headers, payload


# Get the folders
parser = argparse.ArgumentParser(description="Used to identify which scenarios pass and which fail")
parser.add_argument('--api_key',
                    type=str,
                    required=True,
                    help="The API key to use (academic, personal)")
parser.add_argument('--dataset',
                    type=str,
                    required=True,
                    choices=["External_Jutah", "OpenPilot_2k19", "OpenPilot_2016"],
                    help="The dataset you want to process (External_Jutah, OpenPilot_2k19, OpenPilot_2016)")
parser.add_argument('--odd_file',
                    type=str,
                    default="Described_ODD.txt",
                    help="The name of the odd file you want to use")
args = parser.parse_args()

# Get the list of databases
datasets = os.listdir(DATASET_DIRECTORY)
assert args.dataset in datasets, f"The dataset was not found in `{DATASET_DIRECTORY}`"

# Check if the API key file exists
api_key_file = f"{API_KEY_DIRECTORY}/{args.api_key}.key"
assert os.path.exists(api_key_file) == True, f"API key file '{api_key_file}' does not exist."

# Load the prompts
with open(f'../1_Datasets/ODD/{args.odd_file}', 'r') as file:
    # Read the content of the file
    prompt = file.read()

print("Using Prompt:")
print("=====================")
print(prompt)
print("=====================")


INPUT_DIR  = f"{DATASET_DIRECTORY}/{args.dataset}/4_SelectedData"
OUTPUT_DIR = f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions/ChatGPT_Base"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all images
all_images = sorted(glob.glob(f"{INPUT_DIR}/*.png"))

# Load all the descriptions
all_descriptions = sorted(glob.glob(f"{OUTPUT_DIR}/*.txt"))

# For each image
for img in tqdm(all_images):

    # Get the image basename
    basename = os.path.basename(img)

    # Check if we already have this:
    output_txt_file = f"{OUTPUT_DIR}/{basename[:-4]}_output.txt"
    output_json_file = f"{OUTPUT_DIR}/{basename[:-4]}_json.txt"

    # Figure out if we have seen this before
    if output_txt_file in all_descriptions:
        continue

    # Repeat this step until it works
    current_retry = 0
    while current_retry < MAX_RETRY:

        # Create the payload and send it
        headers, payload = create_request(img, prompt, api_key_file)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # Get the response
        response_data = response.json()

        # Check if 'choices' key exists in the response
        if 'choices' in response_data:

            # Get the response
            response_text = response_data["choices"][0]["message"]["content"]

            # Open the file in write mode ('w'). If the file doesn't exist, it will be created.
            with open(f"{output_txt_file}", 'w') as file:
                # Write the text to the file
                file.write(response_text)

            # Open the file in write mode ('w'). If the file doesn't exist, it will be created.
            with open(f"{output_json_file}", 'w') as file:
                # Write the text to the file
                file.write(str(response_data))

            current_retry = 0
            break

        # Else save the response
        else:
            # Handle the error
            print(f"Error detected: Retry attempt {current_retry+1}/{MAX_RETRY}")
            print(f"Error message: {response_data}")

            # Get the current datetime
            now = datetime.datetime.now()
            datetime_str = now.strftime("%Y%m%d_%H%M%S")

            # Open the file in write mode ('w'). If the file doesn't exist, it will be created.
            with open(f'error_{datetime_str}.txt', 'w') as file:
                # Write the text to the file
                file.write(str(response_data))

            # Retry
            current_retry += 1
            if current_retry < MAX_RETRY:
                wait_time = 90 * current_retry
                print(f"Waiting {wait_time} seconds")
                time.sleep(wait_time)
                continue

        if current_retry >= MAX_RETRY:
            exit()

        time.sleep(30)
