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
MAX_RETRY = 5

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
                    choices=["OpenPilot_2k19", "External_jutah", "OpenPilot_2016"],
                    help="The dataset you want to process (OpenPilot_2k19, External_jutah, OpenPilot_2016)")
parser.add_argument('--size',
                    type=int,
                    default=-1,
                    help="The size of the dataset you want to use")
parser.add_argument('--odd_file',
                    type=str,
                    default="ODD_Converted.txt",
                    help="The name of the odd file you want to use")
parser.add_argument('--portion',
                    type=str,
                    required=True,
                    choices=["pass", "fail", "both"],
                    help="Select which type of files you want to use (pass, fail, both)")
args = parser.parse_args()

# Make sure you have set a dataset size
assert args.size > 0, "Dataset size can not be less than or equal to 0"

# Get the list of databases
datasets = os.listdir(DATASET_DIRECTORY)
assert args.dataset in datasets, f"The dataset was not found in `{DATASET_DIRECTORY}`"

# Check if the API key file exists
api_key_file = f"{API_KEY_DIRECTORY}/{args.api_key}.key"
assert os.path.exists(api_key_file) == True, f"API key file '{api_key_file}' does not exist."

# Load the prompts
with open(f'../1_Datasets/ODD/{args.odd_file}', 'r') as file:
    # Read the content of the file
    data = file.read()

# Get each of the questions
odd = data.split("\n")
odd_clean = []

# Clean the data
for q in odd:
    number = q[:q.find(")")]
    question = q[q.find(")")+2:]
    odd_clean.append((number, question))

    # Make sure the directories for this number
    output_dir = f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions_{args.size}/chat_gpt/{number}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Load all images
if args.portion == "both":
    all_images = sorted(glob.glob(f"{DATASET_DIRECTORY}/{args.dataset}/4_SelectedData_{args.size}/*.png"))
elif args.portion == "pass" or args.portion == "fail":
    all_images = sorted(glob.glob(f"{DATASET_DIRECTORY}/{args.dataset}/4_SelectedData_{args.size}/{args.portion}_*.png"))
# Load all the descriptions
all_descriptions = sorted(glob.glob(f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions_{args.size}/chat_gpt/*/*.txt"))

# For each image
for img in tqdm(all_images):

    # Get the image basename
    basename = os.path.basename(img)

    # For each element of the ODD
    for odd_number, odd_prompt in odd_clean:

        # Check if we already have this:
        output_txt_file = f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions_{args.size}/chat_gpt/{odd_number}/{basename[:-4]}_output.txt"
        output_json_file = f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions_{args.size}/chat_gpt/{odd_number}/{basename[:-4]}_json.txt"

        # Figure out if we have seen this before
        if output_txt_file in all_descriptions:
            continue

        # Repeat this step until it works
        current_retry = 0
        while current_retry < MAX_RETRY:

            # Create the payload and send it
            headers, payload = create_request(img, odd_prompt, api_key_file)
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            # Get the response
            response_data = response.json()

            # Check if 'error' key exists in the response
            if 'error' in response_data:
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

            # Check if 'error' key exists in the response
            elif 'choices' in response_data:

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

        if current_retry >= MAX_RETRY:
            exit()

        time.sleep(30)
