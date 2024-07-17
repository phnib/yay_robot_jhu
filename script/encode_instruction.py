"""
This script will loop over all the "episode_{idx}.json" files in the provided dataset_dir, encode the commands using either DistilBERT or CLIP, and save the results to "episode_{idx}_encoded.json" or "episode_{idx}_encoded_clip.json" files in the same directory.

Example usage:
$ python script/encode_instruction.py --dataset_dir /scr/lucyshi/dataset/aloha_bag_3_objects_dagger --encoder distilbert
"""

import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import sys

path_to_yay_robot = os.getenv('PATH_TO_YAY_ROBOT')
# print(path_to_yay_robot)
if path_to_yay_robot:
    sys.path.append(os.path.join(path_to_yay_robot, 'src'))
from aloha_pro.aloha_scripts.utils import initialize_model_and_tokenizer, encode_text


def process_file(file_path, command, encoder, tokenizer, model):


    _, phase_command = command.split("_")[0], " ".join(command.split("_")[1:])
    emb = encode_text(phase_command, encoder, tokenizer, model)
    print(phase_command)

    # Define the content of the episode
    episode_content = [
        {
            "command": phase_command,
            "embedding": emb
        }
    ]
    if os.path.exists(file_path):
        # Read existing data from the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Append new data to the existing data
        data.append(episode_content)

        # Write the episode content to the file
        with open(file_path, "w") as file:
            json.dump(data, file)

    else:
        # Write the episode content to the file
        with open(file_path, "w") as file:
            json.dump(episode_content, file)


def generate_embeddings_file(candidate_texts, encoder, tokenizer, model, output_file):
    # List to store embeddings
    embeddings_list = []

    # Loop through each text and encode
    for text in tqdm(candidate_texts, desc="Processing candidate texts"):
        embedding = encode_text(text, encoder, tokenizer, model)
        embeddings_list.append(embedding)

    # Convert the embeddings list to a numpy array
    embeddings_array = np.array(embeddings_list)

    # Save the embeddings array to a file
    np.save(output_file, embeddings_array)
    print(f"Saved embeddings to {output_file}")


def load_candidate_texts(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        # Extract the instruction (text before the colon), strip whitespace, and then strip quotation marks
        candidate_texts = [line.split(":")[0].strip().strip("'\"") for line in lines]
    return candidate_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode commands in JSON dataset using either DistilBERT or CLIP."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing the JSON files to be processed.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["distilbert", "clip"],
        default="distilbert",
        help="Encoder type to be used ('distilbert' or 'clip').",
    )
    parser.add_argument(
        "--from_count",
        action="store_true",
        help="Generate embeddings directly from instructions in 'count.txt'.",
    )

    args = parser.parse_args()

    tokenizer, model = initialize_model_and_tokenizer(args.encoder)
    dataset_dir = args.dataset_dir
    encoder = args.encoder

    phases = os.listdir(dataset_dir)
    # print(phases)

    # # if args.from_count:
    output_file = os.path.join(dataset_dir, f"candidate_embeddings_{encoder}.json")
    # generate_embeddings_file(
    #     phases, encoder, tokenizer, model, output_file
    # )
    # else:
    # List of files to process
    files_to_process = [
        f
        for f in os.listdir(dataset_dir)
        if not f.endswith(".json")
    ]
    print(files_to_process)
    # Loop over all the "episode_{idx}.json" files in the dataset_dir with a progress bar
    for file_name in tqdm(files_to_process, desc="Processing files"):
        process_file(
            output_file, file_name, encoder, tokenizer, model
        )
