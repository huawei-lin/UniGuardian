import sys
import re
from src.data_loader import read_data
from src.utils import generate_indices_tuples, get_config
from src.backdoor_detector import BackdoorDetector
import argparse
from tqdm import tqdm
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--max_model_len', default=4096, type=int)
    parser.add_argument('--begin_id', default=0, type=int)
    args = parser.parse_args()
    print(f"get args: {args}")

    config = get_config(args.config_path)

    model_config = {
        "model": config.model_name_or_path,
        "max_model_len": args.max_model_len,
    }
    print(f"model_config: {model_config}")

    backdoor_detector = BackdoorDetector(model_config)
    print("get backdoor_detector")

    tokenizer = backdoor_detector.tokenizer

    output_path = config.output_path
    if output_path is not None:
        if args.begin_id <= 0:
            save_fw = open(output_path, "w")
        else:
            save_fw = open(output_path, "a")

    list_data_dict = read_data(config.dataset_path)
    print("get list_data_dict:", len(list_data_dict))
    for data_id, data_dict in enumerate(tqdm(list_data_dict[args.begin_id:])):
        print(data_id)
        instruction = data_dict['instruction']

        messages = [
            {"role": "system", "content": f"{config.system_prompt}"},
            {"role": "user", "content": f"{instruction}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", tokenize=False)

        instruction_splited = re.split(rf'(\s|{tokenizer.eos_token[0]}.*?{tokenizer.eos_token[-1]})', instruction)
        prompt_splited = re.split(rf'(\s|{tokenizer.eos_token[0]}.*?{tokenizer.eos_token[-1]})', prompt)
        print(instruction)
        print(instruction_splited)
        print(prompt_splited)
        begin_index = next((i for i in range(len(prompt_splited) - len(instruction_splited) + 1) if prompt_splited[i:i + len(instruction_splited)] == instruction_splited), -1)
        # mask_indices_list = [[i, i + 1] for i in range(begin_index, begin_index + len(instruction_splited))]
        available_index = []
        for i in range(begin_index, begin_index + len(instruction_splited)):
            word = prompt_splited[i]
            if len(word) > 1:
                available_index.append(i)
            elif word.isprintable() and not word.isspace():
                available_index.append(i)

        K = config.detection.num_masks_per_instruction
        if isinstance(K, float):
            K = int(len(available_index)**K)
        K = min(min(max(1, K), config.detection.max_masks_per_instruction), len(available_index))

        num_masked_instructions = config.detection.num_masked_instructions
        if isinstance(num_masked_instructions, float):
            num_masked_instructions = int(len(available_index)*num_masked_instructions)
        num_masked_instructions = max(1, num_masked_instructions)

        print("len:", len(available_index), "K:", K, "num_masked_instructions:", num_masked_instructions)
        mask_indices_list = generate_indices_tuples(available_index, num_masked_instructions, K, begin_index)

        sample_config={
            **config.generate.__dict__
        }
        loss_list, local_z_score_list, log = backdoor_detector(
            prompt,
            mask_indices_list,
            sample_config=sample_config
        )
        print("---"*20)

        sorted_id = np.argsort(local_z_score_list)[::-1]
        for i in sorted_id:
            print(i, [log["prompt_splited"][log["mask_indices_list"][i][x]] for x in range(len(log["mask_indices_list"][i]))], loss_list[i], local_z_score_list[i])

        result = {
            "instruction": instruction,
            "prompt": prompt,
            "data_dict": data_dict,
            "sample_config": sample_config,
            "K": K,
            "N": num_masked_instructions,
            **log,
        }

        if output_path is not None:
            save_fw.write(json.dumps(result) + "\n")
            save_fw.flush()


if __name__ == "__main__":
    main()
