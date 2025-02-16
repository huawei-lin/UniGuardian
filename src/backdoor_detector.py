from src.logits_guider import bind_sample_wrapper

import re
import torch
import numpy as np
from vllm import LLM, SamplingParams

class BackdoorDetector():
    def __init__(self, model_config):
        gpu_num = torch.cuda.device_count()
        self.max_model_len = model_config.get("max_model_len", 512)
        self.model = LLM(**model_config, dtype="float16", tensor_parallel_size=gpu_num, scheduling_policy="priority", max_num_batched_tokens=4096, max_num_seqs=32)
        self.tokenizer = self.model.get_tokenizer() # must add mask token
        print("model and tokenizer loaded!")

        if not self.tokenizer.mask_token:
            for idx, added_token in self.tokenizer.added_tokens_decoder.items():
                if "unused" in added_token.content or "reserved" in added_token.content:
                    self.tokenizer.mask_token = "[MASK]"
                    self.tokenizer.mask_token_id = idx
        if not self.tokenizer.mask_token:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
            # self.model.resize_token_embeddings(len(self.tokenizer))
        print("tokenizer edited!")

        self.logits_guider = bind_sample_wrapper(self.model.llm_engine.model_executor.driver_worker.model_runner.model)
        print("get logits_guider!")

    def get_masked_word_and_prompts(self, prompt, mask_indices_list):
        prompt_splited = re.split(rf'(\s|{self.tokenizer.eos_token[0]}.*?{self.tokenizer.eos_token[-1]})', prompt)
        self.log["prompt_splited"] = prompt_splited
        if not mask_indices_list:
            mask_indices_list = [(i,) for i in range(len(prompt_splited))]
        print("old mask_indices_list:", mask_indices_list)
        mask_indices_list = self.clean_mask_indices(prompt_splited, mask_indices_list)
        print("new mask_indices_list:", mask_indices_list)
        self.log["mask_indices_list"] = mask_indices_list

        # the first element is original prompt
        masked_words = []
        masked_prompts = []
        for _, mask_indices in enumerate(mask_indices_list):
            masked_words.append([prompt_splited[index] for index in mask_indices])
            masked_prompts.append("".join([
                x if i not in mask_indices else self.tokenizer.mask_token
                for i, x in enumerate(prompt_splited)
            ]))
        self.log["masked_words"] = masked_words
        self.log["masked_prompts"] = masked_prompts

        masked_prompts.insert(0, prompt) # add a base prompt
        return mask_indices_list, masked_words, masked_prompts

    def clean_mask_indices(self, prompt_splited, mask_indices_list):
        new_list = []
        for mask_indices in mask_indices_list:
            sub_list = []
            for i in mask_indices:
                mask_word = prompt_splited[i].strip()
                if len(mask_word) > 1:
                    sub_list.append(i)
                elif len(mask_word) == 0:
                    continue
                elif mask_word.isprintable() and not mask_word.isspace():
                    sub_list.append(i)
            if len(sub_list) > 0:
                new_list.append(tuple(sorted(sub_list)))
        return list(set(new_list))

    def inference(self, masked_prompts, mask_indices_list, sample_config=None):
        def compute_loss(data):
            data = np.array(data)
            weights = np.linspace(1, 0, len(data))**2
            weighted_sum = np.dot(weights, data)
            return weighted_sum / np.sum(weights)

        if sample_config is None:
            sample_config = {}
        batch_size = len(masked_prompts)
        priority = [0] + [1 for i in range(batch_size - 1)]

        sampling_params = SamplingParams(**sample_config)
        self.logits_guider.prepare_for_generation(batch_size, sample_config["max_tokens"])
        generation = self.model.generate(masked_prompts, sampling_params=sampling_params, priority=priority)

        self.log["base_generated_ids"] = generation[0].outputs[0].token_ids 
        self.log["base_generation"] = generation[0].outputs[0].text
        print("---- base generation ----")
        print(self.log["base_generation"])
        print("-------------------------")

        loss_list = [self.logits_guider.loss[i] for i in range(1, batch_size)]
        self.log["base_loss_vector"] = self.logits_guider.loss[0]
        self.log["mask_loss_vector"] = loss_list
        loss_list = [sum(x) for x in loss_list]
        self.log["loss_list"] = loss_list

        return loss_list

    def get_local_zscores(self, loss_list):
        loss_list = np.array(loss_list)
        mean = loss_list.mean()
        std = loss_list.std()
        z_scores = (loss_list - mean) / std

        self.log["local_mean"] = mean
        self.log["local_std"] = std
        return z_scores.tolist()

    def detect(self, prompt, mask_indices_list=None, sample_config=None):
        self.log = {}
        self.log["prompt"] = prompt

        mask_indices_list, masked_words_list, masked_prompt_list = self.get_masked_word_and_prompts(prompt, mask_indices_list)
        loss_list = self.inference(masked_prompt_list, mask_indices_list, sample_config=sample_config)

        local_z_score_list = self.get_local_zscores(loss_list)
        self.log["local_z_score_list"] = local_z_score_list

        return loss_list, local_z_score_list, self.log

    def __call__(self, *args, **kwargs):
        return self.detect(*args, **kwargs)
