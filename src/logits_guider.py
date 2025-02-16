import types
import torch
import torch.nn.functional as F

class LogitsGuider:
    def __init__(self):
        self.batch_size = None
        self.first_seq_id = None
        self.end_seq_id = None
        self.next_token_sample_list = []
        self.base_logits = []
        self.loss = []

    def prepare_for_generation(self, batch_size, max_tokens):
        self.batch_size = batch_size
        self.next_token_sample_list = [None for _ in range(max_tokens + 1)]
        self.base_logits = [None for _ in range(max_tokens + 1)]
        self.loss = {i: [] for i in range(batch_size)}

        if self.first_seq_id is None:
            self.first_seq_id = 0
        else:
            self.first_seq_id = self.end_seq_id
        self.end_seq_id = self.first_seq_id + batch_size

    def sample(self, running_model, logits, sampling_metadata):
        next_tokens = running_model.sampler(logits, sampling_metadata)
        for i, (seq_group, next_token_output) in enumerate(zip(sampling_metadata.seq_groups, next_tokens.outputs)):
            seq_id = seq_group.seq_ids[0]
            if seq_id < self.first_seq_id or seq_id >= self.end_seq_id:
                raise Exception(f"seq_id ({seq_id}) should be larger or equal to first_seq_id ({self.first_seq_id}), \
                        and less than end_seq_id ({self.end_seq_id})")
            output_token_ids_len = len(seq_group.seq_data[seq_id].output_token_ids)
            if seq_id == self.first_seq_id: # base prompt
                self.next_token_sample_list[output_token_ids_len] = next_token_output.samples[0] # get the token of base prompt
                self.base_logits[output_token_ids_len] = logits[i, :].detach()
            else:
                if self.next_token_sample_list[output_token_ids_len] is not None:
                    next_tokens.outputs[i].samples[0] = self.next_token_sample_list[output_token_ids_len]
                    actual_id = seq_id - self.first_seq_id
                    self.loss[actual_id].append(
                        torch.square(
                            F.softmax(self.base_logits[output_token_ids_len], dim=0) \
                            - F.softmax(logits[i, :], dim=0)
                        ).mean().item()
                    )

        return next_tokens

def bind_sample_wrapper(model):
    def sample_wrapper(running_model, logits, sampling_metadata):
        return logits_guider.sample(running_model, logits, sampling_metadata)

    logits_guider = LogitsGuider()
    model.sample = types.MethodType(sample_wrapper, model)
    return logits_guider
