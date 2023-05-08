from transformers import GPT2LMHeadModel
import torch
from tqdm import tqdm
import json

import numpy as np

import sys
from datasets import load_from_disk

from modeling import *
from transformers import AutoTokenizer, AutoConfig

dataset = load_from_disk("data/ambiguous_template")

test_dataset = dataset["test"]
# test_dataset = dataset["validation"]

batch_size = 32

prompts = test_dataset["text"]
labels = test_dataset["label"]
others = test_dataset["other"]

# aggregate two options
prompts = [prompts[i] for i in range(0,len(prompts),2)]
labels = [(labels[i], labels[i+1]) for i in range(0, len(labels), 2)]
others = [others[i] for i in range(0, len(others), 2)]

dir = sys.argv[1]
class_name = sys.argv[2]
name2class = {
    "orig": GPT2LMHeadModel,
    "trime": GPT2LMHeadModelWithTrime,
    "trimebrio": GPT2LMHeadModelWithHistAlign,
}

print(dir, class_name)

cache_dir = None

tokenizer = AutoTokenizer.from_pretrained(dir, cache_dir=cache_dir, padding_side = "left",)
config = AutoConfig.from_pretrained(dir, cache_dir=cache_dir)
model = name2class[class_name].from_pretrained(dir, cache_dir=cache_dir, config=config)
model = model.cuda().half()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

topks = [1,2,5,10,25,50,75,100]
accuracy = {k:[] for k in topks}

preds = []

for i in tqdm(range(0, len(prompts), batch_size)):
    prompt, label, other = prompts[i:i+batch_size], labels[i:i+batch_size], others[i:i+batch_size]
    inps = tokenizer(prompt, padding=True)
    input_ids = [ [tokenizer.eos_token_id] + ii for  ii in inps.input_ids]
    attention_mask = [ [0] + am for am in inps.attention_mask]
    
    input_ids = torch.tensor(input_ids).cuda()
    attention_mask = torch.tensor(attention_mask).cuda()

    gen_tokens = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=len(input_ids[0]) + 1,
        return_dict_in_generate=True,
        output_scores=True,
    )

    label_ = tokenizer([ " ".join(lab) for lab in label]).input_ids

    _label = []

    # account for bpe and starting space for wordpiece
    for x,y in label:
        _label.append ( [tokenizer.encode(" " + x)[0], tokenizer.encode(x)[0], tokenizer.encode(" " + y)[0], tokenizer.encode(y)[0]] )

    for score in gen_tokens.scores:
        _, am = torch.topk(score,100)

        am = am.detach().cpu().numpy()

        for j, b in enumerate(am):

            cur_lab = _label[j]

            first_present = cur_lab[0] in b or cur_lab[1] in b
            second_present = cur_lab[2] in b or cur_lab[3] in b

            # if not (first_present and second_present):
            # print(prompt[j])
            # print(tokenizer.convert_ids_to_tokens(cur_lab), tokenizer.convert_ids_to_tokens(b))

            for topk in topks:
                topk_tok = b[:topk]
                
                first_present = cur_lab[0] in topk_tok or cur_lab[1] in topk_tok
                second_present = cur_lab[2] in topk_tok or cur_lab[3] in topk_tok

                accuracy[topk].append(1 if (first_present and second_present) else 0)

print("accuracy", [np.mean(accuracy[topk]) for topk in topks])