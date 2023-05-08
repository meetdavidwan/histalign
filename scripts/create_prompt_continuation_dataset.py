from datasets import DatasetDict, Dataset, load_from_disk

from transformers import AutoTokenizer

inp_length = 50

d = DatasetDict()
tokenizer = AutoTokenizer.from_pretrained("gpt2",cache_dir="/nas-ssd/davidwan/cache")

for split in ["train", "valid"]:
    z = [line.strip() for line in open("writingPrompts/{}.wp_comb_detok".format(split))]
    d[split] = Dataset.from_dict( {"text": z} )

# test set
z = [line.strip() for line in open("writingPrompts/test.wp_comb_detok")]
# filter example
z = [line for line in z if 50 < len(line.split()) < 1000]

z = tokenizer(z).input_ids
x = [zz[0:inp_length] for zz in z]
y = [zz[inp_length:inp_length+512]  for zz in z]

x_tok = [tokenizer.decode(xx, skip_special_tokens=True) for xx in x]
y_tok = [tokenizer.decode(yy, skip_special_tokens=True) for yy in y]

d["test_{}".format(inp_length)] = Dataset.from_dict( {"prompt": x, "continuation": y, "prompt_tok": x_tok ,"continuation_tok": y_tok} )

print(d)
d.save_to_disk("data/writingprompts")