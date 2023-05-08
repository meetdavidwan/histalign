from datasets import DatasetDict, Dataset
import numpy as np
import random
import os

np.random.seed(42)
random.seed(42)

file_dir = "template_data_preprocessing_code/data/processed/analogy_google/split/0.5"

data = {
    "train": [],
    "validation": [],
    "test": [],
}

cat2prompt = {
    "people": [
        "Between the {} and the {}, I decided to first talk to the",
        "The {} and the {} are my favorites, and I especially love the",
        "The {} and the {} happily live together. One day, bad luck happens to the",
        "The {} and the {} stay at my house, and I need to take care of the",
    ],
    "location":[
        "I went to {} and {} before, and I love one of the places more, which is",
        "{} and {} are my favorites, and I especially love",
        "My uncle used to live in {} and {} but now, he is selling his house in",
        "The traveler plans to visit {} and {}, and the traveler first arrives in",
    ],
    "noun":[
        "love the {} and the {}, and my favorite is the",
        "Yesterday, a man encountered the {} and the {}. Today, he again saw the",
        "There are the {} and the {} in front of a woman, and she decides to pursue the",
        "If you can choose the {} or the {}, would you choose the",
    ]
}

stop_word_set = set(['he', 'she', 'her', 'his'])

for prefix in [ "location_capital-common-countries", "location_capital-world", "location_city-in-state", "location_three_locations", "people_family" ]:
    print(prefix)
    train = [line.strip().split() for line in open(os.path.join(file_dir, prefix + "_train"))]
    valid = [line.strip().split() for line in open(os.path.join(file_dir, prefix + "_one_overlap"))]
    test = [line.strip().split() for line in open(os.path.join(file_dir, prefix + "_no_overlap"))]
    if "three_locations" in prefix:
        typ = prefix.split("_")[0]
        cat = "-".join(prefix.split("_")[1:])
    else:
        typ, cat = prefix.split("_")
    templates = cat2prompt[typ]

    for split, dat in [ ("train",train), ("validation", valid), ("test", test) ]:

        for a,b,c,d in dat:
            for template in templates:
                for x,y,i,j in [ (a,d, b,c), (b,c, a,d)]:
                    if x in stop_word_set or y in stop_word_set:
                        continue
                    data[split].append( (cat, template.format(x,y), x, (i,j) ) )
                    data[split].append( (cat, template.format(x,y), y, (i,j) ) )

                    data[split].append( (cat, template.format(y,x), x, (i,j) ) )
                    data[split].append( (cat, template.format(y,x), y, (i,j) ) )
   
_dataset = DatasetDict()
for split in data:
    cat, x, y, other = zip(*data[split])
    _dataset[split] = Dataset.from_dict(
        {
            "category": cat,
            "text": x,
            "label": y,
            "other": other,
        }
    )

print(_dataset)

_dataset.save_to_disk("data/ambiguous_template")