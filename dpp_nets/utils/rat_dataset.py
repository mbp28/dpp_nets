from torch.utils.data import TensorDataset
from dpp_nets.my_torch.utilities import pad_tensor
from dpp_nets.utilities.io import make_embd
import torch

anno_path = os.path.join(root, 'annotations.json')
rationale_data = read_rationales(anno_path)
max_set_size = 412

reviews = []
rationales_appear = []
rationales_smell = []
rationales_palate = []

def create_bool_list(words, extract):
    bool_list = [False for i in range(len(words))]
    for ex in extract:
        for i in range(*ex):
            bool_list[i] = True
    return bool_list

word_to_ix = make_embd(embd_path, only_index_dict=True)

for rationale in rationale_data:
    words = rationale['x']
    
    extract_appear = rationale['0']
    extract_smell = rationale['1']
    extract_palate = rationale['2']
    
    bool_appear = create_bool_list(words, extract_appear)
    bool_smell = create_bool_list(words, extract_smell)
    bool_palate = create_bool_list(words, extract_palate)
    
    review = [word_to_ix[word] + 1 for word in words if word in word_to_ix]
    rationale_appear = [b for word, b in zip(words, bool_appear) if word in word_to_ix]
    rationale_smell = [b for word, b in zip(words, bool_smell) if word in word_to_ix]
    rationale_palate = [b for word, b in zip(words, bool_palate) if word in word_to_ix]
    
    review = torch.LongTensor(review)
    rationale_appear = torch.ByteTensor(rationale_appear)
    rationale_smell = torch.ByteTensor(rationale_smell)
    rationale_palate = torch.ByteTensor(rationale_palate)
    
    review = pad_tensor(review, 0, 0, max_set_size)
    rationale_appear = pad_tensor(rationale_appear, 0, 0, max_set_size)
    rationale_smell = pad_tensor(rationale_smell, 0, 0, max_set_size)
    rationale_palate = pad_tensor(rationale_palate, 0, 0, max_set_size)
    
    reviews.append(review)
    rationales_appear.append(rationale_appear)
    rationales_smell.append(rationale_smell)
    rationales_palate.append(rationale_palate)
    
reviews = torch.stack(reviews)
rationales_appear = torch.stack(rationales_appear)
rationales_smell = torch.stack(rationales_smell)
rationales_palate = torch.stack(rationales_palate)

targets = torch.stack([rationales_appear, rationales_smell, rationales_palate],dim=1)

dataset = TensorDataset(reviews, targets)
torch.save(dataset, '/Users/Max/data/beer_reviews/pytorch/annotated.pt')