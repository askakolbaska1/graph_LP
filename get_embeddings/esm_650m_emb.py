import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import pandas as pd
import pickle
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

login(token="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")


model_name = "facebook/esm2_t30_150M_UR50D"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

df = pd.read_csv(f"../data/nodes/AA.csv")

emb_dict = {}
for i in tqdm(range(len(df["content"])), leave=True):
    encoded_input = tokenizer(df["content"][i], return_tensors="pt")
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        output = model(**encoded_input)

    embedding = output.last_hidden_state
    embedding = embedding.squeeze()
    embedding = embedding.mean(dim=0)
    embedding = embedding/torch.linalg.norm(embedding)
    emb_dict[df['id'][i].item()] = embedding.half().cpu()

with open(f"data/dicts/dict_ESM_650M.pkl", "wb") as f:
    pickle.dump(emb_dict, f)