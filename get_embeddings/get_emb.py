import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import pandas as pd
import pickle
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

login(token="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")

files = ["AA", "DNA", "NucleicAmbigous", "NucleicMixed",
         "RNA", "SmallMolecule"]

for file in files:
    df = pd.read_csv(f"data/content_id/{file}.csv")

    tokenizer = AutoTokenizer.from_pretrained("PharMolix/BioMedGPT-LM-7B")
    model = AutoModel.from_pretrained("PharMolix/BioMedGPT-LM-7B").to(device)
    model.eval()

    emb_dict = {}
    for i in tqdm(range(len(df["content"]))):
        encoded_input = tokenizer(df["content"][i], return_tensors="pt")
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            output = model(**encoded_input)

        embedding = output.last_hidden_state
        embedding = embedding.squeeze()
        embedding = embedding.mean(dim=0)
        embedding = embedding/torch.linalg.norm(embedding)
        emb_dict[df['id'][i].item()] = embedding.half().cpu()

    with open(f"data/dict_{file}.pkl", "wb") as f:
        pickle.dump(emb_dict, f)