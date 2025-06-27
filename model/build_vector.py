from book_model import SentenceLukeJapanese
import pandas as pd
import torch
import os

os.makedirs("model", exist_ok=True)

MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
model = SentenceLukeJapanese(MODEL_NAME)
data = pd.read_csv("C:/Users/sabb-/GeekSalon/MYPRODUCT/data/bookmeter_ranking.csv")
sentences = data["あらすじ"].tolist()

print("Encoding and saving...")
vectors = model.encode(sentences)
torch.save(vectors, "model/book_vectors.pt")
print("Saved to book_vectors.pt")