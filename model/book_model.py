import torch
import pandas as pd
import scipy.spatial
from transformers import MLukeTokenizer, LukeModel

class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def encode(self, sentences):
        encoded = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        model_output = self.model(**encoded)
        pooled = self._mean_pooling(model_output, encoded["attention_mask"])
        return pooled.cpu()


# モデルとデータを初期化
MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
model = SentenceLukeJapanese(MODEL_NAME)
data = pd.read_csv("data/bookmeter_ranking.csv")

def predict(query):
    book_vectors = torch.load("model/book_vectors.pt")
    query_vec = model.encode([query])[0].unsqueeze(0)
    distances = scipy.spatial.distance.cdist(query_vec.numpy(), book_vectors.numpy(), "cosine")[0]
    idx = distances.argmin()
    matched = data.iloc[idx]
    return {
        "title": matched["タイトル"],
        "author": matched["著者"],
        "summary": matched["あらすじ"],
        "image_url": matched["画像URL"],
        "input_text": query
    }











