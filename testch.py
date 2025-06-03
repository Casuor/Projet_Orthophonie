import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Chargement du modèle E5
model_name = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Fonction pour encoder une phrase avec E5
def encoder_e5(textes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Préfixe requis par le modèle
    textes = [f"query: {texte}" for texte in textes]

    tokens = tokenizer(textes, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens)

    attention_mask = tokens["attention_mask"]
    embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
    return embeddings.cpu().numpy()

# Connexion Chroma
def get_collection(collection_name="bilan_blocs", db_path="base"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name)

# Recherche
def rechercher_similaires(question, collection, top_k=15):
    docs = collection.get(include=["documents", "embeddings", "metadatas"])
    vecteurs = docs["embeddings"]
    textes = docs["documents"]
    metadatas = docs["metadatas"]

    q_vec = encoder_e5([question])[0].reshape(1, -1)
    sims = cosine_similarity(q_vec, vecteurs)[0]

    scores = sorted(zip(sims, textes, metadatas), key=lambda x: x[0], reverse=True)[:top_k]
    return scores

# MAIN
if __name__ == "__main__":
    question = input("❓ Entre ta question : ")

    collection = get_collection()
    resultats = rechercher_similaires(question, collection)

    print("\n--- Résultats similaires ---\n")
    for score, texte, meta in resultats:
        print(f"[Score : {score:.3f}] Bloc {meta['bloc_index']}\n{texte}\n{'-'*60}")
