import fitz
import chromadb
from uuid import uuid4
import torch
from transformers import AutoTokenizer, AutoModel

# Chargement du modèle E5
model_name = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Fonction d'encodage avec le modèle E5
def encoder(blocs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Ajouter "passage: " comme recommandé par le modèle E5
    textes = [f"passage: {bloc}" for bloc in blocs]

    encoded_input = tokenizer(textes, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Moyenne pondérée des embeddings (mean pooling)
    attention_mask = encoded_input["attention_mask"]
    embeddings = (model_output.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
    return embeddings.cpu().numpy()

# === Extraction du texte complet
def extract_full_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc).strip()

# === Découpage logique
def decouper_par_structure(text):
    lignes = text.splitlines()
    blocs, tampon = [], ""

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue
        if ligne.endswith(":") and len(ligne.split()) <= 6:
            if tampon:
                blocs.append(tampon.strip())
                tampon = ""
            tampon += ligne + "\n"
        else:
            tampon += ligne + "\n"
    if tampon:
        blocs.append(tampon.strip())
    return blocs

# === Injection dans ChromaDB avec vecteurs personnalisés
def injecter_dans_chroma(blocs, db_path="base", collection_name="bilan_blocs"):
    client = chromadb.PersistentClient(path=db_path)

    collection = None
    for col in client.list_collections():
        if col.name == collection_name:
            collection = client.get_collection(name=collection_name)
            break

    if collection is None:
        collection = client.create_collection(name=collection_name)

    embeddings = encoder(blocs)

    ids = [str(uuid4()) for _ in blocs]
    metadatas = [{"source": "bilan.pdf", "bloc_index": i} for i in range(len(blocs))]

    collection.add(documents=blocs, embeddings=embeddings.tolist(), ids=ids, metadatas=metadatas)

    print(f"✅ {len(blocs)} blocs injectés dans ChromaDB ({collection_name})")

# === MAIN
if __name__ == "__main__":
    pdf_path = "bilan5.pdf"
    texte = extract_full_text(pdf_path)
    blocs = decouper_par_structure(texte)
    injecter_dans_chroma(blocs)

