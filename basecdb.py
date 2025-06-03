import streamlit as st
import chromadb

def load_chromadb_data():
    client = chromadb.PersistentClient(path="base")
    collection = client.get_or_create_collection(name="bilan_blocs")
    # On récupère documents et metadatas, pas 'ids' dans include !
    results = collection.get(include=["documents", "metadatas"], limit=1000)
    # ids sont présents dans results même s’ils ne sont pas dans include
    return results

st.title("Visualisation des données ChromaDB")

data = load_chromadb_data()

if data["ids"]:
    for idx, doc_id in enumerate(data["ids"]):
        st.subheader(f"Document ID: {doc_id}")
        st.markdown(f"**Source:** {data['metadatas'][idx].get('source', 'N/A')}")
        st.text_area("Texte extrait", data["documents"][idx], height=200)
else:
    st.write("Aucun document trouvé dans ChromaDB.")
