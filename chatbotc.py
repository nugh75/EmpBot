import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def divide_text_into_chunks(text, separator):
    """Dividi il testo in base al separatore."""
    chunks = text.split(separator)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks

def embed_chunks(chunks, model_name="sentence-transformers/all-MiniLM-L12-v2"):
    """Calcola le embeddings per ogni chunk utilizzando Sentence Transformers."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

def index_with_faiss(embeddings):
    """Indicizza le embeddings con FAISS."""
    embeddings = embeddings.cpu().numpy()
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def main():
    st.title("Testo Divider e Indicizzatore con FAISS")

    # Carica il file di testo
    uploaded_file = st.file_uploader("Carica un file di testo (.txt)", type="txt")
    separator = st.text_input("Inserisci il separatore di chunk", value="++++++")

    if uploaded_file is not None:
        # Leggi il contenuto del file
        text = uploaded_file.read().decode('utf-8')

        # Dividi il testo in chunk
        chunks = divide_text_into_chunks(text, separator)
        st.write(f"Testo suddiviso in {len(chunks)} chunk.")

        # Mostra i chunk divisi
        if st.checkbox("Mostra i chunk divisi"):
            for i, chunk in enumerate(chunks, start=1):
                st.write(f"Chunk {i}:")
                st.write(chunk)
                st.write("----")

        # Calcola le embeddings
        embeddings = embed_chunks(chunks)
        
        # Indicizza le embeddings con FAISS
        index = index_with_faiss(embeddings)

        # Mostra il numero di chunk indicizzati
        st.write(f"Numero di chunk indicizzati: {index.ntotal}")

        # Sezione di ricerca
        query = st.text_input("Inserisci una query per trovare chunk simili")
        if query:
            query_embedding = embed_chunks([query])
            D, I = index.search(query_embedding.cpu().numpy(), k=3)

            st.write("Chunk più simili alla query:")
            for idx in I[0]:
                st.write(f"Chunk {idx + 1}:")
                st.write(chunks[idx])
                st.write("----")

if __name__ == "__main__":
    main()
