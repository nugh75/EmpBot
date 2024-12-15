import streamlit as st
from langchain_openai import ChatOpenAI
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import faiss
import numpy as np
from dotenv import load_dotenv
import os

# Carica le istruzioni dal file
with open('instructions.txt', 'r', encoding='utf-8') as f:
    instrucion = f.read()

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Ottieni la chiave API da una variabile d'ambiente
api_key = os.getenv("OPENAI_API_KEY")

# Configura il modello LLN
llm = ChatOpenAI(
    temperature=0.0, 
    api_key=api_key, 
    model_name="gpt-4o-mini"
)

# Configura le embeddings con HuggingFace
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Percorso del file di commozione
file_path = 'commozione.txt'

# Funzione per calcolare e indicizzare le embeddings
def process_text_chunks(text):
    # Dividi il testo in chunk
    chunks = text.split('\n++++++\n')

    # Calcola le embeddings per i chunk
    chunk_embeddings = embeddings_model.embed_documents(chunks)

    # Converti le embeddings in un array numpy
    embeddings_array = np.array(chunk_embeddings)

    # Inizializza l'indice FAISS
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    return index, chunks

# Funzione per trovare chunk simili
def find_similar_chunks(query, index, chunks, top_k=5):
    # Calcola le embeddings per la query
    query_embedding = embeddings_model.embed_query(query)
    query_embedding_array = np.array(query_embedding).reshape(1, -1)

    # Trova i chunk più simili
    D, I = index.search(query_embedding_array, top_k)

    # Restituisci i chunk simili
    similar_chunks = [chunks[idx] for idx in I[0]]
    return similar_chunks

# Configurazione del prompt
prompt_template = ChatPromptTemplate.from_template(
    """Sei un rilevatore di emozioni di commozione. Utilizza le seguenti istruzioni {instrucion} per valutare se l'episodio descritto è un caso di commozione. 

    Episodio: {episode}

    Casi di Commozione: Ecco alcuni esempi di casi reali di commozione {context}. Confronta l'episodio con questi casi.

    Fornisci una risposta con 'Sì' se l'episodio rappresenta un caso di commozione, altrimenti rispondi 'No'. Inoltre, fornisci, andando a capo, una spiegazione dettagliata del perché l'episodio è o non è considerato un caso di commozione. Se si tratta di un caso di commozione, indica gli elementi emotivi coinvolti."""
)


# Funzione per analizzare il sentiment dell'episodio utilizzando il modello e il prompt
def sentiment_analysis_with_context(llm, episode, similar_chunks):
    # Prepara il contesto dai chunk simili
    context = "\n".join(similar_chunks)
    
 
    # Creare il prompt con il contesto e l'episodio
    prompt = prompt_template.format(instrucion=instrucion, episode=episode, context=context)
    
    # Esegui il modello LLaMA
    response = llm.invoke(prompt)

    # Estrai il testo dalla risposta del modello
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Analizza la risposta del modello per determinare il sentiment
    return response_text.strip()

# Configurazione dell'interfaccia Streamlit
def main():
    st.title("""Rilevatore di emozioni di commozione """)

    # Leggi il contenuto del file di commozione
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Processa il testo e indicizza i chunk
    index, chunks = process_text_chunks(text)

    # Form per inserire un episodio
    episode = st.text_area("Racconta un episodio di commozione")

    if st.button("Analizza"):
        # Trova chunk simili
        similar_chunks = find_similar_chunks(episode, index, chunks)

        # Mostra i chunk più simili
        st.write("Episodi di commozione più simili trovati:")
        for i, chunk in enumerate(similar_chunks, start=1):
            st.write(f"**Caso di commozione n. {i}:**")
            st.write(chunk)
            st.write("\n" + "-"*50 + "\n")

        # Analizza il sentimento con contesto
        sentiment = sentiment_analysis_with_context(llm, episode, similar_chunks)

        # Mostra il risultato dell'analisi del sentimento
        st.write(f"Risultato dell'analisi dell'episodio di commozione: {sentiment}")

if __name__ == "__main__":
    main()
