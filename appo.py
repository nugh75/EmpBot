import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import faiss
import numpy as np
import os

# Configura il modello LLaMA locale
llama_model = ChatOpenAI(
    base_url="http://localhost:11434/v1", 
    temperature=0.5, 
    api_key="not-need", 
    model_name="llama3"
)

# Configura le embeddings con HuggingFace
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Percorso del file di commozione
file_path = '/home/nugh75/git-repository/chatbotc/commozione.txt'

# Istruzioni per comprendere il concetto di "commuoversi"
istruzioni = """
Istruzioni per Comprendere il Concetto di "Commuoversi"
Il concetto di "commuoversi" è un'emozione complessa che emerge in diverse situazioni. Di seguito, vengono forniti i diversi aspetti e le condizioni che definiscono questo sentimento, come descritto dai partecipanti allo studio.

Definizione e Caratteristiche
Emozione di Valenza Mista:

"Commuoversi" è descritto come un'emozione temporanea che può essere positiva o negativa. È spesso vista come una combinazione di emozioni benigne e tristi, o di tenerezza e dispiacere.
Emozione Intensa:

È caratterizzata da un'intensità emotiva che può derivare da eventi molto significativi e salienti. È spesso associata a sentimenti di felicità o tristezza immensurabili.
Empatia:

L'empatia gioca un ruolo cruciale. "Commuoversi" implica una connessione profonda con le esperienze emotive di altre persone, come mettere se stessi nei panni degli altri e risuonare con il loro dolore o gioia.
Espressione Emotiva:

L'essere emozionati si manifesta attraverso espressioni emotive spesso incontrollabili, come piangere o sorridere. Può dare una sensazione di liberazione, simile a uno sfogo emotivo.
Condizioni per Commuoversi
Il sentimento di "commuoversi" può essere scatenato da diverse condizioni e circostanze, che possono essere suddivise in quattro categorie principali:

1. Commuoversi "Ego-Centrati" (Goal Personali)
Eventi Rilevanti Personali:
Si è commossi da eventi molto rilevanti che accadono a sé stessi. Questo può includere il raggiungimento di un obiettivo importante o il soddisfacimento di un desiderio a lungo atteso, come la nascita di un figlio o il conseguimento di una laurea.
Anche eventi negativi, come la perdita di una persona cara, possono causare questo tipo di emozione, poiché rappresentano un ostacolo a un obiettivo affettivo.
2. Commuoversi "Altri-Centrati" (Empatici)
Emozioni Vicariose:
Questo tipo di emozione è scatenato da eventi positivi o negativi che accadono ad altri. Si prova empatia per le esperienze altrui, come il successo di un parente o la tragedia di una persona cara.
In questi casi, è fondamentale provare un sincero interesse per l'altro, spesso derivante dall'empatia o dall'affetto.
3. Commuoversi dalla "Bontà del Mondo" (Etico)
Azione Altruistica e Affetto:
Si è commossi nel vedere azioni altruistiche o affettive tra persone, che siano rivolte a sé stessi o a terzi. Questo include anche azioni generose compiute da persone meno potenti o eventi inaspettati di solidarietà.
4. Commuoversi dalla "Bellezza del Mondo" (Estetico)
Piacere Estetico:
Questo tipo di emozione è scatenato dal godimento di bellezze estetiche, come un paesaggio sublime, una musica emozionante o un'opera d'arte. In questi momenti, si può sperimentare un piacere percettivo e spirituale.
Il Ruolo dell'Empatia
L'empatia è strettamente legata al concetto di "commuoversi". È descritta come un'esperienza emotiva intensa derivante dall'identificazione o dalla vicinanza con le esperienze degli altri.
Funzione del Sentimento di "Commuoversi"
Il sentimento di "commuoversi" svolge diverse funzioni importanti nella nostra vita emotiva:

Conferma dell'Immagine di Sé:

Rafforza la nostra identità e ci fa sentire soddisfatti di noi stessi quando raggiungiamo obiettivi significativi o desiderati.
Attaccamento:

Mantiene e rafforza il legame affettivo con le persone amate, ricordandoci l'amore dato e ricevuto.
Altruismo:

Riafferma l'importanza di prendersi cura degli altri e delle loro necessità, evidenziando la nostra speranza in un mondo buono.
Valorizzazione della Bellezza:

Ci ricorda quanto sia importante la bellezza nella nostra vita, fornendo emozioni forti quando ne godiamo.
"""

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
def find_similar_chunks(query, index, chunks, top_k=3):
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
    "{istruzioni}\n\nSei un detector di commozioni. Valuta se il seguente episodio è un caso di commozione o no: \n\nEpisodio: {question} \n\nContesto simile: {context} \n\nFornisci una risposta con 'Sì' se è un caso di commozione, altrimenti rispondi 'No'. Inoltre, fornisci una spiegazione dettagliata del motivo per cui l'episodio è o non è un caso di commozione."
)

# Funzione per analizzare il sentiment dell'episodio utilizzando il modello e il prompt
def sentiment_analysis_with_context(llama_model, episode, similar_chunks):
    # Prepara il contesto dai chunk simili
    context = "\n".join(similar_chunks)

    # Creare il prompt con il contesto, l'episodio, e le istruzioni
    prompt = prompt_template.format(istruzioni=istruzioni, question=episode, context=context)
    
    # Esegui il modello LLaMA
    response = llama_model(prompt)

    # Estrai il testo dalla risposta del modello
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Analizza la risposta del modello per determinare il sentiment
    return response_text.strip()

# Configurazione dell'interfaccia Streamlit
def main():
    st.title("Analisi di Commozione degli Episodi")

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
            st.write(f"**Similar Chunk {i}:**")
            st.write(chunk)
            st.write("\n" + "-"*50 + "\n")

