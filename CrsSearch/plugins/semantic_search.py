# plugins/semantic_search.py
from datasette import hookimpl
from sentence_transformers import SentenceTransformer
import json

# Load the model once when Datasette starts
# (Using the same model name as your indexer)
model = SentenceTransformer('all-MiniLM-L6-v2')

@hookimpl
def prepare_connection(conn):
    """
    Registers the 'embed_text' function in SQLite so you can use it in SQL queries.
    """
    def embed_text(text):
        if not text:
            return None
        # Encode and convert to a JSON string (e.g. "[0.1, 0.2, ...]")
        # sqlite-vec understands JSON arrays automatically.
        vector = model.encode(text)
        return json.dumps(vector.tolist())

    conn.create_function("embed_text", 1, embed_text)