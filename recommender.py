"""
Movie recommendation engine.

Loads the netflix dataset, builds sentence embeddings for each title,
stores them in a FAISS index, and finds the closest matches to whatever
the user types in. Pretty straightforward content-based filtering.
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class RecommendationEngine:
    """
    The actual brains of the app. Takes in the CSV path, builds an embedding
    for every movie/show by mashing together its metadata into one big string,
    then uses FAISS to do fast nearest-neighbor lookups when a user query comes in.
    """

    def __init__(self, csv_path, model_name="all-MiniLM-L6-v2"):
        self.csv_path = csv_path
        self.model_name = model_name
        self.model = None
        self.index = None
        self.df = None
        self.embeddings = None

    def load_data(self):
        """Read the CSV and clean up the messy bits (missing values, etc)."""
        self.df = pd.read_csv(self.csv_path)

        # fill in blanks so we dont crash when concatenating strings later
        fill_cols = ["title", "director", "cast", "country", "listed_in", "description", "rating", "type"]
        for col in fill_cols:
            self.df[col] = self.df[col].fillna("")

        # make release_year a string too, easier to work with
        self.df["release_year"] = self.df["release_year"].fillna(0).astype(int).astype(str)

        # build one combined text blob per row that captures everything useful about the title
        # weighting title and genres heavier by repeating them
        self.df["combined_text"] = self.df.apply(self._build_text_feature, axis=1)

        print(f"Loaded {len(self.df)} titles from {self.csv_path}")

    def _build_text_feature(self, row):
        """
        Smash together all the relevant fields into one string.
        Title and genres get repeated so they carry more weight in the embedding.
        """
        parts = []

        # title matters most, repeat it so the embedding picks up on it
        if row["title"]:
            parts.append(row["title"])
            parts.append(row["title"])

        # type is useful (Movie vs TV Show)
        if row["type"]:
            parts.append(row["type"])

        # genres are super important for recommendations
        if row["listed_in"]:
            parts.append(row["listed_in"])
            parts.append(row["listed_in"])

        # director can help with "movies by Scorsese" type queries
        if row["director"]:
            parts.append(f"directed by {row['director']}")

        # cast helps with "movies with Adam Sandler" type stuff
        if row["cast"]:
            # just grab first few actors, no need for the entire cast list
            actors = [a.strip() for a in row["cast"].split(",")][:5]
            parts.append(f"starring {', '.join(actors)}")

        # country can be useful for "Indian movies" or "Korean dramas"
        if row["country"]:
            parts.append(f"from {row['country']}")

        # year for "90s movies" type queries
        if row["release_year"] and row["release_year"] != "0":
            parts.append(f"released in {row['release_year']}")

        # rating helps with "kids movies" (G, PG) vs "mature content" (R, TV-MA)
        if row["rating"]:
            parts.append(f"rated {row['rating']}")

        # description is the meat of it
        if row["description"]:
            parts.append(row["description"])

        return ". ".join(parts)

    def build_index(self):
        """
        Encode every title into a vector and throw them all into a FAISS index.
        This is the expensive part but it only happens once on startup.
        """
        print(f"Loading sentence-transformers model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        print("Encoding all titles... this takes a minute on first run")
        texts = self.df["combined_text"].tolist()
        self.embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=128)

        # normalize for cosine similarity (FAISS inner product on unit vectors = cosine sim)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms

        # build the FAISS index using inner product (which equals cosine sim after normalization)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype(np.float32))

        print(f"FAISS index built with {self.index.ntotal} vectors, dimension={dimension}")

    def initialize(self):
        """One-shot setup: load data, build embeddings, create index."""
        self.load_data()
        self.build_index()
        print("Ready to go!")

    def recommend(self, query, top_k=10):
        """
        Take a free-form text query, embed it, and find the closest titles.
        Returns a list of dicts with all the relevant info about each match.
        """
        if not self.index or not self.model:
            raise RuntimeError("Engine not initialized yet, call initialize() first")

        # encode the query the same way we encoded the titles
        query_vec = self.model.encode([query])
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        # search the index
        scores, indices = self.index.search(query_vec.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 when there arent enough results
                continue

            row = self.df.iloc[idx]
            results.append({
                "title": row["title"],
                "type": row["type"],
                "director": row["director"] if row["director"] else "N/A",
                "cast": row["cast"] if row["cast"] else "N/A",
                "country": row["country"] if row["country"] else "N/A",
                "release_year": row["release_year"],
                "rating": row["rating"] if row["rating"] else "N/A",
                "duration": row["duration"] if pd.notna(row["duration"]) else "N/A",
                "genres": row["listed_in"] if row["listed_in"] else "N/A",
                "description": row["description"],
                "score": round(float(score), 4),
            })

        return results
