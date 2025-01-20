import umap
import pandas as pd
from sentence_transformers import SentenceTransformer


def generate_add_embeddings(df):
    """
    Generates and adds text embeddings for movie titles using a pre-trained transformer model.

    Parameters:
    -----------
    df (DataFrame): Input DataFrame containing a 'primaryTitle' column with movie titles.

    The function performs the following steps:
    1. Uses the 'all-MiniLM-L6-v2' model from Sentence Transformers to generate embeddings for movie titles.
    2. Applies UMAP dimensionality reduction to reduce embeddings to 10 components.
    3. Converts the reduced embeddings into a DataFrame and merges them with the original DataFrame.

    Returns:
    --------
    DataFrame: The input DataFrame with additional columns representing the reduced title embeddings.
    """
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(df['primaryTitle'].to_numpy())

    # Initialize UMAP reducer
    reducer = umap.UMAP(n_components=10)

    # Fit and transform
    embeddings_reduced = reducer.fit_transform(embeddings)

    # Convert embeddings to DataFrame
    embedding_columns = [f'title_emb_{i}' for i in range(embeddings_reduced.shape[-1])]
    embeddings_df = pd.DataFrame(embeddings_reduced, columns=embedding_columns)

    # Concatenate with the original DataFrame
    df = pd.concat([df, embeddings_df], axis=1)

    return df