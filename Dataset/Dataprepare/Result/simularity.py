import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text):
    """Preprocess text by cleaning and normalizing for similarity calculation.
   
    Removes special characters, extra whitespace, and converts to lowercase
    to prepare text for TF-IDF vectorization.
   
    Args:
        text (str or None): Input text string to preprocess.
   
    Returns:
        str: Preprocessed lowercase text with special characters removed
            and normalized whitespace. Returns empty string if input is None or NaN.
    """
    if pd.isna(text) or text is None:
        return ""
   
    text = str(text).strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
   
    return text


def extract_category_from_filename(filename):
    """Extract eye-tracking scenario category from filename.
   
    Identifies the experimental condition category (single, competition, cooperation)
    based on keyword patterns in the filename.
   
    Args:
        filename (str): The filename to analyze for category classification.
   
    Returns:
        str: Category name - "single" for single-agent scenarios,
            "competition" for competitive scenarios, "cooperation" for
            cooperative scenarios, or "unknown" if no pattern matches.
    """
    if not filename:
        return "unknown"
    filename = filename.lower()
    if "single" in filename:
        return "single"
    elif "comp" in filename:
        return "competition"
    elif "coop" in filename:
        return "cooperation"
    else:
        return "unknown"


def parse_json_sentences(data, json_name):
    """Parse sentences from JSON data containing reference descriptions.
   
    Extracts and processes text descriptions from JSON records, which contain
    reference descriptions for eye-tracking visualization images.
   
    Args:
        data (list): List of JSON records, each containing image paths and
            text descriptions.
        json_name (str): Name identifier for the JSON source file.
   
    Returns:
        list: List of dictionaries, each containing:
            - id (str): Unique identifier for the sentence
            - source (str): Source type, always "json"
            - file (str): Source file name
            - category (str): Experimental condition category
            - text (str): Original text description
            - processed (str): Preprocessed text for similarity calculation
    """
    sentences = []
   
    for item in data:
        if 'class' in item and item['class']:
            processed_text = preprocess_text(item['class'])
            if processed_text:
                sentences.append({
                    'id': f"json_{len(sentences)}",
                    'source': 'json',
                    'file': json_name,
                    'category': extract_category_from_filename(item.get('image1', '')),
                    'text': item['class'],
                    'processed': processed_text
                })
   
    return sentences


def parse_csv_sentences(df):
    """Parse generated sentences from CSV dataframe.
   
    Extracts generated descriptions produced by the vision-language model
    from a CSV file containing experimental results.
   
    Args:
        df (pd.DataFrame): DataFrame containing generated sentences with columns
            for category, style, strategy, root_dir, and generated text.
   
    Returns:
        list: List of dictionaries, each containing:
            - id (str): Unique identifier for the sentence
            - source (str): Source type, always "csv"
            - file (str): Source file identifier, always "generated"
            - category (str): Experimental condition category
            - style (str): Generation style used
            - strategy (str): Generation strategy used
            - root_dir (str): Root directory of source data
            - text (str): Generated text description
            - processed (str): Preprocessed text for similarity calculation
    """
    sentences = []
   
    for _, row in df.iterrows():
        if 'generated' in row and pd.notna(row['generated']):
            processed_text = preprocess_text(row['generated'])
            if processed_text:
                sentences.append({
                    'id': f"csv_{len(sentences)}",
                    'source': 'csv',
                    'file': 'generated',
                    'category': row.get('category', ''),
                    'style': row.get('style', ''),
                    'strategy': row.get('strategy', ''),
                    'root_dir': row.get('root_dir', ''),
                    'text': row.get('generated', ''),
                    'processed': processed_text
                })
   
    return sentences


def calculate_tfidf_similarity(json_sentences, csv_sentences):
    """Calculate cosine similarity between reference and generated sentences.
   
    Uses TF-IDF vectorization and cosine similarity to measure semantic
    similarity between reference descriptions and model-generated descriptions.
   
    Args:
        json_sentences (list): List of reference sentence dictionaries from JSON.
        csv_sentences (list): List of generated sentence dictionaries from CSV.
   
    Returns:
        numpy.ndarray: Similarity matrix of shape (len(json_sentences), len(csv_sentences))
            where each element [i,j] represents the cosine similarity between
            json_sentences[i] and csv_sentences[j]. Values range from 0 to 1.
    """
    all_processed = [s['processed'] for s in json_sentences] + [s['processed'] for s in csv_sentences]
   
    vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
   
    tfidf_matrix = vectorizer.fit_transform(all_processed)
   
    json_matrix = tfidf_matrix[:len(json_sentences)]
    csv_matrix = tfidf_matrix[len(json_sentences):]
   
    similarity_matrix = cosine_similarity(json_matrix, csv_matrix)
   
    return similarity_matrix


def build_similarity_results(similarity_matrix, json_sentences, csv_sentences):
    """Build detailed similarity results from similarity matrix.
   
    Constructs a comprehensive list of similarity comparisons between all
    reference and generated sentence pairs, preserving metadata for analysis.
   
    Args:
        similarity_matrix (numpy.ndarray): Similarity matrix from calculate_tfidf_similarity.
        json_sentences (list): List of reference sentence dictionaries.
        csv_sentences (list): List of generated sentence dictionaries.
   
    Returns:
        list: List of dictionaries, each containing:
            - json_id (str): Reference sentence identifier
            - csv_id (str): Generated sentence identifier
            - similarity (float): Cosine similarity score
            - json_file (str): Reference file name
            - json_category (str): Reference category
            - csv_category (str): Generated category
            - csv_style (str): Generation style
            - csv_strategy (str): Generation strategy
            - csv_root_dir (str): Source data directory
            - json_text (str): Reference text
            - csv_text (str): Generated text
    """
    results = []
   
    for i, json_sent in enumerate(json_sentences):
        for j, csv_sent in enumerate(csv_sentences):
            similarity = similarity_matrix[i][j]
           
            results.append({
                'json_id': json_sent['id'],
                'csv_id': csv_sent['id'],
                'similarity': similarity,
                'json_file': json_sent['file'],
                'json_category': json_sent['category'],
                'csv_category': csv_sent['category'],
                'csv_style': csv_sent.get('style', ''),
                'csv_strategy': csv_sent.get('strategy', ''),
                'csv_root_dir': csv_sent.get('root_dir', ''),
                'json_text': json_sent['text'],
                'csv_text': csv_sent['text']
            })
   
    return results


def calculate_similarity_statistics(results_df):
    """Calculate statistical metrics from similarity results.
   
    Computes summary statistics to evaluate the quality of generated descriptions
    by analyzing their similarity distribution to reference descriptions.
   
    Args:
        results_df (pd.DataFrame): DataFrame containing similarity results with
            a 'similarity' column.
   
    Returns:
        dict: Dictionary containing statistical metrics:
            - mean (float): Average similarity score
            - max (float): Maximum similarity score
            - min (float): Minimum similarity score
            - high_sim_count (int): Number of pairs with similarity > 0.7
            - medium_sim_count (int): Number of pairs with similarity 0.4-0.7
            - low_sim_count (int): Number of pairs with similarity < 0.4
    """
    stats = {
        'mean': results_df['similarity'].mean(),
        'max': results_df['similarity'].max(),
        'min': results_df['similarity'].min(),
        'high_sim_count': len(results_df[results_df['similarity'] >= 0.7]),
        'medium_sim_count': len(results_df[(results_df['similarity'] >= 0.4) & (results_df['similarity'] < 0.7)]),
        'low_sim_count': len(results_df[results_df['similarity'] < 0.4])
    }
   
    return stats
