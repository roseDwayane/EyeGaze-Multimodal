import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import json
import pandas as pd
from datetime import datetime
from Result.similarity.similarity_utils import (
    parse_json_sentences,
    parse_csv_sentences,
    calculate_tfidf_similarity,
    build_similarity_results,
    calculate_similarity_statistics
)


def main():
    """Execute the complete similarity calculation pipeline.
   
    Main execution function that:
    1. Loads reference descriptions from JSON files
    2. Loads generated descriptions from CSV file
    3. Calculates TF-IDF based cosine similarity
    4. Saves detailed results to CSV
    5. Displays statistical summary
   
    The output CSV contains all pairwise similarity scores between reference
    and generated descriptions, enabling detailed analysis of the vision-language
    model's description generation quality.
   
    Args:
        None
   
    Returns:
        None
    """
   
    JSON_FILES = [
        "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/formal_description_json.json",
        "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/lively_description_json.json",
        "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/json_textdescription/mixed_description_json.json"
    ]
   
    CSV_FILE = "/home/cnelabai/PycharmProjects/EyeGaze-Multimodal/natural_generation_results_20250908_023610.csv"
   
    print("=== Sentence Similarity Calculation ===")
    print("Calculating JSON vs CSV similarity")
    print("="*40)
   
    # Load JSON sentences
    json_sentences = []
    for json_file in JSON_FILES:
        print(f"Loading {json_file}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            json_name = os.path.basename(json_file).replace('_description_json.json', '')
            sentences = parse_json_sentences(data, json_name)
            json_sentences.extend(sentences)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
   
    print(f"Loaded {len(json_sentences)} sentences from JSON files")
   
    # Load CSV sentences
    print(f"Loading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
        csv_sentences = parse_csv_sentences(df)
        print(f"Loaded {len(csv_sentences)} generated sentences from CSV file")
    except Exception as e:
        print(f"Error loading {CSV_FILE}: {e}")
        return
   
    if len(json_sentences) == 0 or len(csv_sentences) == 0:
        print("No sentences found, please check file paths")
        return
   
    print(f"\nCalculating similarity for {len(json_sentences)} JSON sentences vs {len(csv_sentences)} CSV sentences")
   
    # Calculate similarity
    print("Calculating TF-IDF vectors...")
    similarity_matrix = calculate_tfidf_similarity(json_sentences, csv_sentences)
    print("Calculating cosine similarity...")
   
    # Build results
    results = build_similarity_results(similarity_matrix, json_sentences, csv_sentences)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('similarity', ascending=False)
   
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"sentence_similarities_{timestamp}.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8')
   
    print(f"\nSimilarity results saved to: {output_file}")
   
    # Calculate and display statistics
    stats = calculate_similarity_statistics(results_df)
   
    print(f"\n=== Statistics Summary ===")
    print(f"JSON sentences: {len(json_sentences)}")
    print(f"CSV sentences: {len(csv_sentences)}")
    print(f"Total comparisons: {len(results)}")
    print(f"Average similarity: {stats['mean']:.4f}")
    print(f"Max similarity: {stats['max']:.4f}")
    print(f"Min similarity: {stats['min']:.4f}")
    print(f"High similarity (>0.7): {stats['high_sim_count']} pairs")
    print(f"Medium similarity (0.4-0.7): {stats['medium_sim_count']} pairs")
    print(f"Low similarity (<0.4): {stats['low_sim_count']} pairs")
   
    # Display top 10
    print(f"\n=== Top 10 Most Similar Pairs ===")
    for idx, row in results_df.head(10).iterrows():
        print(f"{idx+1}. Similarity: {row['similarity']:.4f}")
        print(f"   JSON({row['json_file']}-{row['json_category']}): {row['json_text'][:60]}...")
        print(f"   CSV({row['csv_category']}-{row['csv_style']}): {row['csv_text'][:60]}...")
        print()
   
    print(f"\n=== Complete ===")


if __name__ == "__main__":
    main()
