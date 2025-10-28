"""Test suite for Similarity module.

Tests text preprocessing, similarity calculation, and result analysis
for the eye-tracking data generation project.
"""

import pandas as pd
import numpy as np

# Import similarity functions (same folder, no path needed)
from simularity import (
    preprocess_text,
    extract_category_from_filename,
    parse_json_sentences,
    parse_csv_sentences,
    calculate_tfidf_similarity,
    build_similarity_results,
    calculate_similarity_statistics
)


def test_preprocess_text():
    """Test text preprocessing."""
    print("Testing preprocess_text...")
   
    assert preprocess_text("Hello World!") == "hello world"
    assert preprocess_text("Test, with punctuation.") == "test with punctuation"
    assert preprocess_text("  Multiple   spaces  ") == "multiple spaces"
    assert preprocess_text(None) == ""
    assert preprocess_text("") == ""
   
    print("✓ preprocess_text tests passed!")


def test_extract_category_from_filename():
    """Test category extraction from filename."""
    print("\nTesting extract_category_from_filename...")
   
    assert extract_category_from_filename("single_test.jpg") == "single"
    assert extract_category_from_filename("comp_experiment.png") == "competition"
    assert extract_category_from_filename("coop_data.jpg") == "cooperation"
    assert extract_category_from_filename("random_file.txt") == "unknown"
    assert extract_category_from_filename("") == "unknown"
   
    print("✓ extract_category_from_filename tests passed!")


def test_parse_json_sentences():
    """Test parsing JSON sentences."""
    print("\nTesting parse_json_sentences...")
   
    test_data = [
        {
            "image1": "single_01.png",
            "image2": "single_02.png",
            "class": "This is a test description."
        },
        {
            "image1": "comp_01.png",
            "image2": "comp_02.png",
            "class": "Another test sentence."
        }
    ]
   
    result = parse_json_sentences(test_data, "test_json")
   
    assert len(result) == 2
    assert result[0]["source"] == "json"
    assert result[0]["file"] == "test_json"
    assert result[0]["category"] == "single"
    assert result[0]["text"] == "This is a test description."
    assert "processed" in result[0]
   
    print("✓ parse_json_sentences tests passed!")


def test_parse_csv_sentences():
    """Test parsing CSV sentences."""
    print("\nTesting parse_csv_sentences...")
   
    test_df = pd.DataFrame({
        "category": ["single", "competition"],
        "style": ["formal", "lively"],
        "strategy": ["strategy_1", "strategy_2"],
        "root_dir": ["dir1", "dir2"],
        "generated": ["First generated text.", "Second generated text."]
    })
   
    result = parse_csv_sentences(test_df)
   
    assert len(result) == 2
    assert result[0]["source"] == "csv"
    assert result[0]["category"] == "single"
    assert result[0]["style"] == "formal"
    assert result[0]["text"] == "First generated text."
    assert "processed" in result[0]
   
    print("✓ parse_csv_sentences tests passed!")


def test_calculate_tfidf_similarity():
    """Test TF-IDF similarity calculation."""
    print("\nTesting calculate_tfidf_similarity...")
   
    json_sentences = [
        {
            "id": "json_0",
            "processed": "the cat sat on the mat"
        },
        {
            "id": "json_1",
            "processed": "the dog played in the park"
        }
    ]
   
    csv_sentences = [
        {
            "id": "csv_0",
            "processed": "a cat was sitting on a mat"
        },
        {
            "id": "csv_1",
            "processed": "a dog was playing in a park"
        }
    ]
   
    similarity_matrix = calculate_tfidf_similarity(json_sentences, csv_sentences)
   
    assert similarity_matrix.shape == (2, 2)
    assert similarity_matrix[0, 0] > similarity_matrix[0, 1]  # Cat sentence more similar to cat
    assert similarity_matrix[1, 1] > similarity_matrix[1, 0]  # Dog sentence more similar to dog
    assert np.all(similarity_matrix >= 0) and np.all(similarity_matrix <= 1)
   
    print("✓ calculate_tfidf_similarity tests passed!")


def test_build_similarity_results():
    """Test building similarity results."""
    print("\nTesting build_similarity_results...")
   
    similarity_matrix = np.array([[0.8, 0.3], [0.2, 0.9]])
   
    json_sentences = [
        {
            "id": "json_0",
            "file": "formal",
            "category": "single",
            "text": "JSON text 1"
        },
        {
            "id": "json_1",
            "file": "lively",
            "category": "competition",
            "text": "JSON text 2"
        }
    ]
   
    csv_sentences = [
        {
            "id": "csv_0",
            "category": "single",
            "style": "formal",
            "strategy": "strategy_1",
            "root_dir": "dir1",
            "text": "CSV text 1"
        },
        {
            "id": "csv_1",
            "category": "competition",
            "style": "lively",
            "strategy": "strategy_2",
            "root_dir": "dir2",
            "text": "CSV text 2"
        }
    ]
   
    results = build_similarity_results(similarity_matrix, json_sentences, csv_sentences)
   
    assert len(results) == 4  # 2x2 matrix
    assert results[0]["json_id"] == "json_0"
    assert results[0]["csv_id"] == "csv_0"
    assert results[0]["similarity"] == 0.8
    assert "json_text" in results[0]
    assert "csv_text" in results[0]
   
    print("✓ build_similarity_results tests passed!")


def test_calculate_similarity_statistics():
    """Test similarity statistics calculation."""
    print("\nTesting calculate_similarity_statistics...")
   
    test_df = pd.DataFrame({
        "similarity": [0.9, 0.6, 0.3, 0.8, 0.5, 0.2, 0.7, 0.4]
    })
   
    stats = calculate_similarity_statistics(test_df)
   
    assert "mean" in stats
    assert "max" in stats
    assert "min" in stats
    assert "high_sim_count" in stats
    assert "medium_sim_count" in stats
    assert "low_sim_count" in stats
   
    assert stats["max"] == 0.9
    assert stats["min"] == 0.2
    assert stats["high_sim_count"] == 3  # >0.7: 0.9, 0.8, 0.7
    assert stats["medium_sim_count"] == 3  # 0.4-0.7: 0.6, 0.5, 0.4
    assert stats["low_sim_count"] == 2  # <0.4: 0.3, 0.2
   
    print("✓ calculate_similarity_statistics tests passed!")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Similarity Module Tests")
    print("=" * 50)
   
    test_preprocess_text()
    test_extract_category_from_filename()
    test_parse_json_sentences()
    test_parse_csv_sentences()
    test_calculate_tfidf_similarity()
    test_build_similarity_results()
    test_calculate_similarity_statistics()
   
    print("\n" + "=" * 50)
    print("✓ All tests passed successfully!")
    print("=" * 50)
