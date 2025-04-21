import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import os
import time

def test_bioclinicalbert_with_csv():
    """Simple test to verify Bio_ClinicalBERT works with the clinical_processed.csv file"""
    print("=== Bio_ClinicalBERT Simple Test with clinical_processed.csv ===")
    start_time = time.time()
    
    try:
        # Load the CSV file
        csv_path = "dataset/clinical_processed.csv"
        print(f"\nAttempting to load: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f" CSV loaded successfully! Shape: {df.shape}")
        
        # Display file information
        print("\nDataset information:")
        print(f"- Columns: {df.columns.tolist()}")
        print(f"- Sample rows: {len(df)}")
        
        # Find a text column
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':  # Check for string columns
                # Check if column contains strings by examining first non-null value
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample_val, str) and len(sample_val) > 20:
                    text_columns.append(col)
                    if len(df[col].dropna().iloc[0]) > 100:  # Prefer longer text columns
                        text_columns.insert(0, col)  # Put longer text columns first
        
        if not text_columns:
            print("⚠ Warning: No suitable text column found in the CSV file")
            # Create a dummy text column by combining other columns
            df['combined_text'] = df.astype(str).apply(lambda x: ' '.join(x.values), axis=1)
            text_column = 'combined_text'
        else:
            text_column = text_columns[0]
        
        print(f"\nUsing column '{text_column}' as the text source")
        
        # Display sample texts
        print("\nSample texts from the CSV file:")
        for i in range(min(3, len(df))):
            sample_text = str(df[text_column].iloc[i])
            print(f"[{i+1}] {sample_text[:100]}..." if len(sample_text) > 100 else f"[{i+1}] {sample_text}")
        
        # Load Bio_ClinicalBERT model and tokenizer
        print("\nLoading Bio_ClinicalBERT model and tokenizer...")
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print(" Model loaded successfully!")
        
        # Test tokenization and model with a sample text
        sample_index = 0
        sample_text = str(df[text_column].iloc[sample_index])
        print(f"\nTokenizing sample text: \"{sample_text[:50]}...\"")
        
        # Tokenize
        inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512)
        print(f" Text tokenized: {len(inputs['input_ids'][0])} tokens")
        
        # Run through model
        print("\nRunning sample through Bio_ClinicalBERT...")
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get embedding
        embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        print(f" Generated embedding of size: {embedding.shape}")
        print(f"Embedding sample values: {embedding[0, :5].tolist()}")
        
        # Test batch processing with a few samples
        print("\nTesting batch processing...")
        batch_size = min(5, len(df))
        batch_texts = df[text_column].astype(str).iloc[:batch_size].tolist()
        
        # Tokenize batch
        batch_inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                                 return_tensors="pt", max_length=512)
        
        # Run batch through model
        with torch.no_grad():
            batch_outputs = model(**batch_inputs)
        
        batch_embeddings = batch_outputs.last_hidden_state[:, 0, :]
        print(f" Generated {batch_size} embeddings of shape: {batch_embeddings.shape}")
        
        total_time = time.time() - start_time
        print(f"\n Test completed successfully in {total_time:.2f} seconds!")
        print("Bio_ClinicalBERT can successfully process your clinical_processed.csv file.")
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTest failed. Please check the error message above.")

if __name__ == "__main__":
    test_bioclinicalbert_with_csv()