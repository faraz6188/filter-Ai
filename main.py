"""
MyronPromos
Nick Feuer
nfeuer@myron.com
05-28-2025
"""

import os
from google import genai
from google.genai import types as genai_types
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import sys
import time

# --- Configuration ---
CSV_FILE_PATH = 'c:\\Users\\Nick\\Downloads\\products_for_embedding.csv'
GEMINI_EMBEDDING_MODEL_NAME = "gemini-embedding-exp-03-07" # From your example
EMBEDDINGS_NPY_FILE = 'product_embeddings.npy'
PRODUCT_DATA_JSON_FILE = 'product_data.json'
EMBEDDING_BATCH_SIZE = 50  # Number of texts to process in one API call (max is often 100-200, 50 is safe)
DELAY_BETWEEN_BATCHES = 60 # Seconds to wait between batch API calls (e.g., 0.5 for 120 req/min theoretical max)

# --- Global Variables ---
product_data = []
product_embeddings = None
gemini_client = None

# --- Helper Functions ---
def parse_text_for_details(text_for_embedding_string):
    """
    Parses the 'text_for_embedding' string to extract product name, primary category, and URL ending.
    """
    product_name = "Unknown Product"
    primary_category = "Unknown Category"
    product_url_ending = ""
    
    # Regex to capture the main parts based on the provided examples
    # It looks for ". category:", ". url:", and ". description:" as delimiters
    match = re.match(r"([\s\S]+?)\.\s*category:\s*([\s\S]+?)\.\s*url:\s*([\s\S]+?)\.\s*description:", text_for_embedding_string, re.IGNORECASE | re.DOTALL)

    if match:
        product_name = match.group(1).strip()
        category_string = match.group(2).strip()
        product_url_ending = match.group(3).strip()

        if category_string:
            categories = [cat.strip() for cat in category_string.split(',')]
            if categories:
                first_cat = categories[0]
                # Clean up common suffixes like " - l1" or ", nan"
                # These are the corrected regex lines:
                first_cat = re.sub(r'\s*-\s*l\d+$', '', first_cat, flags=re.IGNORECASE).strip()
                first_cat = re.sub(r',\s*nan$', '', first_cat, flags=re.IGNORECASE).strip() 
                primary_category = first_cat if first_cat else "Unknown Category"
    else:
        # Fallback if regex doesn't match
        print(f"Warning: Could not parse details using main regex for text: '{text_for_embedding_string[:70]}...'")
        # Basic fallback for product name if the full structure isn't there
        parts = text_for_embedding_string.split('.')
        if len(parts) > 0: 
            product_name = parts[0].strip()
        # You might add more sophisticated fallback parsing here if needed

    return {
        "name": product_name,
        "category": primary_category,
        "url_ending": product_url_ending,
        "full_text": text_for_embedding_string
    }

def initialize_gemini_client():
    global gemini_client
    api_key_to_use = 'AIzaSyD-pwjVilEFhZ_lOzLJAO4RgSoNYh3rbNw'
    try:
        api_key_env = 'AIzaSyD-pwjVilEFhZ_lOzLJAO4RgSoNYh3rbNw'
        if not api_key_env:
            api_key_env = os.environ.get("GEMINI_API_KEY")
            if api_key_env: print("Found GEMINI_API_KEY environment variable.")
        
        if api_key_env:
            print(f"Using API key from environment variable ('{ 'GOOGLE_API_KEY' if os.environ.get('GOOGLE_API_KEY') else 'GEMINI_API_KEY'}').")
            api_key_to_use = api_key_env
        else:
            print("Neither GOOGLE_API_KEY nor GEMINI_API_KEY environment variable found.")
            api_key_input = input("Please enter your API Key: ").strip()
            if not api_key_input: print("No API Key provided."); return False
            api_key_to_use = api_key_input
        
        gemini_client = genai.Client(api_key=api_key_to_use)
        print("Successfully initialized genai.Client with API key.")
        try:
            print("Attempting to list a few models to confirm client is working...")
            models_iterable = gemini_client.models.list()
            count = 0
            for model_obj in models_iterable:
                if count < 2:
                    model_name_to_print = getattr(model_obj, 'name', f"ID: {getattr(model_obj, 'id', 'N/A')}")
                    print(f"- Found model: {model_name_to_print}")
                count += 1;  _ = count >=2 and next(iter([]), None) # Terse break
            if count > 0: print("Model listing test successful.")
            else: print("No models listed by client.models.list().")
        except AttributeError: print("Note: client.models.list() method not found or changed. Skipping model listing test.")
        except Exception as e_list: print(f"Note: Error during model listing test: {e_list}.")
        return True
    except Exception as e_general: print(f"Error in initialize_gemini_client: {e_general}"); traceback.print_exc(); return False


# MODIFIED load_and_embed_data function
def load_and_embed_data():
    global product_data, product_embeddings, gemini_client
    if not gemini_client:
        print("Error: Gemini client not initialized."); initialize_gemini_client(); return False

    print(f"Loading data from {CSV_FILE_PATH}...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        # Pre-parse all details to avoid parsing inside the loop repeatedly if possible
        # Or, ensure original texts are easily accessible by ID for product_data construction.
        
        all_original_texts = df['text_for_embedding'].tolist()
        all_ids = df['id'].tolist()
        
        # Create a mapping from ID to original text and parsed details for quick lookup
        id_to_data_map = {}
        for i, item_id in enumerate(all_ids):
            original_text = all_original_texts[i]
            if not isinstance(original_text, str) or not original_text.strip():
                 print(f"Warning: Skipping empty or invalid text for ID {item_id} during initial mapping.")
                 continue
            id_to_data_map[item_id] = {
                "original_text": original_text,
                "details": parse_text_for_details(original_text)
            }

        # Filter out IDs that had invalid text
        valid_ids_for_embedding = [id_val for id_val in all_ids if id_val in id_to_data_map]
        texts_for_embedding_api = [id_to_data_map[id_val]["original_text"] for id_val in valid_ids_for_embedding]
        
        total_valid_texts = len(texts_for_embedding_api)
        if total_valid_texts == 0:
            print("No valid texts found to embed after initial processing.")
            return False

        print(f"Generating embeddings for {total_valid_texts} valid texts in batches of {EMBEDDING_BATCH_SIZE} using model '{GEMINI_EMBEDDING_MODEL_NAME}'...")
        
        # Clear global lists before populating
        product_data.clear() 
        retrieved_embeddings_list = [] # Temporary list for embeddings

        for i in range(0, total_valid_texts, EMBEDDING_BATCH_SIZE):
            current_batch_texts = texts_for_embedding_api[i : i + EMBEDDING_BATCH_SIZE]
            current_batch_ids = valid_ids_for_embedding[i : i + EMBEDDING_BATCH_SIZE]

            if not current_batch_texts: continue # Should not happen if total_valid_texts > 0

            print(f"Processing batch {i // EMBEDDING_BATCH_SIZE + 1}/{(total_valid_texts + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE} (texts {i+1}-{min(i + EMBEDDING_BATCH_SIZE, total_valid_texts)} of {total_valid_texts})...")
            
            try:
                response = gemini_client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL_NAME,
                    contents=current_batch_texts, # Pass the list of texts for this batch
                    config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                
                if response.embeddings and len(response.embeddings) == len(current_batch_texts):
                    for k, embedding_obj in enumerate(response.embeddings):
                        item_id = current_batch_ids[k]
                        data_for_id = id_to_data_map[item_id]
                        
                        if hasattr(embedding_obj, 'values'):
                            retrieved_embeddings_list.append(embedding_obj.values)
                            # Append corresponding product data
                            product_data.append({
                                "id": item_id,
                                "name": data_for_id["details"]["name"],
                                "category": data_for_id["details"]["category"],
                                "url": f"www.myron.com{data_for_id['details']['url_ending']}",
                                "original_text": data_for_id["original_text"]
                            })
                        else:
                            print(f"Warning: Embedding object for ID {item_id} in batch does not have '.values'. Skipping.")
                else:
                    print(f"Warning: Embedding response for batch did not return expected number of embeddings. Skipping batch.")
                    print(f"Batch texts count: {len(current_batch_texts)}, Embeddings received: {len(response.embeddings) if response.embeddings else 0}")

                # Throttling: Wait after each successful batch API call
                if i + EMBEDDING_BATCH_SIZE < total_valid_texts: # Don't sleep after the very last batch
                    print(f"Batch processed. Sleeping for {DELAY_BETWEEN_BATCHES} seconds...")
                    time.sleep(DELAY_BETWEEN_BATCHES)

            except Exception as e:
                print(f"Error embedding batch starting with text for ID {current_batch_ids[0]}: {e}"); traceback.print_exc()
                print(f"This batch will be skipped. Consider reducing EMBEDDING_BATCH_SIZE or increasing DELAY_BETWEEN_BATCHES if it's a rate limit issue.")
                # If a rate limit error occurs within a batch, you might want a longer specific sleep.
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    print("Rate limit error encountered. Sleeping for 10 seconds...")
                    time.sleep(10)
        
        if not retrieved_embeddings_list:
            print("Error: No embeddings were successfully generated after processing all batches."); return False
        
        product_embeddings = np.array(retrieved_embeddings_list, dtype=object)
        try:
            product_embeddings = np.array(retrieved_embeddings_list).astype(float)
        except ValueError:
            print("Warning: Could not convert all retrieved embeddings to a uniform float numpy array.")

        if product_embeddings.size == 0 or not product_data:
            print("Error: Batch embedding resulted in empty data (no successful embeddings or product_data)."); return False
        if len(product_data) != product_embeddings.shape[0]:
            print(f"Error: Mismatch between number of embeddings ({product_embeddings.shape[0]}) and product data entries ({len(product_data)}).")
            return False
            
        print(f"Data loaded. Embeddings shape: {product_embeddings.shape}, Products collected: {len(product_data)}")
        return True
        
    except Exception as e:
        print(f"Critical error in load_and_embed_data: {e}"); traceback.print_exc(); return False


# --- Flask Application (search_suggestions_api remains the same) ---
app = Flask(__name__)
CORS(app)
@app.route('/search', methods=['POST'])
def search_suggestions_api():
    global product_embeddings, product_data, gemini_client # gemini_client is global

    # --- ADD THIS BLOCK TO ENSURE CLIENT IS INITIALIZED ---
    if not gemini_client:
        print("Search API: gemini_client is None. Attempting to re-initialize...")
        if not initialize_gemini_client(): # This function sets the global gemini_client
            print("Search API: Failed to re-initialize Gemini client during request.")
            return jsonify({"error": "Gemini client could not be initialized for search."}), 500
        if not gemini_client: # Check again, in case initialize_gemini_client failed silently (it shouldn't)
            print("Search API: Critical error - Gemini client is still None after re-initialization attempt.")
            return jsonify({"error": "Critical: Gemini client unavailable for search."}), 500
        print("Search API: Gemini client was re-initialized successfully during request.")
    # --- END OF ADDED BLOCK ---

    # The rest of your function remains the same
    if product_embeddings is None or not product_data or product_embeddings.size == 0:
        return jsonify({"error": "Embeddings/data not loaded or empty."}), 500

    query_data = request.get_json();
    if not query_data or 'query' not in query_data: return jsonify({"error": "Missing 'query'"}), 400
    search_query = query_data['query']
    if not search_query.strip(): return jsonify({"categories": [], "products": []})

    try:
        response = gemini_client.models.embed_content( # Uses the (potentially re-initialized) gemini_client
            model=GEMINI_EMBEDDING_MODEL_NAME,
            contents=[search_query], 
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        # ... (rest of your try block: processing embeddings, similarity search, etc.) ...
        if response.embeddings and len(response.embeddings) > 0 and hasattr(response.embeddings[0], 'values'):
            query_embedding = np.array(response.embeddings[0].values).reshape(1, -1)
        else:
            print(f"Error: Unexpected embedding response for query '{search_query}'.")
            return jsonify({"error": "Failed to generate query embedding."}), 500
        
        if product_embeddings.ndim == 1: 
            if product_embeddings.shape[0] > 0 and len(product_embeddings.shape) > 1 and product_embeddings.shape[1] > 0 :
                 pass 
            elif product_embeddings.shape[0] > 0 and len(product_embeddings.shape) == 1: 
                 product_embeddings = product_embeddings.reshape(1, -1) 
            else:
                 print("Warning: product_embeddings is 1D and malformed or empty."); return jsonify({"error": "No product data for search."}),500

        similarities = cosine_similarity(query_embedding, product_embeddings)[0]
        N_RESULTS_TO_CONSIDER = 15; MAX_CATEGORIES = 3; MAX_PRODUCTS = 3
        top_n_indices = np.argsort(similarities)[-N_RESULTS_TO_CONSIDER:][::-1]
        suggested_categories_dict = {}; suggested_products_list = []; seen_product_names = set()
        for index in top_n_indices:
            if index < len(product_data):
                item = product_data[index]; similarity_score = float(similarities[index])
                category_name = item.get('category', 'UC');
                if category_name != "UC" and category_name not in suggested_categories_dict and len(suggested_categories_dict) < MAX_CATEGORIES:
                    suggested_categories_dict[category_name] = similarity_score
                product_name = item.get('name', 'UP')
                if product_name != "UP" and product_name not in seen_product_names and len(suggested_products_list) < MAX_PRODUCTS:
                    suggested_products_list.append({"name": product_name, "url": item.get('url', '#'), "similarity": similarity_score})
                    seen_product_names.add(product_name)
            if len(suggested_categories_dict) >= MAX_CATEGORIES and len(suggested_products_list) >= MAX_PRODUCTS: break
        final_categories = list(suggested_categories_dict.keys())
        return jsonify({"categories": final_categories, "products": suggested_products_list})
    except Exception as e: 
        print(f"Error during search for '{search_query}': {e}"); traceback.print_exc()
        return jsonify({"error": "Search error."}), 500
    
# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- AI Search POC Backend (with Caching & Batching) ---")
    print(f"Python Version: {sys.version.split()[0]} on {sys.platform}")
    # ... (SDK module path printing) ...

    # Attempt to load processed data from files first
    loaded_from_files = False
    if os.path.exists(EMBEDDINGS_NPY_FILE) and os.path.exists(PRODUCT_DATA_JSON_FILE):
        print(f"\nFound existing data files: {EMBEDDINGS_NPY_FILE}, {PRODUCT_DATA_JSON_FILE}")
        print("Attempting to load pre-computed embeddings and product data...")
        try:
            # Load NumPy array for embeddings
            # Ensure global variables are assigned within this block if loading is successful
            loaded_embeddings = np.load(EMBEDDINGS_NPY_FILE)
            
            # Load JSON for product data (list of dicts)
            with open(PRODUCT_DATA_JSON_FILE, 'r', encoding='utf-8') as f:
                loaded_product_data = json.load(f)

            # Basic validation of loaded data
            if (isinstance(loaded_embeddings, np.ndarray) and loaded_embeddings.size > 0 and
                isinstance(loaded_product_data, list) and len(loaded_product_data) > 0 and
                loaded_embeddings.shape[0] == len(loaded_product_data)):
                
                product_embeddings = loaded_embeddings # Assign to global
                product_data = loaded_product_data   # Assign to global
                
                print(f"Successfully loaded {len(product_data)} items and their embeddings from files.")
                print(f"Embeddings array shape: {product_embeddings.shape}")
                loaded_from_files = True
            else:
                print("Loaded data is invalid, empty, or mismatched. Will re-generate embeddings.")
                # Ensure globals are reset for fresh generation
                product_embeddings = None
                product_data = []
        except Exception as e:
            print(f"Error loading data from files: {e}. Will re-generate embeddings.")
            traceback.print_exc()
            product_embeddings = None # Ensure reset
            product_data = []

    if not loaded_from_files:
        print("\nStep 1: Initializing Gemini Client (as data was not loaded from files or loading failed)...")
        if not initialize_gemini_client():
            print("Gemini Client initialization failed. Exiting.")
            exit()

        print("\nStep 2: Loading and embedding data from source... This may take a while.")
        if load_and_embed_data(): # This function populates the global product_embeddings and product_data
            print("Data embedding process complete.")
            if product_embeddings is not None and product_data:
                try:
                    print(f"Saving embeddings to {EMBEDDINGS_NPY_FILE}...")
                    np.save(EMBEDDINGS_NPY_FILE, product_embeddings)
                    
                    print(f"Saving product data to {PRODUCT_DATA_JSON_FILE}...")
                    with open(PRODUCT_DATA_JSON_FILE, 'w', encoding='utf-8') as f:
                        json.dump(product_data, f, indent=4) # indent for readability
                    print("Successfully saved newly generated embeddings and product data for future runs.")
                except Exception as e:
                    print(f"Error saving freshly generated data: {e}")
                    traceback.print_exc()
                    print("Proceeding with in-memory data for this session.")
            else:
                print("Warning: Embedding process finished but global data variables are not populated correctly. Cannot save.")
        else:
            print("\nFailed to load data or generate new embeddings. Server cannot start.")
            exit() # Exit if fresh data generation fails and nothing was loaded

    # Proceed to start Flask app if data is available (either loaded or freshly generated)
    if product_embeddings is not None and product_data and len(product_data) > 0:
        print(f"\nStep 3: Starting Flask server with {len(product_data)} items...")
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("Data is not available (neither loaded nor generated successfully). Flask server not started.")