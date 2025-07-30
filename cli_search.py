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
GEMINI_EMBEDDING_MODEL_NAME = "gemini-embedding-exp-03-07"
EMBEDDINGS_NPY_FILE = 'product_embeddings.npy'
PRODUCT_DATA_JSON_FILE = 'product_data.json'
EMBEDDING_BATCH_SIZE = 50
DELAY_BETWEEN_BATCHES = 60 # Seconds
SEARCH_RESULTS_JSON_FILE = 'search_result.json' # For CLI search output

# --- Search Configuration ---
MAX_CATEGORIES = 3  
MAX_PRODUCTS = 300   # Upper limit for products
SIMILARITY_THRESHOLD = 0.6769071797519335 # Test only

# --- Global Variables ---
product_data = []
product_embeddings = None
gemini_client = None

# --- Helper Functions ---
# parse_text_for_details function remains the same
def parse_text_for_details(text_for_embedding_string):
    """
    Parses the 'text_for_embedding' string to extract product name, primary category, and URL ending.
    """
    product_name = "Unknown Product"
    primary_category = "Unknown Category"
    product_url_ending = ""
    
    match = re.match(r"([\s\S]+?)\.\s*category:\s*([\s\S]+?)\.\s*url:\s*([\s\S]+?)\.\s*description:", text_for_embedding_string, re.IGNORECASE | re.DOTALL)

    if match:
        product_name = match.group(1).strip()
        category_string = match.group(2).strip()
        product_url_ending = match.group(3).strip()

        if category_string:
            categories = [cat.strip() for cat in category_string.split(',')]
            if categories:
                first_cat = categories[0]
                first_cat = re.sub(r'\s*-\s*l\d+$', '', first_cat, flags=re.IGNORECASE).strip()
                first_cat = re.sub(r',\s*nan$', '', first_cat, flags=re.IGNORECASE).strip() 
                primary_category = first_cat if first_cat else "Unknown Category"
    else:
        print(f"Warning: Could not parse details using main regex for text: '{text_for_embedding_string[:70]}...'")
        parts = text_for_embedding_string.split('.')
        if len(parts) > 0: 
            product_name = parts[0].strip()

    return {
        "name": product_name,
        "category": primary_category,
        "url_ending": product_url_ending,
        "full_text": text_for_embedding_string
    }

# --- Initialize Gemini Client ---
def initialize_gemini_client():
    global gemini_client
    api_key_to_use = 'AIzaSyD-pwjVilEFhZ_lOzLJAO4RgSoNYh3rbNw' # Use environment variable for production, hardcoded for testing

    if not api_key_to_use:
        api_key_env_google = os.environ.get("GOOGLE_API_KEY")
        api_key_env_gemini = os.environ.get("GEMINI_API_KEY")
        if api_key_env_google:
            print("Using API key from GOOGLE_API_KEY environment variable.")
            api_key_to_use = api_key_env_google
        elif api_key_env_gemini:
            print("Using API key from GEMINI_API_KEY environment variable.")
            api_key_to_use = api_key_env_gemini
        
    if not api_key_to_use:
        print("API key not found in environment variables (GOOGLE_API_KEY or GEMINI_API_KEY).")
        api_key_to_use = 'AIzaSyD-pwjVilEFhZ_lOzLJAO4RgSoNYh3rbNw' # Fallback to hardcoded key, fix for production
        print("Using the API key provided in the script for initialization.")

    if not api_key_to_use:
        print("Critical: Gemini API Key is not configured. Cannot initialize client.")
        return False

    try:
        gemini_client = genai.Client(api_key=api_key_to_use)
        print("Successfully initialized genai.Client.")
        return True
    except Exception as e_general:
        print(f"Error in initialize_gemini_client: {e_general}"); traceback.print_exc(); return False

# --- Load and Embed Data ---
def load_and_embed_data():
    global product_data, product_embeddings, gemini_client
    if not gemini_client:
        print("Error: Gemini client not initialized in load_and_embed_data. Attempting to initialize...");
        if not initialize_gemini_client(): # Ensure client is initialized
             print("Failed to initialize Gemini client within load_and_embed_data.")
             return False
        if not gemini_client: # Double check
            print("Critical: Gemini client still None after init attempt in load_and_embed_data.")
            return False


    print(f"Loading data from {CSV_FILE_PATH}...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        all_original_texts = df['text_for_embedding'].tolist()
        all_ids = df['id'].tolist()
        
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

        valid_ids_for_embedding = [id_val for id_val in all_ids if id_val in id_to_data_map]
        texts_for_embedding_api = [id_to_data_map[id_val]["original_text"] for id_val in valid_ids_for_embedding]
        
        total_valid_texts = len(texts_for_embedding_api)
        if total_valid_texts == 0:
            print("No valid texts found to embed after initial processing.")
            return False

        print(f"Generating embeddings for {total_valid_texts} valid texts in batches of {EMBEDDING_BATCH_SIZE} using model '{GEMINI_EMBEDDING_MODEL_NAME}'...")
        
        product_data.clear() 
        retrieved_embeddings_list = [] 

        for i in range(0, total_valid_texts, EMBEDDING_BATCH_SIZE):
            current_batch_texts = texts_for_embedding_api[i : i + EMBEDDING_BATCH_SIZE]
            current_batch_ids = valid_ids_for_embedding[i : i + EMBEDDING_BATCH_SIZE]

            if not current_batch_texts: continue

            print(f"Processing batch {i // EMBEDDING_BATCH_SIZE + 1}/{(total_valid_texts + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE}...")
            
            try:
                response = gemini_client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL_NAME,
                    contents=current_batch_texts,
                    config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                
                if response.embeddings and len(response.embeddings) == len(current_batch_texts):
                    for k, embedding_obj in enumerate(response.embeddings):
                        item_id = current_batch_ids[k]
                        data_for_id = id_to_data_map[item_id]
                        
                        if hasattr(embedding_obj, 'values'):
                            retrieved_embeddings_list.append(embedding_obj.values)
                            product_data.append({
                                "id": item_id,
                                "name": data_for_id["details"]["name"],
                                "category": data_for_id["details"]["category"],
                                "url": f"www.myron.com{data_for_id['details']['url_ending']}",
                                "original_text": data_for_id["original_text"]
                            })
                        else:
                            print(f"Warning: Embedding object for ID {item_id} lacks '.values'. Skipping.")
                else:
                    print(f"Warning: Embedding response for batch mismatch. Skipping batch.")

                if i + EMBEDDING_BATCH_SIZE < total_valid_texts:
                    print(f"Batch processed. Sleeping for {DELAY_BETWEEN_BATCHES} seconds...")
                    time.sleep(DELAY_BETWEEN_BATCHES)

            except Exception as e:
                print(f"Error embedding batch for ID {current_batch_ids[0]}: {e}"); traceback.print_exc()
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    print("Rate limit error likely. Sleeping for an extended period (e.g., 60s)...")
                    time.sleep(60) # Longer sleep for rate limits
        
        if not retrieved_embeddings_list:
            print("Error: No embeddings generated."); return False
        
        product_embeddings_np_array = np.array(retrieved_embeddings_list)
        # Attempt conversion to float, handle potential errors if embeddings are not uniform
        try:
            product_embeddings = product_embeddings_np_array.astype(float)
        except ValueError:
            print("Warning: Could not convert all embeddings to float. Using dtype=object for NumPy array.")
            print("This might affect cosine_similarity if embeddings are not numerical lists/arrays of the same dimension.")
            product_embeddings = product_embeddings_np_array


        if product_embeddings.size == 0 or not product_data:
            print("Error: Batch embedding resulted in empty data."); return False
        if len(product_data) != product_embeddings.shape[0]:
            print(f"Error: Mismatch: embeddings ({product_embeddings.shape[0]}) vs product data ({len(product_data)}).")
            return False
            
        print(f"Data loaded. Embeddings shape: {product_embeddings.shape}, Products: {len(product_data)}")
        return True
        
    except Exception as e:
        print(f"Critical error in load_and_embed_data: {e}"); traceback.print_exc(); return False

# --- CLI Search Function ---
def perform_cli_search(search_query, client, embeddings, data):
    """
    Performs a search based on the query and returns results filtered by a similarity threshold
    and capped by MAX_PRODUCTS and MAX_CATEGORIES.
    """
    # Access global configuration for search parameters
    global GEMINI_EMBEDDING_MODEL_NAME, MAX_CATEGORIES, MAX_PRODUCTS, SIMILARITY_THRESHOLD

    if embeddings is None or not data or embeddings.size == 0:
        print("Error: Embeddings/data not loaded or empty for CLI search.")
        return None
    if not client:
        print("Error: Gemini client not available for CLI search.")
        return None
    if not search_query.strip():
        return {"categories": [], "products": []}

    try:
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL_NAME,
            contents=[search_query],
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        
        if response.embeddings and len(response.embeddings) > 0 and hasattr(response.embeddings[0], 'values'):
            query_embedding = np.array(response.embeddings[0].values).reshape(1, -1)
        else:
            print(f"Error: Failed to generate query embedding for '{search_query}'.")
            return None
        
        current_product_embeddings = embeddings
        if current_product_embeddings.ndim == 1:
            try:
                current_product_embeddings = np.vstack(current_product_embeddings)
            except ValueError as ve:
                print(f"Error: Could not reshape/stack product_embeddings from 1D to 2D: {ve}")
                return None
        
        if current_product_embeddings.shape[1] != query_embedding.shape[1]:
            print(f"Error: Query embedding dimension ({query_embedding.shape[1]}) does not match product embedding dimension ({current_product_embeddings.shape[1]}).")
            return None


        similarities = cosine_similarity(query_embedding, current_product_embeddings)[0]
        
        # Get all indices sorted by similarity (highest to lowest)
        all_sorted_indices = np.argsort(similarities)[::-1]
        
        suggested_categories_dict = {}
        suggested_products_list = []
        seen_product_names = set()

        print(f"Filtering results with similarity threshold: {SIMILARITY_THRESHOLD}")
        print(f"Max products to return: {MAX_PRODUCTS}, Max categories: {MAX_CATEGORIES}")

        for index in all_sorted_indices:
            similarity_score = float(similarities[index])

            # Primary filter: Stop if similarity is below the threshold
            if similarity_score < SIMILARITY_THRESHOLD:
                print(f"Stopping at similarity {similarity_score:.4f} (below threshold {SIMILARITY_THRESHOLD}).")
                break 

            # Stop if we have already collected enough products AND categories (respecting MAX limits)
            if len(suggested_products_list) >= MAX_PRODUCTS and len(suggested_categories_dict) >= MAX_CATEGORIES:
                print(f"Reached MAX_PRODUCTS ({MAX_PRODUCTS}) and MAX_CATEGORIES ({MAX_CATEGORIES}).")
                break
            
            if index < len(data):
                item = data[index]
                
                # Add category if not already at max categories
                if len(suggested_categories_dict) < MAX_CATEGORIES:
                    category_name = item.get('category', 'Unknown Category')
                    if category_name != "Unknown Category" and category_name not in suggested_categories_dict:
                        suggested_categories_dict[category_name] = similarity_score # Store with score, or just add name

                # Add product if not already at max products and product name not seen
                if len(suggested_products_list) < MAX_PRODUCTS:
                    product_name = item.get('name', 'Unknown Product')
                    if product_name != "Unknown Product" and product_name not in seen_product_names:
                        suggested_products_list.append({
                            "name": product_name, 
                            "url": item.get('url', '#'), 
                            "similarity": similarity_score,
                            # "original_text": item.get("original_text", "") 
                        })
                        seen_product_names.add(product_name)
                        # print(f"Added product: {product_name} (Sim: {similarity_score:.4f})") # For debugging
            else:
                print(f"Warning: Index {index} out of bounds for product_data (len: {len(data)}).")


        final_categories = list(suggested_categories_dict.keys())
        # The products in suggested_products_list are already ordered by relevance
        # because we iterated through all_sorted_indices.

        print(f"Found {len(suggested_products_list)} relevant products and {len(final_categories)} categories above threshold.")
        return {"categories": final_categories, "products": suggested_products_list}

    except Exception as e: 
        print(f"Error during CLI search for '{search_query}': {e}"); traceback.print_exc()
        return None
        

        final_categories = list(suggested_categories_dict.keys()) # Or sort dict by value if needed
        return {"categories": final_categories, "products": suggested_products_list}

    except Exception as e: 
        print(f"Error during CLI search for '{search_query}': {e}"); traceback.print_exc()
        return None

# --- Flask Application ---
app = Flask(__name__)
CORS(app)

@app.route('/search', methods=['POST'])
def search_suggestions_api():
    global product_embeddings, product_data, gemini_client
    
    if not gemini_client:
        print("Search API: gemini_client is None. Attempting re-initialization...")
        if not initialize_gemini_client():
            return jsonify({"error": "Gemini client could not be initialized."}), 500
        if not gemini_client:
             return jsonify({"error": "Critical: Gemini client unavailable."}), 500
        print("Search API: Gemini client re-initialized successfully.")

    query_data = request.get_json()
    if not query_data or 'query' not in query_data:
        return jsonify({"error": "Missing 'query'"}), 400
    search_query = query_data['query']

    results = perform_cli_search(search_query, gemini_client, product_embeddings, product_data)

    if results is None:
        return jsonify({"error": "Search processing error."}), 500
    else:
        return jsonify(results)


if __name__ == '__main__':
    print("--- AI Search Backend (with Caching, Batching & CLI Search Option) ---")
    print(f"Python Version: {sys.version.split()[0]} on {sys.platform}")

    loaded_from_files = False
    if os.path.exists(EMBEDDINGS_NPY_FILE) and os.path.exists(PRODUCT_DATA_JSON_FILE):
        print(f"\nFound existing data files: {EMBEDDINGS_NPY_FILE}, {PRODUCT_DATA_JSON_FILE}")
        print("Attempting to load pre-computed embeddings and product data...")
        try:
            loaded_embeddings = np.load(EMBEDDINGS_NPY_FILE, allow_pickle=True) # allow_pickle if dtypes are objects
            with open(PRODUCT_DATA_JSON_FILE, 'r', encoding='utf-8') as f:
                loaded_product_data = json.load(f)

            if (isinstance(loaded_embeddings, np.ndarray) and loaded_embeddings.size > 0 and
                isinstance(loaded_product_data, list) and len(loaded_product_data) > 0 and
                loaded_embeddings.shape[0] == len(loaded_product_data)):
                
                product_embeddings = loaded_embeddings
                product_data = loaded_product_data
                
                print(f"Successfully loaded {len(product_data)} items and their embeddings from files.")
                print(f"Embeddings array shape: {product_embeddings.shape}, dtype: {product_embeddings.dtype}")
                loaded_from_files = True
            else:
                print("Loaded data is invalid or mismatched. Will re-generate.")
                product_embeddings = None
                product_data = []
        except Exception as e:
            print(f"Error loading data from files: {e}. Will re-generate.")
            traceback.print_exc()
            product_embeddings = None
            product_data = []


    if not gemini_client: # Check if client already initialized
        print("\nStep 1: Initializing Gemini Client...")
        if not initialize_gemini_client():
            print("Gemini Client initialization failed. Exiting.")
            sys.exit(1)

    if not loaded_from_files:
        if not gemini_client: # Should have been initialized above if not loaded_from_files
            print("Critical: Gemini client not initialized before attempting to embed new data. Exiting.")
            sys.exit(1)

        print("\nStep 2: Loading and embedding data from source... This may take a while.")
        if load_and_embed_data():
            print("Data embedding process complete.")
            if product_embeddings is not None and product_data:
                try:
                    print(f"Saving embeddings to {EMBEDDINGS_NPY_FILE}...")
                    np.save(EMBEDDINGS_NPY_FILE, product_embeddings)
                    
                    print(f"Saving product data to {PRODUCT_DATA_JSON_FILE}...")
                    with open(PRODUCT_DATA_JSON_FILE, 'w', encoding='utf-8') as f:
                        json.dump(product_data, f, indent=4)
                    print("Successfully saved newly generated embeddings and product data.")
                except Exception as e:
                    print(f"Error saving freshly generated data: {e}"); traceback.print_exc()
            else:
                print("Warning: Embedding process finished but global data is not populated. Cannot save.")
        else:
            print("\nFailed to load data or generate new embeddings. Cannot proceed.")
            sys.exit(1)

    # --- All Initializations Done - Offer CLI Search or Server ---
    if product_embeddings is not None and product_data and len(product_data) > 0 and gemini_client:
        print("\nInitialization complete. Data and Gemini client are ready.")
        
        while True:
            choice = input("Choose an action: (S)earch via CLI, (W)eb server, (E)xit [S/W/E]: ").strip().upper()
            if choice == 'S':
                search_term = input("Enter your search term: ").strip()
                if search_term:
                    print(f"Performing CLI search for: '{search_term}'...")
                    results = perform_cli_search(search_term, gemini_client, product_embeddings, product_data)
                    if results:
                        try:
                            with open(SEARCH_RESULTS_JSON_FILE, 'w', encoding='utf-8') as f:
                                json.dump(results, f, indent=4)
                            print(f"Search results saved to '{SEARCH_RESULTS_JSON_FILE}'")
                        except Exception as e:
                            print(f"Error saving search results to JSON: {e}")
                    else:
                        print("CLI search did not return results or encountered an error.")
                else:
                    print("Search term cannot be empty.")
                # Decide if you want to loop for more CLI searches or exit
                another_action = input("Perform another action? (Y/N): ").strip().upper()
                if another_action != 'Y':
                    break 
            elif choice == 'W':
                print(f"\nStep 3: Starting Flask server with {len(product_data)} items...")
                try:
                    app.run(host='0.0.0.0', port=5001, debug=True) # debug True for development only
                except Exception as e:
                    print(f"Failed to start Flask server: {e}")
                break 
            elif choice == 'E':
                print("Exiting.")
                break 
            else:
                print("Invalid choice. Please enter 'S', 'W', or 'E'.")
    else:
        print("\nData is not available or Gemini client failed to initialize. Cannot proceed.")

    print("Application finished.")