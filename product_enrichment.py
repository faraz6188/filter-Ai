"""
MyronPromos
Nick Feuer
nfeuer@myron.com
05-28-2025
"""

import csv
import json
import time
import os
import argparse
import random
from typing import List, Dict, Any
from google import genai # Using this import as per your latest example
from google.genai import types # Added for GenerateContentConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'product_enrichment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class ProductEnricher:
    def __init__(self, api_key: str, model_name: str = "gemini-pro", tier: str = "free", test_mode: bool = False):
        """Initialize the Gemini API client using genai.Client."""
        self.api_key = api_key
        self.model_name = model_name # Store model_name for use in API calls
        self.test_mode = test_mode
        try:
            self.client = genai.Client(api_key=self.api_key) # Initialize genai.Client
        except Exception as e:
            logging.error(f"Failed to initialize genai.Client: {e}")
            raise
            
        self.tier = tier
        # This dictionary remains to hold config params; will be used to create types.GenerateContentConfig
        self.generation_config_params = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        self.request_times = []
        self.rate_limit_window = 60
        self.rate_limits = {
            'free': {'rpm': 15, 'delay': 4},
            'tier1': {'rpm': 2000, 'delay': 0.03},
            'tier2': {'rpm': 10000, 'delay': 0.006},
            'tier3': {'rpm': 30000, 'delay': 0.002}
        }
        
        # Set up JSON response logging for test mode
        if self.test_mode:
            self.json_response_log_file = f'c:\\Users\\Nick\\Downloads\\test_mode_json_responses_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
            logging.info(f"Test mode: JSON responses will be logged to {self.json_response_log_file}")
    
    def _log_json_response(self, product_name: str, product_id: str, prompt: str, raw_response: str, parsed_json: Dict[str, Any]):
        """Log JSON response details in test mode for quality review."""
        if not self.test_mode:
            return
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "product_name": product_name,
            "product_id": product_id,
            "prompt": prompt,
            "raw_ai_response": raw_response,
            "parsed_json": parsed_json,
            "response_length": len(raw_response),
            "json_keys": list(parsed_json.keys()) if isinstance(parsed_json, dict) else []
        }
        
        try:
            with open(self.json_response_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logging.warning(f"Could not write JSON response log for {product_name}: {e}")
    
    def _check_rate_limit(self):
        """Check if we're within rate limits and add appropriate delay if needed."""
        current_time = time.time()
        tier_limits = self.rate_limits.get(self.tier, self.rate_limits['free'])
        
        self.request_times = [t for t in self.request_times if current_time - t < self.rate_limit_window]
        
        if len(self.request_times) >= tier_limits['rpm']:
            oldest_request_time_in_window = self.request_times[0] if self.request_times else current_time
            wait_time = (self.rate_limit_window - (current_time - oldest_request_time_in_window)) + 1
            
            if wait_time > 0:
                logging.info(f"Approaching rate limit ({tier_limits['rpm']} RPM). Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                current_time_after_wait = time.time()
                self.request_times = [t for t in self.request_times if current_time_after_wait - t < self.rate_limit_window]

        self.request_times.append(time.time())
    
    def create_enrichment_prompt(self, product: Dict[str, str]) -> str:
        """Create a prompt for Gemini to enrich product data."""
        prompt = f"""
        Analyze the following product and generate additional metadata to improve searchability:

        Product Name: {product.get('name', '')}
        Description: {product.get('description', '')}
        Current Categories: {product.get('categories', '')}

        Please provide a JSON response with the following structure:
        {{
            "additional_categories": ["list of 3-5 relevant categories not already mentioned"],
            "subcategories": ["list of 3-5 specific subcategories"],
            "tags": ["list of 10-15 relevant tags for search optimization"],
            "search_keywords": ["list of 5-10 keywords users might search for"],
            "attributes": {{
                "use_cases": ["list of 2-4 use cases"],
                "target_audience": ["list of 2-3 target audiences"],
                "related_terms": ["list of 3-5 related or synonym terms"]
            }}
        }}

        Focus on terms that would help users find this product through search. Be specific and relevant.
        Return only valid JSON without any markdown formatting.
        """
        return prompt
    
    def enrich_product(self, product: Dict[str, str]) -> Dict[str, Any]:
        """Enrich a single product with AI-generated metadata with retry logic."""
        return self._retry_with_backoff(
            lambda: self._enrich_product_internal(product),
            product_name=product.get('name', 'Unknown Product')
        )
    
    def _enrich_product_internal(self, product: Dict[str, str]) -> Dict[str, Any]:
        """Internal method to enrich a single product using genai.Client."""
        self._check_rate_limit() 
        
        prompt = self.create_enrichment_prompt(product)
        product_name = product.get('name', 'Unknown Product')
        product_id = product.get('id', 'unknown_id')

        # Create the typed configuration object from stored parameters
        typed_config = types.GenerateContentConfig(
            temperature=self.generation_config_params.get("temperature"),
            top_p=self.generation_config_params.get("top_p"),
            top_k=self.generation_config_params.get("top_k"),
            max_output_tokens=self.generation_config_params.get("max_output_tokens")
        )
        

        response = self.client.models.generate_content(
            model=f"models/{self.model_name}", # Models are typically prefixed with "models/"
            contents=[prompt], 
            config=typed_config
        )
        
        response_text = ""
        try:
            # Access response.text as per the user's example.
            # Ensure to check for empty text which can indicate a problem (e.g. safety block with no clear text output)
            if hasattr(response, 'text'):
                response_text = response.text.strip()
            
            if not response_text:
                # Log more details if available, e.g., prompt_feedback for safety reasons
                feedback_info = ""
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    feedback_info = f" Prompt Feedback: {response.prompt_feedback}."
                
                logging.error(f"Empty text in response for product {product_name}.{feedback_info} Full Response: {response}")
                raise ValueError(f"Response from API was empty or malformed (empty text).{feedback_info}")

        except AttributeError: # If response object doesn't even have a .text attribute
            logging.error(f"No .text attribute in response for product {product_name}. Response: {response}")
            raise ValueError(f"Response from API was malformed (no .text attribute). Full response: {response}")
        except Exception as e: # Catch other potential issues with accessing response.text
            logging.error(f"Error accessing or processing response.text for product {product_name}: {e}. Response: {response}")
            raise ValueError(f"Could not retrieve or process text from API response: {e}. Full response: {response}")

        # Store the raw response before processing for test mode logging
        raw_response_text = response_text

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        try:
            enrichment_data = json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error for product {product_name}: {e}. Response text: '{response_text}'")
            
            # In test mode, still log the failed response for debugging
            if self.test_mode:
                self._log_json_response(
                    product_name=product_name,
                    product_id=product_id,
                    prompt=prompt,
                    raw_response=raw_response_text,
                    parsed_json={"error": f"JSON parsing failed: {e}"}
                )
            
            raise ValueError(f"JSON parsing failed: {e}. Text was: '{response_text}'") 
        
        # Log the successful JSON response in test mode
        if self.test_mode:
            self._log_json_response(
                product_name=product_name,
                product_id=product_id,
                prompt=prompt,
                raw_response=raw_response_text,
                parsed_json=enrichment_data
            )
        
        enriched_product = product.copy()
        enriched_product.update(enrichment_data)
        
        logging.info(f"Successfully enriched product: {product_name}")
        return enriched_product
        
    def _retry_with_backoff(self, func, max_retries: int = 5, product_name: str = "Unknown Product"):
        """
        Retry a function with exponential backoff for rate limit and transient errors.
        """
        base_delay = 2  
        max_delay = 300 
        
        for attempt in range(max_retries):
            try:
                return func() 
            except ValueError as e: 
                logging.error(f"Non-retryable ValueError for {product_name}: {e}")
                return None 
            except Exception as e: 
                error_str = str(e).lower()

                is_rate_limit = any(term in error_str for term in [
                    'rate limit', 'quota', 'too many requests', '429',
                    'resource exhausted', 'user facing backend quota', 'requests per minute'
                ]) or "ResourceExhausted" in type(e).__name__
                
                is_transient = any(term in error_str for term in [
                    'timeout', 'connection', 'temporary', '500', '502', '503', '504',
                    'service unavailable', 'internal error', 'unavailable',
                    'a model iteration is not available for this model', 'deadlineexceeded'
                ]) or any(err_type_name in type(e).__name__ for err_type_name in [
                    "ServiceUnavailable", "DeadlineExceeded", "InternalServerError"
                ])
                                
                if is_rate_limit or is_transient:
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        jitter = random.uniform(0, delay * 0.1) 
                        actual_delay = delay + jitter
                        
                        error_type = "Rate limit" if is_rate_limit else "Transient error"
                        logging.warning(
                            f"{error_type} for {product_name} (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {actual_delay:.1f} seconds... Error: {type(e).__name__} - {e}"
                        )
                        time.sleep(actual_delay)
                        
                        if is_rate_limit and attempt >= 1: 
                            logging.info(f"Additional cooldown for repeated rate limit on {product_name}...")
                            time.sleep(min(10 * (attempt + 1), 60)) 
                    else:
                        logging.error(
                            f"Max retries ({max_retries}) exceeded for {product_name} after {attempt + 1} attempts. Last Error: {type(e).__name__} - {e}"
                        )
                        raise 
                else: # Non-retryable (not ValueError, not classified as rate limit/transient)
                    logging.error(f"Non-retryable error (unclassified: {type(e).__name__}) for {product_name}: {e}")
                    raise 
        
        logging.error(f"Failing {product_name} after {max_retries} retries (should have raised). Returning None.")
        return None

    def process_csv(self, input_file: str, output_file: str, max_workers: int = 5, 
                    batch_size: int = 100, checkpoint_file: str = None, test_mode: bool = False):
        """Process CSV file with batching, checkpoint support, and incremental writes."""
        products_to_process = []
        processed_product_ids = set()
        
        if not test_mode and checkpoint_file and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                    processed_product_ids = set(checkpoint_data.get('processed_ids', []))
                    logging.info(f"Resuming from checkpoint: {len(processed_product_ids)} products already processed.")
            except json.JSONDecodeError:
                logging.warning(f"Could not parse checkpoint file {checkpoint_file}. Starting fresh.")
            except Exception as e:
                logging.warning(f"Could not load checkpoint file {checkpoint_file} due to {e}. Starting fresh.")

        
        with open(input_file, 'r', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            # fieldnames = reader.fieldnames if reader.fieldnames else [] # Not strictly needed here
            for i, row in enumerate(reader):
                if test_mode and len(products_to_process) >= 50: # Limit new products for test mode
                    break 
                
                product_id = row.get('id')
                if not product_id: 
                    product_name_slug = "".join(filter(str.isalnum, row.get('name', f'unnamedproduct{i}'))).lower()
                    product_id = f"{product_name_slug}_{random.randint(1000,9999)}_{i}"
                    row['id'] = product_id 

                if product_id not in processed_product_ids:
                    products_to_process.append(row)
        
        if test_mode:
            logging.info(f"TEST MODE: Processing up to {len(products_to_process)} products from '{input_file}' (max 50 new).")
            logging.info(f"JSON responses will be logged to: {self.json_response_log_file}")
        else:
            logging.info(f"Loaded {len(products_to_process)} new products to process from '{input_file}'.")
        
        if not products_to_process:
            logging.info("No new products to process based on input and checkpoint.")
            return

        total_products_overall = len(products_to_process)
        total_batches = (total_products_overall + batch_size - 1) // batch_size
        
        all_processed_ids_current_run = set()

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_products_overall)
            current_batch_items = products_to_process[start_idx:end_idx]
            
            if not current_batch_items: 
                continue

            logging.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(current_batch_items)} products).")
            
            enriched_batch_results = self._process_batch(current_batch_items, max_workers)
            
            # Filter out None results if enrich_product can return None on non-retryable failure
            valid_enriched_results = [p for p in enriched_batch_results if p is not None]
            successful_ids_in_batch = {p['id'] for p in valid_enriched_results if p.get('id')}
            all_processed_ids_current_run.update(successful_ids_in_batch)
            
            should_append = batch_num > 0 or (os.path.exists(output_file) and os.path.getsize(output_file) > 0)
            if valid_enriched_results:
                 self._write_results(output_file, valid_enriched_results, append=should_append)

            if not test_mode and checkpoint_file:
                # Combine existing processed IDs with newly successful ones for checkpointing
                combined_processed_ids = processed_product_ids.union(all_processed_ids_current_run)
                self._save_checkpoint(checkpoint_file, list(combined_processed_ids))
            
            if batch_num < total_batches - 1:
                pause_time = 30 if self.tier == 'free' else 10 
                logging.info(f"Pausing {pause_time} seconds between batches...")
                time.sleep(pause_time)
        
        final_processed_count = len(processed_product_ids.union(all_processed_ids_current_run))
        logging.info(f"Processing complete. Processed approximately {final_processed_count} products in total (including previous runs if checkpointed).")
        
        if test_mode:
            logging.info("\nTEST MODE COMPLETE!")
            logging.info(f"Output potentially saved to: {output_file}")
            logging.info(f"JSON responses logged to: {self.json_response_log_file}")
            logging.info("Review the enriched data and JSON response log, then run without --test flag for complete processing.")

    def _process_batch(self, batch_items: List[Dict], max_workers: int) -> List[Dict]:
        """Process a batch of products with concurrency, rate limiting, and retries."""
        results_for_batch = []
        
        if self.tier == 'free':
            max_workers = min(max_workers, 1)  

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_original_product = {
                executor.submit(self.enrich_product, product_item): product_item 
                for product_item in batch_items
            }
            
            for i, future in enumerate(as_completed(future_to_original_product)):
                original_product = future_to_original_product[future]
                product_name_for_log = original_product.get('name', original_product.get('id', 'Unknown Product'))
                
                try:
                    enriched_result = future.result() 
                    if enriched_result: # enrich_product could return None on non-retryable error
                        results_for_batch.append(enriched_result)
                    else:
                        logging.warning(f"Enrichment returned None (non-retryable failure) for product '{product_name_for_log}'. Including original product in output for completeness.")
                        results_for_batch.append(original_product) # Add original product if enrichment failed fundamentally
                except Exception as e: 
                    logging.error(f"Failed to process product '{product_name_for_log}' in batch after all retries: {type(e).__name__} - {e}. Including original product.")
                    results_for_batch.append(original_product) # Add original product on critical failure
                
                if (i + 1) % 10 == 0 or (i + 1) == len(batch_items):
                    logging.info(f"Processed {i + 1}/{len(batch_items)} products in current batch.")
        
        return results_for_batch

    def _save_checkpoint(self, checkpoint_file: str, processed_ids_list: List[str]):
        """Save checkpoint for resume capability."""
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_ids': processed_ids_list,
                    'timestamp': datetime.now().isoformat()
                }, f)
            logging.info(f"Checkpoint saved to {checkpoint_file} with {len(processed_ids_list)} processed IDs.")
        except IOError as e:
            logging.error(f"Could not write checkpoint file {checkpoint_file}: {e}")

    def _write_results(self, output_file: str, products_to_write: List[Dict], append: bool = False):
        """Write results to CSV, creating header if new, or appending if specified."""
        if not products_to_write:
            logging.info("No products to write in current batch.")
            return
        
        all_fieldnames_set = set()
        for p in products_to_write:
            all_fieldnames_set.update(p.keys())
        
        if 'id' in all_fieldnames_set:
            final_fieldnames = ['id'] + sorted([fn for fn in all_fieldnames_set if fn != 'id'])
        else:
            final_fieldnames = sorted(list(all_fieldnames_set))

        csv_ready_rows = []
        for product_dict in products_to_write:
            row = {}
            for field in final_fieldnames:
                value = product_dict.get(field)
                if isinstance(value, (list, dict)):
                    try:
                        row[field] = json.dumps(value)
                    except TypeError as te:
                        logging.warning(f"Could not JSON serialize field '{field}' for product ID '{product_dict.get('id', 'N/A')}': {te}. Storing as string.")
                        row[field] = str(value)
                elif value is None:
                    row[field] = '' 
                else:
                    row[field] = str(value) 
            csv_ready_rows.append(row)
        
        file_exists_and_not_empty = os.path.exists(output_file) and os.path.getsize(output_file) > 0
        mode = 'a' if append and file_exists_and_not_empty else 'w'
        
        try:
            with open(output_file, mode, newline='', encoding='utf-8') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=final_fieldnames, extrasaction='ignore')
                if mode == 'w' or not file_exists_and_not_empty: 
                    writer.writeheader()
                writer.writerows(csv_ready_rows)
            logging.info(f"Successfully wrote {len(csv_ready_rows)} products to '{output_file}' (mode: {mode}).")
        except IOError as e:
            logging.error(f"Could not write to output file {output_file}: {e}")
    
    def create_embeddings_ready_text(self, enriched_product_data: Dict[str, Any]) -> str:
        """Create optimized text for embedding from enriched product data."""
        parts = []
        
        parts.append(f"Product Name: {enriched_product_data.get('name', '')}")
        description = enriched_product_data.get('description', '')
        if description: parts.append(f"Description: {description}")
        
        categories = enriched_product_data.get('categories', '')
        if categories:
            if isinstance(categories, str): 
                 parts.append(f"Original Categories: {categories}")
            elif isinstance(categories, list): 
                 parts.append(f"Original Categories: {', '.join(map(str, categories))}")

        for field_key in ['additional_categories', 'subcategories', 'tags', 'search_keywords']:
            data_val = enriched_product_data.get(field_key)
            if data_val:
                current_list = []
                if isinstance(data_val, str): # Potentially a JSON string from CSV
                    try:
                        loaded_data = json.loads(data_val)
                        if isinstance(loaded_data, list):
                            current_list = [str(item) for item in loaded_data if item is not None]
                    except json.JSONDecodeError: # If not JSON, treat as single item if non-empty
                        current_list = [data_val.strip()] if data_val.strip() else []
                elif isinstance(data_val, list):
                    current_list = [str(item) for item in data_val if item is not None]
                
                if current_list:
                    parts.append(f"{field_key.replace('_', ' ').title()}: {', '.join(current_list)}")
        
        attributes_data = enriched_product_data.get('attributes')
        if attributes_data:
            attrs_dict = {}
            if isinstance(attributes_data, str): # Potentially a JSON string from CSV
                try:
                    attrs_dict = json.loads(attributes_data)
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse attributes JSON string for embedding: {attributes_data[:100]}") # Log snippet
            elif isinstance(attributes_data, dict):
                attrs_dict = attributes_data
            
            if isinstance(attrs_dict, dict):
                for attr_name, attr_values_list in attrs_dict.items():
                    if isinstance(attr_values_list, list) and attr_values_list:
                        str_values = [str(v) for v in attr_values_list if v is not None]
                        if str_values:
                             parts.append(f"{attr_name.replace('_', ' ').title()}: {', '.join(str_values)}")
        
        return " | ".join(filter(None, parts)) 

def main():
    parser = argparse.ArgumentParser(
        description='Enrich product data using Google Gemini AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in test mode (first 50 products only, from input)
  python product_enrichment.py --test
  
  # Run full enrichment
  python product_enrichment.py --full
  
  # Specify custom input/output files
  python product_enrichment.py --test --input my_products.csv --output test_enriched.csv
  
  # Run with specific tier settings (e.g., tier1)
  python product_enrichment.py --full --tier tier1 --workers 5 --batch-size 50 --model "gemini-1.5-flash"
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--test', action='store_true', 
                           help='Test mode: Process only the first ~50 new products from the input CSV.')
    mode_group.add_argument('--full', action='store_true', 
                           help='Full mode: Process entire CSV file, respecting checkpoints.')
    
    parser.add_argument('--input', type=str, default='c:\\Users\\Nick\\Downloads\\65NA_Products.csv', # More generic default
                       help='Input CSV file (default: full_product_NA.csv)')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output CSV file name (default changes based on mode)')
    # Removed default API key from argparse for security. User must provide it.
    parser.add_argument('--api-key', type=str, default='AIzaSyD-pwjVilEFhZ_lOzLJAO4RgSoNYh3rbNw',
                       help='Gemini API key (reads from GEMINI_API_KEY env var by default, or provide here)')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Gemini model name to use (e.g., gemini-pro, gemini-1.5-flash) (default: gemini-pro)')
    parser.add_argument('--tier', type=str, default='free',
                       choices=['free', 'tier1', 'tier2', 'tier3'],
                       help='API tier for rate limiting (default: free)')
    parser.add_argument('--batch-size', type=int, default=None, 
                       help='Number of products to process in each batch (auto-calculated if not set)')
    parser.add_argument('--workers', type=int, default=None, 
                       help='Max concurrent workers for API calls (auto-calculated if not set)')
    
    args = parser.parse_args()
    
    if not args.api_key:
        logging.error("ERROR: Gemini API key not found. Set GEMINI_API_KEY env variable or use --api-key.")
        print("ERROR: Please provide a Gemini API key via --api-key flag or set the GEMINI_API_KEY environment variable.")
        return

    INPUT_CSV = args.input
    test_mode = args.test
    
    if args.output:
        OUTPUT_CSV = args.output
    else: 
        base_name, ext = os.path.splitext(INPUT_CSV)
        OUTPUT_CSV = f"{base_name}_test_enriched{ext}" if test_mode else f"{base_name}_enriched{ext}"
    
    CHECKPOINT_FILE = None
    if not test_mode:
        output_dir = os.path.dirname(OUTPUT_CSV) or '.'
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
        checkpoint_base_name = os.path.splitext(os.path.basename(OUTPUT_CSV))[0]
        CHECKPOINT_FILE = os.path.join(output_dir, f"{checkpoint_base_name}_checkpoint.json")

    TIER = args.tier
    MODEL_NAME = args.model # Use the model from arguments
    
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file '{INPUT_CSV}' not found.")
        print(f"ERROR: Input file '{INPUT_CSV}' not found.")
        return
    
    try:
        enricher = ProductEnricher(args.api_key, model_name=MODEL_NAME, tier=TIER, test_mode=test_mode)
    except Exception as e:
        print(f"Failed to initialize ProductEnricher: {e}. Check API key and model name.")
        logging.critical(f"Failed to initialize ProductEnricher: {e}", exc_info=True)
        return

    initial_rows_in_csv = 0
    try:
        with open(INPUT_CSV, 'r', encoding='utf-8') as f_count:
            reader = csv.reader(f_count)
            header = next(reader, None)
            if header: # Count only data rows
                initial_rows_in_csv = sum(1 for _ in reader)
    except Exception as e:
        logging.warning(f"Could not accurately count rows in {INPUT_CSV}: {e}. Proceeding with defaults if needed.")
        initial_rows_in_csv = 1000 # Fallback for calculations if count fails

    display_row_count = min(initial_rows_in_csv, 50) if test_mode else initial_rows_in_csv

    # Auto-determine batch size
    if args.batch_size:
        batch_size = args.batch_size
    elif test_mode:
        batch_size = 10
    elif TIER == 'free':
        batch_size = 5 
    elif initial_rows_in_csv < 200: 
        batch_size = 20 
    elif initial_rows_in_csv < 1000:
        batch_size = 50
    elif initial_rows_in_csv < 10000:
        batch_size = 100
    else: 
        batch_size = 200
    batch_size = max(1, batch_size) # Ensure batch_size is at least 1

    # Auto-determine workers
    if args.workers:
        max_workers = args.workers
    elif TIER == "free":
        max_workers = 1 
    elif test_mode:
        max_workers = 2
    elif initial_rows_in_csv < 200:
        max_workers = 2
    elif initial_rows_in_csv < 1000:
        max_workers = 3 # Reduced default for broader safety
    elif initial_rows_in_csv < 10000:
        max_workers = 5 # Reduced default
    else: 
        max_workers = 5 # Capped default max_workers
    max_workers = max(1, max_workers) # Ensure max_workers is at least 1

    print("\n" + "="*60)
    print(f"GEMINI PRODUCT ENRICHMENT - {'TEST MODE' if test_mode else 'FULL RUN'}")
    print("="*60)
    print(f"Input file:         {INPUT_CSV}")
    print(f"Output file:        {OUTPUT_CSV}")
    if not test_mode and CHECKPOINT_FILE:
        print(f"Checkpoint file:    {CHECKPOINT_FILE}")
    if test_mode:
        print(f"JSON response log:  {enricher.json_response_log_file}")
    print(f"Model:              {MODEL_NAME}")
    print(f"Products in CSV:    {initial_rows_in_csv} (approx. {display_row_count} to process if test/new run)")
    print(f"API Tier:           {TIER.upper()}")
    print(f"Batch size:         {batch_size}")
    print(f"Max workers:        {max_workers}")
    if test_mode:
        print("\n⚠️  TEST MODE: Processing up to ~50 new products from input.")
        print("📝 AI JSON responses will be logged for quality review.")
    print("="*60 + "\n")
    
    if not test_mode:
        try:
            user_response = input("Proceed with full enrichment? (y/N): ")
            if user_response.lower() != 'y':
                print("Enrichment cancelled by user.")
                return
        except EOFError: 
            logging.info("No user input (EOFError). Proceeding with full enrichment automatically.")
            print("No user input detected, proceeding with full enrichment...")

    start_time = time.time()
    try:
        enricher.process_csv(
            INPUT_CSV, 
            OUTPUT_CSV, 
            max_workers=max_workers,
            batch_size=batch_size,
            checkpoint_file=CHECKPOINT_FILE, 
            test_mode=test_mode
        )
    except Exception as e:
        logging.critical(f"A critical error occurred during enrichment process: {e}", exc_info=True)
        print(f"AN ERROR OCCURRED. Please check the log file. Error: {type(e).__name__} - {e}")
    finally:
        end_time = time.time()
        logging.info(f"Total execution time: {end_time - start_time:.2f} seconds.")
        print(f"Total execution time: {end_time - start_time:.2f} seconds.")

    print("\n" + "="*60)
    print(f"SAMPLE ENRICHED DATA FROM: {OUTPUT_CSV}")
    print("="*60)
    
    try:
        if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
            print(f"Output file '{OUTPUT_CSV}' not found, empty, or not created. Cannot display samples.")
        else:
            with open(OUTPUT_CSV, 'r', encoding='utf-8') as f_sample:
                reader = csv.DictReader(f_sample)
                sample_count = 0
                for product_row in reader: # Iterate through reader
                    if sample_count < 3:
                        print(f"\n--- Product Sample {sample_count+1} ---")
                        print(f"Name: {product_row.get('name', 'N/A')}")
                        
                        for field_name in ['additional_categories', 'tags', 'search_keywords']:
                            if field_name in product_row and product_row[field_name]:
                                data_content = product_row[field_name]
                                try:
                                    parsed_data = data_content # Default to string
                                    if isinstance(data_content, str) and (data_content.startswith('[') or data_content.startswith('{')):
                                        parsed_data = json.loads(data_content)
                                except json.JSONDecodeError:
                                    pass # Keep as string if not valid JSON

                                if isinstance(parsed_data, list):
                                    display_items = [str(item) for item in parsed_data[:3]] 
                                    print(f"{field_name.replace('_', ' ').title()}: {', '.join(display_items)} {'...' if len(parsed_data) > 3 else ''}")
                                elif isinstance(parsed_data, str): 
                                    print(f"{field_name.replace('_', ' ').title()}: {parsed_data[:100]}{'...' if len(parsed_data) > 100 else ''}")
                        
                        embedding_text_sample = enricher.create_embeddings_ready_text(product_row)
                        print(f"\nEmbedding text preview:")
                        print(f"{embedding_text_sample[:250]}{'...' if len(embedding_text_sample) > 250 else ''}")
                        sample_count +=1
                    else:
                        break 
                if sample_count == 0 : 
                     print(f"No data rows found in '{OUTPUT_CSV}' to display samples.")

    except Exception as e:
        print(f"Error reading or displaying samples from output file '{OUTPUT_CSV}': {e}")
    
    # Display JSON response log summary in test mode
    if test_mode and hasattr(enricher, 'json_response_log_file'):
        print("\n" + "="*60)
        print("JSON RESPONSE LOG SUMMARY")
        print("="*60)
        try:
            if os.path.exists(enricher.json_response_log_file):
                with open(enricher.json_response_log_file, 'r', encoding='utf-8') as f:
                    log_entries = []
                    for line in f:
                        try:
                            log_entries.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
                    
                    if log_entries:
                        print(f"📊 Total AI responses logged: {len(log_entries)}")
                        print(f"📄 JSON response log file: {enricher.json_response_log_file}")
                        
                        # Show summary statistics
                        successful_responses = [entry for entry in log_entries if 'error' not in entry.get('parsed_json', {})]
                        failed_responses = [entry for entry in log_entries if 'error' in entry.get('parsed_json', {})]
                        
                        print(f"✅ Successful responses: {len(successful_responses)}")
                        if failed_responses:
                            print(f"❌ Failed responses: {len(failed_responses)}")
                        
                        if successful_responses:
                            avg_response_length = sum(entry.get('response_length', 0) for entry in successful_responses) / len(successful_responses)
                            print(f"📝 Average response length: {avg_response_length:.0f} characters")
                            
                            # Show sample of JSON keys from responses
                            all_keys = set()
                            for entry in successful_responses[:5]:  # Sample first 5
                                all_keys.update(entry.get('json_keys', []))
                            if all_keys:
                                print(f"🔑 JSON keys found: {', '.join(sorted(all_keys))}")
                        
                        print(f"\n💡 Review the complete JSON responses in: {enricher.json_response_log_file}")
                        print("   Each line contains a complete log entry with prompt, response, and parsed JSON.")
                    else:
                        print(f"⚠️  No valid JSON log entries found in {enricher.json_response_log_file}")
            else:
                print(f"⚠️  JSON response log file not found: {enricher.json_response_log_file}")
        except Exception as e:
            print(f"❌ Error reading JSON response log: {e}")
    
    print("\n" + "="*60)
    if test_mode:
        print("✅ TEST COMPLETE! Review the output and logs. For full processing, run without --test.")
        print("📋 Check the JSON response log to verify AI response quality.")
    else:
        print("✅ ENRICHMENT PROCESS FINISHED! Review the output and logs.")
    print(f"Output data should be in: {OUTPUT_CSV}")
    if test_mode:
        print(f"JSON responses logged in: {enricher.json_response_log_file}")
    print("="*60)

if __name__ == "__main__":
    main()