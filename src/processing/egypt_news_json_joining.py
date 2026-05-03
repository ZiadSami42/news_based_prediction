import json
import os
import glob

def merge_json_to_json(input_folder: str, output_json: str) -> None:
    """
    Reads daily JSON files from a directory, removes duplicates based on 'title',
    and outputs a filtered, merged JSON file.
    """
    # Define the exact keys to keep based on your requirements
    keys_to_keep = [
        "title", "lead", "category", "date", "author", 
        "provider", "tags", "body", "url"
    ]
    
    seen_titles = set()
    merged_data = []
    processed_count = 0
    duplicate_count = 0
    empty_file_count = 0

    # Locate all JSON files in the target directory
    file_pattern = os.path.join(input_folder, "*.json")
    json_files = glob.glob(file_pattern)
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return

    for file_path in sorted(json_files):
        with open(file_path, mode='r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {file_path}: Invalid JSON.")
                continue
            
            # Handle empty JSON files '[]'
            if not data:
                empty_file_count += 1
                continue
                
            for article in data:
                title = article.get("title")
                
                # Skip if there is no title or if it's already been processed
                if not title or title in seen_titles:
                    if title:
                        duplicate_count += 1
                    continue
                    
                # Mark title as seen
                seen_titles.add(title)
                
                # Flatten the 'tags' array into a single delimited string as requested
                tags = article.get("tags", [])
                if isinstance(tags, list):
                    article["tags"] = "|".join(tags)
                else:
                    article["tags"] = str(tags) if tags else ""
                    
                # Construct the filtered object safely
                filtered_article = {key: article.get(key, "") for key in keys_to_keep}
                
                merged_data.append(filtered_article)
                processed_count += 1

    # Write the merged list to the final JSON file
    with open(output_json, 'w', encoding='utf-8') as out_f:
        # indent=4 makes it readable; ensure_ascii=False preserves non-English text
        json.dump(merged_data, out_f, indent=4, ensure_ascii=False)

    # Output execution statistics
    print("Data processing complete.")
    print("-" * 30)
    print(f"Total files scanned: {len(json_files)}")
    print(f"Empty files skipped: {empty_file_count}")
    print(f"Unique records merged: {processed_count}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Final file saved as: {output_json}")


if __name__ == "__main__":
    # Update these paths as needed
    INPUT_DIR = "./egypt_news_scraped_json_files" 
    OUTPUT_FILE = "egypt_news_data.json"
    
    if os.path.exists(INPUT_DIR):
        merge_json_to_json(INPUT_DIR, OUTPUT_FILE)
    else:
        print(f"Directory '{INPUT_DIR}' does not exist. Please update the path.")