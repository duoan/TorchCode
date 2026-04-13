import json
import os
import glob
import re

def parse_notebook_template(filepath: str) -> dict:
    """
    Parses a Jupyter Notebook template to extract the problem description and initial code.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        description = ""
        initial_code = ""
        
        for cell in nb.get("cells", []):
            if cell["cell_type"] == "markdown":
                # Only take the first markdown block (usually the problem statement)
                if not description:
                    source = cell.get("source", [])
                    # Filter out Colab badge lines
                    filtered_source = [
                        line for line in source 
                        if "![Open In Colab]" not in line
                    ]
                    description = "".join(filtered_source).strip()
                    
            elif cell["cell_type"] == "code":
                source = cell.get("source", [])
                source_str = "".join(source)
                
                # Check for the implementation placeholder
                if "# ✏️ YOUR IMPLEMENTATION HERE" in source_str:
                    initial_code = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\n\n" + source_str
                    
        return {
            "description": description,
            "initial_code": initial_code
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return {
            "description": "Error loading description.",
            "initial_code": "# Error loading template code."
        }

def get_all_templates(templates_dir: str = "../templates") -> dict:
    """
    Returns a dictionary mapping task_ids to their extracted template data.
    task_id is inferred from the filename (e.g., '01_relu.ipynb' -> 'relu').
    """
    templates = {}
    
    # Resolve absolute path based on current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    templates_path = os.path.join(base_dir, templates_dir)
    
    for filepath in glob.glob(os.path.join(templates_path, "*.ipynb")):
        filename = os.path.basename(filepath)
        if filename == "00_welcome.ipynb":
            continue
            
        # Extract task_id (e.g. 01_relu.ipynb -> relu)
        match = re.match(r"^\d+_(.+)\.ipynb$", filename)
        if match:
            task_id = match.group(1)
            templates[task_id] = parse_notebook_template(filepath)
            
    return templates

if __name__ == "__main__":
    # Test parser
    res = get_all_templates()
    if "relu" in res:
        print("Successfully parsed relu:")
        print("Description:", res["relu"]["description"][:100], "...")
        print("Code:", res["relu"]["initial_code"][:50], "...")
