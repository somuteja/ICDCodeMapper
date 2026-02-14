import csv
import json

def add_dot_to_code(code):
    """Convert code like 'A000' to 'A00.0'"""
    return f"{code[:3]}.{code[3:]}"

def create_embedded_text(code_dotted, long_description, category_title):
    """Create the embedded text in the specified XML-like format"""
    return f"""<code>{code_dotted}</code>
<description>{long_description}</description>
<category>{category_title}</category>
<system>ICD-10</system>"""

def preprocess_icd10_data(input_file, output_file):
    """
    Preprocess ICD-10 codes from CSV to the required format
    """
    preprocessed_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            code = row['code']
            code_dotted = add_dot_to_code(code)
            long_description = row['long_description']
            category_title = row['category_title']

            embedded_text = create_embedded_text(code_dotted, long_description, category_title)

            
            record = {
                "code": code,
                "code_dotted": code_dotted,
                "long_description": long_description,
                "category_title": category_title,
                "embedded_text": embedded_text,
                "system": "ICD-10",
                "query_type": "diagnosis"
            }

            preprocessed_data.append(record)

    # Save as CSV file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['code', 'code_dotted', 'long_description', 'category_title', 'embedded_text', 'system', 'query_type']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(preprocessed_data)

    print(f"Preprocessed {len(preprocessed_data)} ICD-10 codes")
    print(f"Output saved to: {output_file}")

    # Print a sample record
    if preprocessed_data:
        print("\nSample record:")
        sample = preprocessed_data[0]
        for key, value in sample.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    input_file = "icd10_codes.csv"
    output_file = "icd10_codes_preprocessed.csv"

    preprocess_icd10_data(input_file, output_file)
