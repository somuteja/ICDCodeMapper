import csv


def create_embedded_text(code, long_description, category_title):
    return f"""<code>{code}</code>
<description>{long_description}</description>
<category>{category_title}</category>
<system>ICD-10-PCS</system>"""


def preprocess_icd10_pcs_data(input_file, output_file):
    preprocessed_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)

        for row in reader:
            code = row[0].strip()
            long_description = row[1].strip()

            # Root operation is the first word of the description (e.g. "Bypass", "Excision")
            category_title = long_description.split()[0] if long_description else ""

            embedded_text = create_embedded_text(code, long_description, category_title)

            record = {
                "code": code,
                "code_dotted": code,  # PCS codes have no dot notation
                "long_description": long_description,
                "category_title": category_title,
                "embedded_text": embedded_text,
                "system": "ICD-10-PCS",
                "query_type": "procedure",
            }

            preprocessed_data.append(record)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['code', 'code_dotted', 'long_description', 'category_title', 'embedded_text', 'system', 'query_type']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(preprocessed_data)

    print(f"Preprocessed {len(preprocessed_data)} ICD-10-PCS codes")
    print(f"Output saved to: {output_file}")

    if preprocessed_data:
        print("\nSample record:")
        sample = preprocessed_data[0]
        for key, value in sample.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    input_file = "icd_10_pcs.csv"
    output_file = "icd_10_pcs_preprocessed.csv"

    preprocess_icd10_pcs_data(input_file, output_file)
