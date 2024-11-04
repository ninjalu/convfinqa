import re


def get_qa_template() -> str:
    return """
    You are an expert in reading and understanding financial documents, and doing financial calculations.
    You task is to analyse the contexts given in texts and answer the questions based on the information provided only.
    Please think step by step reasoning and or calculation by:
    Step 1: List facts from the context that can help us find out the answer that is in their original form.
    Please be aware that the facts could come from the text or the facts extracted from the table.
    Step 2: reasoning or calculation.
    The output should follow a json format like this with all double quotes:
    ```{{"answer": <the answer here>,"contexts": <list facts in step 1>}}```.
    The numeric values can be in the forms of integers, floats, or percentages (please use %). 
    If the numeric value is float or percentage, please keep only one decimal point
    So if the answer is 0.123, you should return 0.1, if the answer is 12.34%, you should return 12.3%
    They could also carry units, especially monetary units such as $.
    If you can't find the facts from the context to answer the question, output {{'prediction': 'No answer found'}}
    <context>
    {context}
    Question {question}
    """
    # The final sentence follow this format with the numeric value highlighted with *** like this:


def get_table_str_template() -> str:
    return """
    You are an expert in reading and understanding financial tables.
    Your task is to extract concise facts from a table, and list them as a series of plain sentences.
    The first row are the columns, the first column are the row names.
    They facts should only deviate from the table provided
    Table is provided in the format of html code.
    Table: {element}
    """


def clean_table(table: list) -> list:
    table[0] = [""] + table[0]
    seen = set()
    cleaned_table = []
    for row in table:
        cleaned_row = []
        for i, cell in enumerate(row):
            if i == 0 or cell not in seen:
                cleaned_row.append(cell)
                if i != 0:
                    seen.add(cell)
        cleaned_table.append(cleaned_row)

    return cleaned_table


def prep_table(table: list) -> str:
    cleaned_table = clean_table(table)
    html_table = '<table border="1">\n'
    for row in cleaned_table:
        html_table += "  <tr>\n"
        for cell in row:
            html_table += f"    <td>{cell}</td>\n"
        html_table += "  </tr>\n"
    html_table += "</table>"
    return html_table


def prep_doc(doc: dict) -> list:
    pre_texts = doc["pre_text"]
    post_texts = doc["post_text"]
    table = prep_table(doc["table"])
    context = pre_texts + [table] + post_texts
    return context


def extract_json_from_string(input_string):
    json_pattern = r"\{.*?\}"
    json_matches = re.findall(json_pattern, input_string, re.DOTALL)
    return json_matches[0] if json_matches else ""


def split_questions(data: list) -> list:
    new_data = []
    for item in data:
        # Find all keys that start with 'qa'
        qa_keys = [key for key in item.keys() if key.startswith("qa")]

        # Create a new dictionary for each question
        for qa_key in qa_keys:
            new_item = {k: v for k, v in item.items() if not k.startswith("qa")}
            new_item["qa"] = item[qa_key]
            new_data.append(new_item)

    return new_data
