# Class for LLM module that fixes code that raised an error
from src.utils import clean_code

class ErrorCorrector:
    def __init__ (self, llm_client):
        self.llm_client = llm_client

    def handle(self, question, error_message, original_code, df_summary):
        system_prompt = f"""
You are a python debugging assistant. Given the users question, dataframe summary,
the original python code, and the error traceback, fix the code so it runs successfully 
and still answers the question. Return ONLY valid, correct python code using the variable
'df' for the dataframe and the result assigned to the variable 'result'.
Do not include any explanations, only return the corrected code.
"""
        user_prompt = f"""

User Question:
{question}

Original Code:
{original_code}

Error Traceback:
{error_message}

Dataframe Summary:
{df_summary}
            """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Generate and clean up code
        corrected_code = self.llm_client.generate(messages=messages)
        corrected_code = clean_code(corrected_code)

        return corrected_code