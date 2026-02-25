# Class for statistics LLM module that generates and executes statistical analysis code based on user question classification
import pandas as pd 
import numpy as np
from scipy import stats
import sklearn
import traceback
from src.utils import clean_code, is_code_safe, run_with_timeout
from collections import Counter

class StatisticsModule:
    """
    Handles statistical analysis requests. 
    Uses LLM to suggest statistical tests in python code based on dataset summary and user question.
    Returns a tabular or numeric result.
    Uses conversation history for context
    """

    def __init__(self, df, llm_client):
        self.df = df
        self.llm_client = llm_client

    def handle(self, question, df_summary, messages=None):

        if messages is None:
            messages = []

        system_prompt = f"""
        You are a python statistical analysis assistant.
        Given the dataframe summary below, generate **only executable Python code** that answers the user's statistical question.

        **Rules**        
        - The input dataframe is named 'df'.
        - Assign the final answer to a variable named 'result'.
        - Each row corresponds to one subject/sample.
        - Do **not** include any import statements — assume `pandas (pd)`, `numpy (np)`, `scipy.stats (stats)` and `scikit-learn (sklearn)` are already imported.
        - Only use columns and data types shown in the summary. Do **not** assume any others.
        - You may create temporary variables, but never modify or overwrite 'df'.
        - **SAFETY**: Never include code that writes to disk or removes files, accesses the network, or executes system commands.

        **Data Handling**
        - For comma-separated columns, always use:  
            `str.split(r'\\s*,\\s*', regex=True)` followed by `explode()` only when required.
        - When exploding **multiple columns**, they MUST come from the **same original DataFrame** so co-occurring values remain aligned.
            *Never* explode two different DataFrames and align them by index.
        - NEVER create contingency tables from independently exploded columns.
            Always explode **within the same DataFrame** before grouping.
        - Always drop rows containing NA values *only for the variables used in the test*. Do not treat NAs as 0 or impute them.
        - For correlation tests, ensure both vectors have equal length after NA removal.
        - Ensure all column names are unique to avoid errors.

        **Output**
        - For any statistical test including t-test, Mann–Whitney, chi-square, ANOVA, Kruskal–Wallis, correlations:
            • Return a **single-row pandas DataFrame** assigned to `result`, unless instructed otherwise.
            • The DataFrame columns MUST be named after the statistics returned.

        - Standard output formats:
            * Generic statistical test:
            result = pd.DataFrame({{'statistic': [stat],'p_value': [p_value]}})

            * Correlation tests (Pearson/Spearman):
            result = pd.DataFrame({{'correlation': [corr],'p_value': [p_value]}})

        - For COUNT-type questions requiring a single numeric answer:
            Assign the scalar directly:
            result = some_number

        - If the requested analysis cannot be completed using the available schema, assign a short explanatory **string** to `result`. Do not include coding keywords.

        - Output **only** executable Python code — no markdown, comments, or explanations.

        {df_summary}
        """

        # Build messages for LLM
        full_messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        full_messages.extend(messages)

        # Add current user question 
        full_messages.append({"role": "user", "content": question})

        # Generate and clean up code
        code = self.llm_client.generate(messages=full_messages)
        code = clean_code(code)

        return code

    def execute_code(self, code):
        # Executes LLM-generated statistical code safely and returns the result

        # Restrict variables accessible during execution
        try:
            # Comprehensive safe builtins for statistical operations
            safe_builtins = {
                # Type constructors
                "list": list, "dict": dict, "set": set, "tuple": tuple,
                "str": str, "int": int, "float": float, "bool": bool,
                "frozenset": frozenset, "bytes": bytes,
                
                # Iteration & functional programming
                "range": range, "enumerate": enumerate, "zip": zip,
                "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
                
                # Aggregation
                "len": len, "sum": sum, "min": min, "max": max,
                "all": all, "any": any,
                
                # Math & rounding (critical for statistics)
                "abs": abs, "round": round, "pow": pow, "divmod": divmod,
                
                # Type introspection
                "type": type, "isinstance": isinstance, "issubclass": issubclass,
                "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
                "callable": callable,
                
                # String operations
                "ord": ord, "chr": chr, "repr": repr, "ascii": ascii,
                
                # Exceptions (important for statistical edge cases)
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "KeyError": KeyError,
                "IndexError": IndexError, "AttributeError": AttributeError,
                "ZeroDivisionError": ZeroDivisionError, "RuntimeWarning": RuntimeWarning,
                
                # Utilities
                "print": print, "format": format, "hash": hash,
                "id": id, "hex": hex, "bin": bin, "oct": oct,
                
                # Slicing
                "slice": slice,
            }
            
            # Provide a copy of the dataframe to avoid modifications
            safes = {"pd": pd, "np": np, "stats": stats, "Counter": Counter, "sklearn": sklearn,
                     "__builtins__": safe_builtins,"df" : self.df.copy()} 

            # execute the generated code (it should assign the output to variable `result`)
            # First check if code is safe to execute using keywords and pattern matching (utils.py)
            if not is_code_safe(code):
                return {
                    "type": "text",
                    "code": None,
                    "data": "The generated code may contain unsafe operations and will not be executed. Please try again."
                }
            
            # Run in separate thread with timeout to prevent infinite loops or excessively long execution
            result = run_with_timeout(code, safes, timeout = 30)

            if result is None:
                return {
                    "type": "error",
                    "code": code,
                    "data": "No variable named 'result' found. Please ensure your code assigns output to 'result'."
                }
            
            # Handle other result types
            elif isinstance(result, (int, float, np.number, np.generic)):
                display_data = float(result)
                type_str = "number"
            elif isinstance(result, pd.DataFrame):
                display_data = result
                type_str = "dataframe"
            elif isinstance(result, pd.Series):
                # Convert series to DF with index levels 
                display_data = result.reset_index()
                value_col = result.name if result.name else "value"
                index_cols = [
                    c if (isinstance(c, str) and c.strip() != "") else f"index_{i}"
                    for i, c in enumerate(result.index.names or [])
                ]
                # If the Series has no index names, assign generic ones
                if not index_cols or any(name is None for name in index_cols):
                    index_cols = [f"index_{i}" for i in range(display_data.shape[1] - 1)]
                display_data.columns = index_cols + [value_col]
                type_str = "dataframe"
            else:
                display_data = str(result)
                type_str = "text"

            return {"type": type_str,
                    "code": code,
                    "data": display_data}
        
        except TimeoutError:
            return {
                "type": "error",
                "code": code,
                "data": "Code execution timed out. The operation may be too complex or inefficient. Please try a simpler question or check the code for potential infinite loops."
            }
        
        except Exception as e:

            err = traceback.format_exc()

            return {
                "type": "error",
                "code": code,
                "data": f"Error executing statistical code: {e}\n\n{err}"
                }
