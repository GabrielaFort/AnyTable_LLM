# Class for tableqa LLM module that generates and executes query code based on user question classification
import pandas as pd
import numpy as np
import traceback
from src.utils import clean_code, is_code_safe, run_with_timeout
from scipy import stats
from collections import Counter

class TableQAModule:
    def __init__(self, df, llm_client):
        self.df = df
        self.llm_client = llm_client

    def handle(self, question, df_summary, messages=None):
        """
        Handle query questions with conversation history.
        Args:
            question: User's current question
            messages: Previous conversation messages
            df_summary: Summary of df schema from summarize_dataframe
        Returns:
            Dict with type, data, and code.
        """
        if messages is None:
            messages = []

        # Prepare system prompt
        system_prompt = f"""
        You are a python data analysis assistant.
        Given the dataframe summary below, write executable python code using pandas to answer the user's question.
        
        **Rules**
        - Use dataframe name 'df'.
        - Assign the final answer to 'result'.
        - Do **not** include any import statements — assume `pandas (pd)`, `numpy (np)`, and `scipy.stats (stats)` are already imported.
        - Only use columns and data types shown in the summary. Do **not** assume others. 
        - You may create temp DataFrames/Series/columns but never modify 'df'.
        - Handle comma-separated values using `str.split(r'\\s*,\\s*', regex=True)` or `explode()`.
        - If exploding multiple columns, pad shorter lists with `None` before exploding.
        - Prefer using `.str.contains(..., case=False, na=False)` for text matching — particularly for comma-separated values.
        - Exclude missing values safely, but do not drop rows unless necessary.
        - Unless explicitly instructed otherwise, always prefer returning a DataFrame that preserves the full schema of the original data (same column names, same order) — especially for
          filtering or lookup-style questions.
        - For counts/comparisons, group logically and use '.nunique()' where appropriate.
        - Ensure all column names are unique to avoid errors.
        - Include columns in the output that provide context for the answer. The answer must make sense on its own given the question.
            - **FOR EXAMPLE**, if the question is "What are the top 3 most commonly reported irAEs?" - the output should include columns for BOTH the irAE name and the count, not just the count.
        - **SAFETY**: Never include code that writes to disk or removes files, accesses the network, or executes system commands.

        **Output**
            - a DataFrame subset/summary that answers the question, **or**
            - a single numeric/scalar value if suitable.
        - If unanswerable from schema, assign a short explanatory **string** to 'result'. Do not include coding keywords.
        - Output **only** executable **Python code**, no markdown or explanations.

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
        # Executes LLM-generated query code safely and returns the result
        
        # Restrict variables accessible during execution
        try:
            # Comprehensive safe builtins for data querying operations
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
                
                # Math & rounding
                "abs": abs, "round": round, "pow": pow, "divmod": divmod,
                
                # Type introspection (needed for pandas/numpy operations)
                "type": type, "isinstance": isinstance, "issubclass": issubclass,
                "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
                "callable": callable,
                
                # String operations
                "ord": ord, "chr": chr, "repr": repr, "ascii": ascii,
                
                # Exceptions (for error handling in generated code)
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "KeyError": KeyError,
                "IndexError": IndexError, "AttributeError": AttributeError,
                "ZeroDivisionError": ZeroDivisionError,
                
                # Utilities
                "print": print, "format": format, "hash": hash,
                "id": id, "hex": hex, "bin": bin, "oct": oct,
                
                # Slicing
                "slice": slice,
            }

            safes = {"pd": pd, "np": np, "stats": stats, "Counter": Counter,
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
                display_data = result.replace("", pd.NA)
                type_str = "dataframe"

            elif isinstance(result, pd.Series):
                # Convert series to DF with index levels 
                display_data = result.reset_index(drop=True).to_frame()
                display_data = display_data.replace("", pd.NA)
                type_str = "dataframe"

            # Lists and NumPy arrays
            elif isinstance(result, np.ndarray):
                if result.ndim == 1:
                    display_data = pd.DataFrame(result, columns=["value"])
                    display_data = display_data.replace("", pd.NA)
                    type_str = "dataframe"
                else:
                    display_data = pd.DataFrame(result)
                    display_data = display_data.replace("", pd.NA)
                    type_str = "dataframe"

            elif isinstance(result, list):
            # Convert simple lists to a single-column DataFrame
                try:
                    arr = np.array(result, dtype=object)
                    if arr.ndim == 1:
                        display_data = pd.DataFrame(arr, columns=["value"])
                        display_data = display_data.replace("", pd.NA)
                        type_str = "dataframe"
                    else:
                        display_data = pd.DataFrame(arr)
                        display_data = display_data.replace("", pd.NA)
                        type_str = "dataframe"
                except Exception:
                    display_data = pd.DataFrame(result, columns=["value"])
                    display_data = display_data.replace("", pd.NA)
                    type_str = "dataframe"

            elif isinstance(result, pd.Index):
                display_data = result.tolist()
                display_data = pd.DataFrame(display_data, columns=["value"])
                display_data = display_data.replace("", pd.NA)
                type_str = "dataframe"

            elif isinstance(result, set):
                display_data = pd.DataFrame(sorted(result), columns=["value"])
                display_data = display_data.replace("", pd.NA)
                type_str = "dataframe"
            
            elif isinstance(result, dict):
                display_data = pd.DataFrame(result)
                display_data = display_data.replace("", pd.NA)
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
            return {"type": "error",
                    "code": code,
                    "data": f"Error executing query: {e}\n\n{err}"
                    }