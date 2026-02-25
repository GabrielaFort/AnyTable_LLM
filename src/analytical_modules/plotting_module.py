# Class for plotting LLM module that generates and executes plotting code based on user question classification
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for safe plotting

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import traceback
from scipy import stats
from src.utils import clean_code, is_code_safe, run_with_timeout
import plotly.express as px
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly
from matplotlib_venn._common import VennDiagram
from matplotlib_venn import venn2, venn3
from collections import Counter
import threading

_mpl_lock = threading.Lock()  # Lock for matplotlib state access

class PlottingModule:
    """
    Handles visualization requests.
    Uses LLM to suggest a plotly plot based on dataset summary and user query.
    Returns a plotly figure object to display in streamlit frontend.
    Uses conversation history for context
    """
    def __init__(self, df, llm_client):
        self.df = df
        self.llm_client = llm_client

    def handle(self, question, df_summary, messages=None):

        if messages is None:
            messages = []

        # Build system prompt
        system_prompt = f"""
        You are a python plotting assistant.
        Given the dataframe summary below, write Python code using using plotly to create an informative, interactive plot that answers the user's question. 
        
        **Rules**
        - Use the dataframe variable name 'df'.
        - Assign the final figure object to a variable named 'result'.
        - In addition, create a pandas DataFrame named `plot_data` that contains the exact data used to construct the plot (one row per point/bar/etc).
        - Do **not** include any import statements — assume `pandas (pd)`, `numpy (np)`, `plotly.express (px)`, `plotly.graph_objects (go)`, `scipy.stats (stats)`, `matplotlib.pyplot (plt)`, `venn2` and `venn3` are already imported.
        - Only use columns and data types shown in the summary. Do **not** assume any others.
        - You may create temporary variables, but never modify or overwrite 'df'.
        - When splitting comma-separated values, ALWAYS assign and explode within the SAME DataFrame (e.g., df_copy['col'] = df_copy['col'].str.split(...); df_copy = df_copy.explode('col')). Never create a separate exploded Series and assign it back.
        - All axis update methods must use Plotlys plural API: use update_xaxes(), update_yaxes(). Never use update_xaxis or update_yaxis.
        - All plots must use only px.* or go.*; do not use matplotlib unless producing a Venn diagram.
        - **SAFETY**: Never include code that writes to disk or removes files, accesses the network, or executes system commands.

        **Data Handling**
        - Always handle comma-separated values using `str.split(r'\\s*,\\s*', regex=True)` and `explode()`.
        - Handle missing values safely by excluding them; never impute or alter data.
        - Ensure the plot contains meaningful data — avoid empty or all-NaN visualizations.

        **Special Cases**
        - For overlap or co-occurrence plots:
            - You may use `venn2` or `venn3` from `matplotlib_venn`.
            - Assign the resulting Matplotlib Axes object to `result` instead of a Plotly figure.
            - Do **not** attempt to recreate Venn diagrams using Plotly.

        **Output**
        - Assign the final Plotly (or Matplotlib) figure to the variable `result` and the intermediate DataFrame to `plot_data`.
        - Do not produce textual summaries or markdown.
        - If the requested plot is not feasible given the schema, assign a short explanatory **string** to `result` instead. Do not include coding keywords.
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
        # Safe execution of LLM-suggested plotting code
        # Returns figure or text result

        # Restrict variables accessible during execution
        try:
            # Comprehensive safe builtins for plotting operations
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
                
                # Type introspection
                "type": type, "isinstance": isinstance, "issubclass": issubclass,
                "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
                "callable": callable,
                
                # String operations
                "ord": ord, "chr": chr, "repr": repr, "ascii": ascii,
                
                # Exceptions
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
            
            safes = {
                "pd": pd, "np": np, "stats": stats, "sns":sns, "px":px, "go":go, "plt":plt,
                "venn2":venn2, "venn3":venn3, "Counter": Counter, 
                "__builtins__": safe_builtins, "df" : self.df.copy()   
            }

            # execute the generated code (it should assign the output to variable `result`)
            # First check if code is safe to execute using keywords and pattern matching (utils.py)
            if not is_code_safe(code):
                return {
                    "type": "text",
                    "code": None,
                    "data": "The generated code may contain unsafe operations and will not be executed. Please try again."
                }
            
            # Guard all matplotlib state access with a lock 
            with _mpl_lock:
                start_fignums = set(plt.get_fignums())
                keep_fignums = set()  # Track fignums to keep (e.g., for venn diagrams)
            
                try:
                    # Run in separate thread with timeout to prevent infinite loops or excessively long execution 
                    result = run_with_timeout(code, safes, timeout = 30)

                    # Retrieve result if defined
                    result = safes.get("result", None)
                    plot_data = safes.get("plot_data", None)

                    # If 'result' is a plotly figure
                    if hasattr(result, "to_plotly_json"): 
                        return {
                            "type": "plotly",
                            "code": code,
                            "data": result,
                            "plot_data": plot_data}

                    # If 'result' is a matplotlib Figure or Axes
                    elif isinstance(result, plt.Figure) or hasattr(result, "figure"):
                        fig = result.figure if hasattr(result, "figure") else result
                        # Check to see if venn diagram, if so keep as matplotlib object
                        if "matplotlib_venn" in code.lower() or "venn2" in code.lower() or "venn3" in code.lower():
                            if hasattr(fig, "number"):
                                keep_fignums.add(fig.number)
                            return {"type": "plot", "code": code, "data": fig, "plot_data": plot_data}
                        # Else, convert to plotly for interactive display
                        plotly_fig = mpl_to_plotly(fig)
                        plt.close(fig)
                        return {"type": "plotly", "code": code, "data": plotly_fig, "plot_data": plot_data}
                    
                    elif isinstance(result, VennDiagram):
                        fig = result.figure
                        if hasattr(fig, "number"):
                            keep_fignums.add(fig.number)
                        return {"type": "plot", "code": code, "data": fig, "plot_data": plot_data}
                    
                    # If 'result' is a string (e.g., polite note)
                    elif isinstance(result, str):
                        return {
                            "type": "text",
                            "code": code,
                            "data": result,
                            "plot_data": plot_data}
                    
                    else:
                        return {
                            "type": "text",
                            "code": code, 
                            "data": "No valid figure or text result returned. Please try again or rephrase your question.",
                            "plot_data": plot_data
                        }
                    
                # Clean up any unclosed plots 
                finally:
                    end_fignums = set(plt.get_fignums())
                    leaked_fignums = end_fignums - start_fignums
                    for fignum in leaked_fignums:
                        if fignum not in keep_fignums:
                            plt.close(fignum)

        except TimeoutError:
            return {
                "type": "error",
                "code": code,
                "data": "Code execution timed out. The operation may be too complex or inefficient. Please try a simpler question or check the code for potential infinite loops.",
                "plot_data": None
            }
        
        except Exception as e:
            err = traceback.format_exc()
            return {
                "type": "error",
                "code": code, 
                "data": f"Error creating plot: {e}\n\n{err}",
                "plot_data": None
            }
        

