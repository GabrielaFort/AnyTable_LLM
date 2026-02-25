class QuestionClassifier:
    """
    Classifies user questions into categories: 'tableqa' or 'plot' or 'stats' or 'guideline'.
    Uses LLM to interpret question if no heuristic matches are found.
    """
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def classify(self, question, messages=None):
        
        plot_keywords = ["plot", "graph", "chart", "histogram", "scatter", "volcano","heatmap", "piechart", "pie chart", "donut"]
        stat_keywords = ["statistical", "correlation", "regression", "significant", "p-value", "anova", "t-test", "mann-whitney","wilcoxon","chi-square", "average","median","mean","standard deviation","confidence interval"]
        query_keywords = ["how many", "number of", "what is the total", "total number of", "list all","list unique","find all"]
        question_lower = question.lower()

        # Keyword-based detection
        if any(word in question_lower for word in plot_keywords):
            return "plot"
        elif any(word in question_lower for word in stat_keywords):
            return "stats"
        elif any(word in question_lower for word in query_keywords):
            return "tableqa"

        # Fallback to LLM classification
        if messages is None:
            messages = []

        system_prompt = f"""
You are a routing classifier. Your job is to identify what type of task the user's question represents.

Classify the following question into ONE of these categories:

1. tableqa   — questions requiring dataframe queries, filtering, grouping, counting, comparisons, or extracting values from the dataset.
2. stats     — questions requiring statistical tests, correlations, summary statistics, confidence intervals, regressions, distributions.
3. plot      — questions requiring a figure or visualization based on the dataframe.

RULES:
- Return ONLY the category name: one of {'tableqa', 'stats', 'plot'}.
- Do NOT explain your reasoning.
- If completely irrelevant (i.e. why is the sky blue?), return 'tableqa' as a default to route to the most general agent.
        """

        # Build messages for LLM
        full_messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        full_messages.extend(messages)

        # Add current user question
        full_messages.append({"role": "user", "content": question})        
            
        classification = self.llm_client.generate(messages=full_messages).strip().lower()

        if classification in ["tableqa", "plot", "stats"]:
            return classification

        # Default to 'tableqa' if unsure
        return "tableqa"