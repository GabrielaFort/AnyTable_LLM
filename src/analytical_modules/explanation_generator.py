# Class for LLM module that generates a plain language explanation of what the generated code does, based on the previous question and generated code.
class ExplanationGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def _extract_last_turn(self, messages):
        last_code = None
        last_question = None

        # Find most recent assistant code generation and user question
        # Most recent = last occurance in messages list
        for msg in reversed(messages):
            role = msg.get("role")
            content = msg.get("content")

            # Find last assistant message
            if last_code is None and role == "assistant" and content:
                last_code = content
                continue
            # Find last user message before that assistant message
            if last_code is not None and role == "user" and content:
                last_question = content
                break

        # Fallback logic - return most recent user message in the whole history
        if last_question is None:
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    last_question = msg.get("content")
                    break   

        return last_question, last_code
    

    def generate_explanation(self, messages):
        """
        Generate a short explanation of what the code does.
        Will only be run if the previous question resulted in code generation.
        Args:
            messages - list of previous conversation messages
        """
        system_prompt = f"""
Given a user's question and the Python code that was generated to answer the question,
provide a brief, clear explantaion of what the analysis does.

Rules:
- Keep it 1-2 sentences maximum.
- Use plain language, no technical jargon or code syntax.
- Focus on WHAT was done, not HOW.
- Explain what YOU did to answer the question.
- CRITICAL: ONLY INCLUDE THE SHORT EXPLANATION. Do **not** include the original question or generated code.
- Do NOT include any thoughts or reasoning steps.

Examples:
Question: "Show me lung cancer patients with a rash"
Code: df[(df['tumor_type'] == 'lung cancer') & (df['irae'].str.contains('rash', case=False))]
Explanation: "I filtered the dataset to show only lung cancer patients who experienced a rash as an irAE."

Question: "Test for an association between irae type and sex"
Code: df2 = df[['irae_type','sex']].dropna()\ndf2['irae_type'] = df2['irae_type'].astype(str).str.split(r'\\s*,\\s*')\ndf2 = df2.explode('irae_type')\ntab = pd.crosstab(df2['irae_type'], df2['sex'])\nchi2, p_value, dof, _ = stats.chi2_contingency(tab)\nresult = pd.DataFrame({{'statistic':[chi2], 'p_value':[p_value]}})
Explanation: "I checked whether the distribution of different irAE types varies between male and female patients by building a table of counts and running a chi‑square test of independence.
"""

        # Build messages for LLM
        last_question, last_code = self._extract_last_turn(messages)

        if not last_code:
            # If we can't find any code in the history, we can't generate an explanation. Return empty string.
            return ""
        
        full_messages = [{"role": "system", "content": system_prompt},{"role": "user", "content": f"Question:{last_question}\nCode:{last_code}\n\nExplanation:"}]

        # Generate explanation
        explanation = self.llm_client.generate(messages=full_messages)

        # Clean up the explanation (remove quotes, extra whitespace)
        explanation = explanation.strip().strip('"').strip("'").strip()
        
        return explanation
