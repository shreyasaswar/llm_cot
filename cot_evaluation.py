import os
from openai import OpenAI
import json

# Initialize the client with your API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # Alternatively, you can directly pass your API key (not recommended for security reasons)
    # api_key="YOUR_API_KEY",
)

def generate_cot_samples(question, num_samples=5):
    cot_samples = []
    for _ in range(num_samples):
        prompt = f"""
You are a knowledgeable medical assistant. When answering the following question, first generate a detailed step-by-step reasoning process. Then, provide a concise final answer based on your reasoning.

**Question:**
{question}

**Your detailed reasoning:**

"""
        response = client.chat.completions.create(
            model='gpt-4o-mini',  # Use 'gpt-4' if you have access
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,  # Higher temperature for diversity
            max_tokens=500,
            n=1,
            stop=["**Final Answer:**"]
        )
        cot = response.choices[0].message.content.strip()
        # Generate the final answer
        final_answer_prompt = f"{prompt}{cot}\n\n**Final Answer:**\n"
        final_answer_response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "user", "content": final_answer_prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        final_answer = final_answer_response.choices[0].message.content.strip()
        cot_samples.append({'cot': cot, 'final_answer': final_answer})
    return cot_samples


def evaluate_cot(cot):
    evaluation_prompt = f"""
As an expert evaluator, assess the following reasoning based on the criteria below. Provide a score from 1 to 10 for each criterion, along with a brief explanation.

**Reasoning to Evaluate:**
{cot}

**Evaluation Criteria:**
1. Logical Consistency
2. Accuracy
3. Relevance
4. Completeness
5. Clarity

**Please provide your evaluation in the following JSON format:**

{{
  "Logical Consistency": {{"score": <score>, "explanation": "<brief explanation>"}},
  "Accuracy": {{"score": <score>, "explanation": "<brief explanation>"}},
  "Relevance": {{"score": <score>, "explanation": "<brief explanation>"}},
  "Completeness": {{"score": <score>, "explanation": "<brief explanation>"}},
  "Clarity": {{"score": <score>, "explanation": "<brief explanation>"}}
}}
"""
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0,
        max_tokens=500,
    )
    evaluation_text = response.choices[0].message.content.strip()
    try:
        evaluation_scores = json.loads(evaluation_text)
        return evaluation_scores
    except json.JSONDecodeError:
        # Handle parsing error
        return None


def select_best_cot(cot_samples):
    best_score = -1
    best_sample = None
    for sample in cot_samples:
        evaluation = evaluate_cot(sample['cot'])
        if evaluation:
            total_score = sum(item['score'] for item in evaluation.values())
            sample['evaluation'] = evaluation
            sample['total_score'] = total_score
            if total_score > best_score:
                best_score = total_score
                best_sample = sample
    return best_sample



def get_final_answer(question):
    cot_samples = generate_cot_samples(question)
    best_sample = select_best_cot(cot_samples)
    if best_sample:
        return best_sample
    else:
        # Fallback to a simple answer if evaluation fails
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "user", "content": f"Answer the following question concisely:\n\n{question}"}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        final_answer = response.choices[0].message.content.strip()
        return {'final_answer': final_answer, 'cot': None, 'evaluation': None, 'total_score': None}


if __name__ == "__main__":
    question = "What are the potential side effects of prolonged use of corticosteroids?"
    result = get_final_answer(question)
    print("Final Answer:\n", result['final_answer'])
    print("\nSelected Chain-of-Thought:\n", result['cot'])
    print("\nEvaluation Scores:\n", result['evaluation'])
    print("\nTotal Score:", result['total_score'])
