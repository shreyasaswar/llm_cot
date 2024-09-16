import json
import streamlit as st
from openai import OpenAI
import pandas as pd

# Initialize the client with your API key
client = OpenAI(
    api_key=st.secrets["openai"]["api_key"],
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
        if sample['evaluation']:
            total_score = sample['total_score']
            if total_score > best_score:
                best_score = total_score
                best_sample = sample
    return best_sample




def get_final_answer(question, num_samples=5):
    cot_samples = generate_cot_samples(question, num_samples)
    evaluated_samples = []
    for sample in cot_samples:
        evaluation = evaluate_cot(sample['cot'])
        if evaluation:
            total_score = sum(item['score'] for item in evaluation.values())
            sample['evaluation'] = evaluation
            sample['total_score'] = total_score
        else:
            sample['evaluation'] = None
            sample['total_score'] = 0  # Or assign a default score
        evaluated_samples.append(sample)
    best_sample = select_best_cot(evaluated_samples)
    if best_sample:
        best_sample['cot_samples'] = evaluated_samples  # Include all samples in the result
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
        return {'final_answer': final_answer, 'cot': None, 'evaluation': None, 'total_score': None, 'cot_samples': evaluated_samples}




# Set page configuration
st.set_page_config(page_title="POC - Evaluating and Improving Maya Response", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Function to set custom styles
def set_custom_styles():
    st.markdown(
        """
        <style>
        /* Apply dark theme to the app */
        .main {
            background-color: #121212;
            color: #FFFFFF;
        }
        /* Adjust text color */
        h1, h2, h3, h4, h5, h6, p, div, label, span {
            color: #FFFFFF;
        }
        /* Style headings */
        h1, h2, h3, h4, h5, h6 {
            color: #FF6F61;
        }
        /* Style sections */
        .section {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        /* Style separator */
        .separator {
            border-top: 1px solid #FF6F61;
            margin: 25px 0;
        }
        /* Style buttons */
        .stButton>button {
            background-color: #FF6F61;
            color: #FFFFFF;
            border-radius: 5px;
            border: none;
        }
        /* Style slider */
        .stSlider > div > div > div {
            color: #FF6F61;
        }
        /* Style success message */
        .stAlert {
            background-color: #2E2E2E;
            color: #FFFFFF;
            border-left: 5px solid #FF6F61;
        }
        /* Style tables */
        .stTable {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }

        /* Footer style */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #121212;
            color: grey;
            text-align: center;
            padding: 10px 0;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_styles()

# Create columns
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.title("🩺 MAYA")
    st.markdown("""
    Welcome to the **MAYA** chatbot! This application uses advanced **Chain-of-Thought reasoning** to provide accurate answers to your medical questions.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Default question
    default_question = "What are the potential side effects of prolonged use of corticosteroids?"
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    question = st.text_input("💬 **Enter your medical question here:**", value=default_question)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    num_samples = st.slider("🧠 **Select the number of reasoning paths to generate:**", min_value=1, max_value=5, value=3)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    if st.button("✨ Get Answer"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Generating the best answer..."):
                    result = get_final_answer(question, num_samples=num_samples)

                # Display results in sections
                st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
                st.markdown("## 🏆 Final Answer")
                st.success(result['final_answer'])

                st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
                st.markdown("## 🔗 Chain-of-Thought Reasoning")
                st.markdown(f"<div class='section'>{result['cot']}</div>", unsafe_allow_html=True)

                st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
                st.markdown("## 📊 Evaluation Scores")
                if result['evaluation']:
                    evaluation_df = pd.DataFrame.from_dict(result['evaluation'], orient='index')
                    st.dataframe(evaluation_df.style.set_properties(**{'background-color': '#1E1E1E', 'color': 'white'}))
                    st.write(f"**Total Score:** {result['total_score']}")
                else:
                    st.write("No evaluation scores available.")

                # Footer Section
                st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class='footer'>
                    Developed by <a href="https://www.linkedin.com/in/shreyasaswar/" target="_blank" style="color: grey;"><strong>Shreyas Aswar</strong></a>
                    </div>
                    """                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("### 📝 What to Expect")
    st.markdown("""
    **Ask a medical question** and click **"Get Answer"** to receive:

    - **Final Answer**: The most accurate response generated.
    - **Reasoning Process**: Step-by-step thoughts leading to the answer.
    - **Evaluation Scores**: Assessment of the reasoning based on key criteria.

    **How It Works:**

    This app enhances responses by:

    - Generating multiple reasoning paths (Chain-of-Thought).
    - Evaluating each path using predefined metrics.
    - Selecting the best answer based on evaluation scores.

    **Key Concepts:**

    - **Chain-of-Thought (CoT)**: Encourages step-by-step reasoning.
    - **Response Evaluation**: Improves answer quality by scoring reasoning.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
