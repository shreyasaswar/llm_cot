# llm_cot
An attempt in mimicking the opneAI's Strawberry models chain of thought process to get more accurate responses.  


Welcome to the MAYA Medical Assistant with Enhanced Reasoning project! This application leverages advanced Large Language Models (LLMs) and Chain-of-Thought (CoT) reasoning techniques to provide accurate and contextually relevant answers to medical questions. Our focus is on improving LLM responses through systematic evaluation and feedback mechanisms.

## 🌟 Key Features

- **Interactive Q&A Interface**: Engage with a sophisticated medical AI assistant.
- **Chain-of-Thought (CoT) Reasoning**: Witness the step-by-step thought process of the LLM.
- **Multi-Path Reasoning**: Generate multiple reasoning paths for each query.
- **Automated Response Evaluation**: Assess LLM outputs based on predefined criteria.
- **Feedback Loop**: Implement a system to improve LLM performance over time.
- **User-Friendly UI**: Dark-themed interface with clear section distinctions.

## 🧠 How It Works

1. **Input**: User submits a medical question.
2. **CoT Generation**: The LLM generates multiple Chain-of-Thought reasoning paths.
3. **Evaluation**: Each reasoning path undergoes automated assessment based on:
   - Logical Consistency
   - Factual Accuracy
   - Relevance to the Query
   - Completeness of Response
   - Clarity of Explanation
4. **Selection**: The highest-scoring reasoning path is chosen.
5. **Output**: The final answer and reasoning process are presented to the user.

## 🔬 LLM Improvement Techniques

### Chain-of-Thought (CoT) Prompting
- Encourages the LLM to break down complex problems into steps.
- Improves transparency and allows for better error detection.
- Enhances the model's ability to handle multi-step reasoning tasks.

### Multiple Reasoning Paths
- Generates diverse approaches to the same problem.
- Increases the likelihood of finding an optimal solution.
- Allows for comparative analysis of different reasoning strategies.

### Automated Evaluation System
- Implements a scoring mechanism for each reasoning path.
- Utilizes predefined criteria to assess response quality.
- Mimics reinforcement learning principles for continuous improvement.

### Feedback Loop Integration
- Collects user feedback on the quality and accuracy of responses.
- Incorporates feedback into the evaluation system for future queries.
- Aims to fine-tune the LLM's performance over time.

## 🛠️ Technical Stack

- **LLM**: OpenAI GPT model (via API)
- **Frontend**: Streamlit
- **Backend**: Python
- **Evaluation Engine**: Custom-built scoring system

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/medical-assistant-llm.git
   cd medical-assistant-llm
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   - Create a `.streamlit/secrets.toml` file
   - Add your API key: `openai_api_key = "YOUR_API_KEY_HERE"`

### Running the App

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser to interact with the app.

## 📊 Evaluation Metrics

Our evaluation system assesses each reasoning path on a scale of 1-10 for each criterion:

1. **Logical Consistency**: Coherence and validity of the reasoning.
2. **Factual Accuracy**: Correctness of medical information provided.
3. **Relevance**: Alignment with the user's query.
4. **Completeness**: Thoroughness of the explanation.
5. **Clarity**: Ease of understanding for the end-user.

The total score determines the best reasoning path to be presented to the user.

## 🔄 Continuous Improvement Cycle

1. **Generate**: Create multiple CoT reasoning paths.
2. **Evaluate**: Score each path using our automated system.
3. **Select**: Choose the highest-scoring path.
4. **Present**: Display the chosen response to the user.
5. **Collect Feedback**: Gather user input on response quality.
6. **Analyze**: Review feedback and evaluation scores.
7. **Refine**: Adjust prompts and evaluation criteria based on insights.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 📬 Contact

For questions or support, please contact shreyas.aswar21@gmail.com or open an issue on GitHub.



**Disclaimer**: This application is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
