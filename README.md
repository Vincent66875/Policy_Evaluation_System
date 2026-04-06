# Privacy Policy Persona Simulator

This program is a **human-centric simulation engine** designed to evaluate the understandability of privacy policies. Unlike standard LLM benchmarks that aim for perfect accuracy, this tool uses **Persona-Based Prompting** and **Stochastic Bias Layers** to mimic how different demographic groups perceive and misinterpret legal jargon.

---

## 🚀 Key Features

* **Multi-Persona Modeling**: Simulates three distinct archetypes:
    * **Expert**: High technical literacy, precise reading.
    * **Human Avg**: Average literacy; prone to skimming and missing nuances.
    * **Non-Expert Seniors**: Cautious readers easily overwhelmed by jargon.
* **Cognitive Bias Simulation**: Implements human-like flaws including:
    * **Jargon Penalty**: Non-experts struggle with terms like "ISO27001".
    * **Skimming Bias**: Models how users focus on headings while ignoring fine print.
    * **Uncertainty/Omission**: Simulates the tendency to choose "Cannot be determined" when confused.
* **Dynamic Memory Stream**: Personas "remember" previous reflections, allowing for consistent behavior across trials.
* **Perceptual Masking**: Automatically "fuzzes" technical terms for non-experts (e.g., changing "ISO27001" to "a technical code") to simulate real-world comprehension gaps.

---

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/yourusername/persona-simulator.git](https://github.com/yourusername/persona-simulator.git)
    cd persona-simulator
    ```

2.  **Install dependencies**:
    ```bash
    pip install openai
    ```

3.  **Set up Environment Variables**:
    ```bash
    export OPENAI_API_KEY='your-api-key-here'
    # Optional: export OPENAI_MODEL='gpt-4o'
    ```

---

## 💻 Usage

Run the simulation via the command line. You can specify the number of trials and the random seed for reproducibility.

```bash
python main.py --trials 30 --seed 7