# sci-tech-bot
Conversational Chatbot for Science and Technology, General Knowledge leveraging the SQuAD (Stanford Question Answering) Dataset

**Table of Contents**

1. [Overview](#overview)  
2. [Objectives](#objectives)  
3. [Problem Statement](#problem-statement)  
4. [Solution](#solution)  
5. [Dataset](#dataset)  
6. [Methods Used](#methods-used)  
7. [Technologies](#technologies)  
8. [Installation](#installation)  
9. [Usage](#usage)  
10. [Team Members](#team-members)  
11. [Acknowledgments](#acknowledgments)  
12. [License](#license)  

---

### Overview
This project aims to develop a conversational chatbot that specializes in answering questions related to Science and Technology, General Knowledge by
leveraging the Stanford Question Answering 1.1 Dataset (SQuAD). The chatbot is designed to offer personalized learning experiences through natural dialogue interactions,
engaging users with follow-up questions and detailed responses.

### Objectives
- Explore, filter, and refine the Stanford Q&A (SQuAD) 1.1 dataset.
- Develop a conversational chatbot specialized in Science and Technology, General Knowledge.
- Create an interactive experience that caters to personalized learning needs.
- Build a chatbot that can handle follow-up questions and maintain a natural dialogue flow.

### Problem Statement
Students often require personalized learning experiences based on their unique interests and learning pace.
Traditional teaching methods and curriculum structures cannot fully address individual learning needs.
There is a need for an engaging solution that provides personalized, topic-specific learning support.
A conversational chatbot that specializes in answering questions related to Science and Technology can offer a more engaging and interactive learning experience.

### Solution
The solution is to create a conversational chatbot capable of interacting naturally with users and providing answers to Science and Technology questions. It will:
- Use natural dialogue conversations to mimic human interaction.
- Answer follow-up and contextual questions for a deeper understanding.
- Leverage the SQuAD 1.1 dataset for training and evaluating its question-answering capabilities.

### Dataset
We are utilizing the Stanford Question Answering Dataset (SQuAD), a popular large-scale dataset for NLP-based question-answering tasks.
It contains questions derived from Wikipedia articles and is widely used for training and benchmarking machine learning models in the Q&A domain.

**Dataset link:** 
https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset
For more information on Stanford Question Answering SQuAD Dataset
[Stanford Q&A Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/)

### Methods Used
- Data Exploration and Filtering
- Natural Language Processing
- Machine Learning
- Dialogue Management
- Python Programming
- TKInter in python for user interface

### Technologies
- Stanford Question Answering Dataset (SQuAD) 1.1
- Python Libraries (TensorFlow, PyTorch, NLTK, TKInter, HuggingFace Transformers, Pandas, Dataset, Scikit-learn, better-profanity)
- Chatbot Frameworks 
- APIs for Natural Language Processing

### Installation

To set up the conversational chatbot project, follow these steps:

1. **Clone the Repository:**
   Open a terminal and run the following command to clone the repository to your local machine:
   ```bash
   git clone https://github.com/maheshbabu-usd/aai-520-group06-genai-bot.git
   ```

2. **Navigate to the Project Directory:**
   Change into the project directory:
   ```bash
   cd aai-520-group06-genai-bot/Code/Chatbot
   ```

3. **Set Up a Virtual Environment (Optional but Recommended):**
   Create a virtual environment to manage dependencies. You can use `venv` for this purpose:
   ```bash
   python -m venv venv
   ```
   Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**
   Install the required Python packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the SQuAD Dataset:**
   Ensure that the SQuAD dataset is available at the specified path in the code. The dataset is typically loaded directly via the `datasets` library,
   but if you need a local copy, you can download it from https://www.kaggle.com/datasets/stanfordu/stanford-question-answering-dataset
   for latest version of SQuAD refer to [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/).

### Usage

To run the chatbot application, follow these steps:
1. **Install the pre-requisites**
   Install Dependencies using (pip install -r requirements.txt)
2. **Ensure Path to Dataset**
   Check and ensure the SQuAD_1.1_QandA_Indexed.csv is available in the Dataset sub folder
3. **Launch the Application:**
   In the terminal, while still in the project directory, execute the following command:
   ```bash
   python chatbot.py
   ```

2. **Interact with the Chatbot:**
   - A Graphical User Interface (GUI) window will open.
   - You can type your queries related to Science & Technology and General Knowledge in the input field.
   - Press **Enter** or click the arrow button to send your queries to chatbot and receive answers.

3. **Features:**
   - Apart from questions available in the Stanford SQuAD 1.1 dataset, the chatbot supports follow-up questions on the fly related to the context
   - Maintains a natural dialogue flow.
   - You can view context, summarize it or switch to new topic
   - You can clear the conversation and start afresh
   - Chatbot also performs profanity check and doesnot allow profane language in the conversation

### Team Members

Mahesh Babu • Keerthana • Paritosh Umeshan

### Acknowledgments
We would like to thank to our Instructor and project advisor Haisav Chokshi for his invaluable guidance and support throughout the project.
Special thanks to the creators of the SQuAD dataset for providing such a valuable resource for our research.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

