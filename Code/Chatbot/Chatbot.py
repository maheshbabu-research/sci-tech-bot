# Sci-Tech Chatbot user interface implemented using tkinter
# To implement chatbot user interface
import tkinter as tk  
# working with data
import pandas as pd
# use of datasets
from datasets import Dataset
# for conversation box to view conversations in the chatbot
from tkinter import scrolledtext
# to include an image button
from tkinter import PhotoImage
# for extracting keywords from the given input query
from sklearn.feature_extraction.text import TfidfVectorizer
# for working with datasets
from datasets import load_dataset
# to use the pre-trained BERT tranformer QA pipeline
from transformers import pipeline
# for profanity check
from better_profanity import profanity

# load the required dataset
global_data_path = "./Dataset/SQuAD_1.1_QandA_Indexed.csv"

# question-answer dataframe (from .csv)
global_qa_df = pd.DataFrame()
# standard squad dataset dataframe
global_squad_df = pd.DataFrame()

# query input field
global_input_field = None
# conversation box (Scrolled Text) UI
global_conversation_box = None

# submit button (image) button to submit query
global_submit_button = None

# Chatbot Window
global_window = None

# Question-Answer Pipeline
global_qa_pipeline = None

# flag indicating if this is new topic
global_is_new_topic = True

# query index retrieved based on TF-IDF keyword search
global_qindex = None


def Init():
    """
    Init function to initial the chatbot
    Load the SQuAD dataset and setup the Q&A Pipeline
    Args:
        None

    Returns:
         pandas dataframe: QA dataframe loaded from .csv SQuAD dataset
         dataset: standard SQuAD dataset 1.1
    """

    print("Initializing Chatbot ...")
    global global_qa_pipeline

    print("loading SQuAD dataset ...")
    # load dataset
    qa_df, squad = load_squad_dataset()

    # Load the question-answering pipeline
    print("Setting up Q&A Pipeline ...")
    global_qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
    return qa_df, squad


def load_squad_dataset():
    """
    Loads the dataset from the specified path.

    Returns:
        pd.DataFrame: The loaded SQuAD dataset from .csv file.
        Dataset: the loaded SQuAD dataset from standard SQuAD dataset
    """
    import pandas as pd
    from datasets import Dataset

    # use the global variable having the data path
    global global_data_path

    df = pd.read_csv(global_data_path)

    # Display a sample of the dataset
    df.head()

    df

    # Load the SQuAD dataset
    squad = load_dataset("squad")
    return df, squad

# Function to extract the keywords (top N) from given query text
def extract_keywords_tfidf(text, top_n=3):
    """
    Extracts the keywords using the TF-IDF Vectorizer
    important function to retrieve Top N keywords
    from given text

    Args:
        text: query text having keywords
        int: indicating top N (default=3)

    Returns:
        string: First Top Keyword
        string: Second Top Keyword
        string: Third Top Keyword
    """
    try:
        # Define a single document (list of documents if more)
        documents = [text]

        if not documents:
            raise ValueError("All documents are empty or contain only stop words.")
        
        # Initialize the TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')

        # Fit and transform the document
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()

        # Get the dense representation of the TF-IDF matrix for the single document
        dense_matrix = tfidf_matrix.todense()

        # Convert it to a 1D array
        dense_array = dense_matrix[0].tolist()[0]

        # Create a dictionary of words and their TF-IDF scores
        word_tfidf_dict = dict(zip(feature_names, dense_array))

        # Sort the dictionary by TF-IDF scores in descending order
        sorted_words = sorted(word_tfidf_dict.items(), key=lambda x: x[1], reverse=True)

        # Extract the top N keywords
        keywords = [word for word, score in sorted_words[:top_n]]

        return keywords
    except ValueError as e:
        # handle Value Error during vectorization
        print(f"Error during vectorization: {e}")

        # set empty (space) strings to be returned
        keywords = [" ", " ", " "]
        return keywords

def search_keywords(keyword1, keyword2, keyword3):
    """
    Searches the indexed SQuAD dataset (.csv) file
    to obtain the context (question-answer) index

    Args:
    string: First Keyword (highest length keyword)
    string: Second Keyword (second highest length keyword)
    string: Third Keyword (third highest length keyword)
    
    Returns:
        int: The index of the matching question searched by keywords
    """
    # Iterate through each row in the DataFrame
    index = 0
    foundIndex = False

    # used to check for only 2 keywords match
    twoKeywords = False

    for index, row in global_qa_df.iterrows():
        question = row['question']
        ques_lower = question.lower()

        # Check if keyword1 is present in the question
        if keyword1 in ques_lower:
            # Check if keyword2 is also present in the same question
            if len(keyword2) > 2 and keyword2 in ques_lower:
                twoKeywords = True
                if len(keyword3) > 2 and keyword3 in ques_lower:
                  print(question)
                  # Print the matching row's index, question, and answer
                  print(f"Index: {index}")
                  print(f"Question: {row['question']}")
                  print(f"Answer: {row['answer']}\n")
                  foundIndex = True
                  break

    if foundIndex == True:
        return index
    elif twoKeywords == True:
        # There can be cases where there are only two keywords found in a short query
        print(question)
        # Print the matching row's index, question, and answer
        print(f"Index: {index}")
        print(f"Question: {row['question']}")
        print(f"Answer: {row['answer']}\n")
        return index
    else: 
        # if keyword search fails they return -1 indicating not found
        return -1


def get_context(qindex):
    """
    Get the context given the row index

    Args:
        int: row index of the context (QA) row in dataframe

    Returns:
        string: The response as having the context id and the context
        used to view the context
    """
    global global_squad_df
    # get the qa row corresponding to the index
    qa_row = global_squad_df['train'][qindex]
    # fetch the context
    context = qa_row['context']
    # fetch the unique context id
    context_id = qa_row['id']

    # construct the context id + context response 
    context_resp = "Context Id: " + context_id + "\n" + "Context :" + context.strip(' \t\'"')

    # return the response string
    return context_resp

# Function to summarize text
def summarize_text(text, num_sentences):
    """
    returns the summary of the given text

    Args:
        string: Given input text to summarize
        int: limit to number of sentences

    Returns:
        string: summary of the given input text
        used to view the summary of the context
    """
 
    # Load a pre-trained summarization model
    summarizer = pipeline("summarization")

    # Generate summary
    summary = summarizer(text, max_length=num_sentences * 20, min_length=num_sentences * 10, do_sample=False)

    # Return the summarized text
    return summary[0]['summary_text']


# Function to get the row by given id
def get_row_by_id(df, given_id):
    """
    get the context (QA) row in the SQuAD data frame

    Args:
        pd->Dataframe: Given data frame
        string: unique id of the context (row)

    Returns:
        data row: row in the data frame
        used to retrieve the context (QA) row given unique context id
    """
    # Use boolean indexing to find the row with the specific id
    result = df.loc[df['id'] == given_id]
    
    # Check if the result is empty
    if result.empty:
        print(f"No row found with id={given_id}.")
    else:
        return result


def answer_question(qindex, question):
    """
    retrieve and return the answer to the given input query (question)

    Args:
        int: context row (index into the SQuAD dataframe)
        string: input query (question) by the user

    Returns:
        string: answer to the given query (input)
        used to retrieve the answer for given query
    """
    global global_squad_df
    global global_qa_pipeline
    from datasets import load_dataset
    from transformers import pipeline

    # Get the first example from the training set
    qa_row = global_squad_df['train'][qindex]

    # Extract context and question
    context = qa_row['context']

    id = qa_row['id']
    print("id: " + id)

    print("Context:", context)
    print("Question:", question)


    # Perform inference
    # important function call to retrieve the answer for given input query
    # using the QA pipeline of the pre-trained BERT model
    result = global_qa_pipeline(question=question, context=context)

    print("Answer:", result['answer'])

    return result['answer']

    

# Function to check for profanity
def detect_profanity(text):
    """
    Detects profanity using better_profanity

    Args:
        text: user's input query
        
    Returns:
        string: Either Profanity detcted or No profanity detected
        string: Second Top Keyword
        string: Third Top Keyword
    """
    # Load the default profanity list
    profanity.load_censor_words()
    
    # Check if profanity exists
    if profanity.contains_profanity(text):
        return "Profanity detected"
    else:
        return "No profanity detected"

# Function to handle chatbot responses
def chatbot_response(user_input):
    """
    Implements the chatbot response
    Important function controls the dialogue flow
    Args:
        text: user's input query
        
    Returns:
        string: Chatbot's response text
    """
    global global_conversation_box

    # ensure the user's input query is always converts to all lower text
    user_inp_lower = user_input.lower()

    # trim any unwanted white spaces
    trimmed_text = user_inp_lower.strip('\'"')
    
    # trim any unwanted white spaces
    trimmed_text = trimmed_text.strip(' \t\'"')

    # handle simple conversation starters
    if "hello" in trimmed_text:
        return "Hello! How can I assist you today?"
    elif "how are you" in  trimmed_text:
        return "I am doing great !! How are you doing?"
    elif "who are you" in trimmed_text:
        return "I am a Generative AI Chatbot created as part of AAI-520 Course Group 06 Team Project"
    elif "bye" in trimmed_text:
        return "Goodbye! Have a nice day!"
    # if empty input
    elif (len(trimmed_text) == 0):
        # set prompt for the user
        return "Ask a question ..."
    # if clear command then clear the chat conversation
    elif("clear" in trimmed_text):
        global_conversation_box.delete('1.0', tk.END)

    else:
        # finally respond to user appropriately
        return respond_to_user(trimmed_text)

# function to respond to the user input query
def respond_to_user(user_input_msg):
    """
    Implements the chatbot response
    Important function that handles commands, query response
    Args:
        string: user's input query
        
    Returns:
        string: answer text
    """

    global global_is_new_topic
    global global_qindex

    # new topic command
    new_topic = "new topic"

    # view context command
    view_context = "view context"

    # summarize command
    summarize_word = "summarize"

    # three keywords to be searched
    kw1 = kw2 = kw3 = ""

    # ensure input query to all lowercase
    user_input_msg = user_input_msg.lower()

    # check if user input is "new topic" command
    if(user_input_msg in new_topic):
          global_is_new_topic = True
          # return resetting context response message
          answer = "Resetting context ... proceed with questions on New Topic"
          return answer;

    # check if user input message is "view context" command
    if(user_input_msg in view_context):
        # retrieve the context and return it
        answer = get_context(global_qindex)
        return answer;

    # check if the user input is "summarize" command
    if(user_input_msg in summarize_word):
        # retieve the summary of the context 
        answer = summarize_text(get_context(global_qindex),2)
        return answer;
        
    if(global_is_new_topic == True):
        # extract keywords using TF-IDF vectorizer
        keywords = extract_keywords_tfidf(user_input_msg)
        print("Keywords: ")
        for word in keywords:
            print(word)

        # ensure valid keywords to have a successful search in
        # SQuAD dataset
        if(len(keywords) < 2):
            answer = "I cannot understand your query.  Provide well-formed query."
            return answer;

        # if first keyword is not good enough, ask for a detailed query
        if(len(keywords[0]) <= 2):
            answer = "provide a more detailed query"
            return answer;
        kw1 = keywords[0]

        # if 2nd keyword is not provided
        if((len(keywords[1]) <=2)): # or (len(keywords[2]) <=2)):
            answer = "I cannot answer your query, as it is outside the scope of my knowledge"
            return answer;
        kw2 = keywords[1]

        if(len(keywords) > 2):
            kw3 = keywords[2]

        global_qindex = search_keywords(kw1, kw2,kw3)

        if(global_qindex == -1):
            answer = "I cannot answer your query, as it is outside the scope of my knowledge"
            return answer;

    answer = answer_question(global_qindex, user_input_msg)
    global_is_new_topic = False
    return answer;
    

# Function to process user input and display conversation
def send_message():
    """
    construct the response and update the chat conversation
    """
    global global_input_field
    global global_conversation_box

    # get the user input from the text field
    user_input = global_input_field.get()
    
    # ensure text is lowercase
    trimmed_text = user_input.lower();

    # trim any unwanted white spaces
    trimmed_text = trimmed_text.strip('\'"')
    trimmed_text = trimmed_text.strip(' \t\'"')

    # perform profanity check
    result = detect_profanity(trimmed_text)
  
    # Define a tag for green text
    global_conversation_box.tag_configure("green", foreground="green")

    # Define a tag for blue text
    global_conversation_box.tag_configure("blue", foreground="blue")
     
    # Append user input to the conversation box
    global_conversation_box.config(state=tk.NORMAL)
    
    # if profanity detect respond appropriately
    if(result == "Profanity detected"):
        print(result)
        global_conversation_box.insert(tk.END, "Sci-Tech Yoda: " + "Please refrain from using profane language" + "\n\n", "blue")
    else: 
        # include the user query in chat conversation only after passing profanity check
        global_conversation_box.insert(tk.END, "You: " + user_input + "\n","green")
        # Clear the input field
        global_input_field.delete(0, tk.END)

        # Get chatbot response
        bot_response = chatbot_response(user_input)
        # Append chatbot response to the conversation box
        if(type(bot_response) is not type(None) and len(bot_response)>0):
            global_conversation_box.insert(tk.END, "Sci-Tech Yoda: " + bot_response + "\n\n", "blue")

    global_conversation_box.config(state=tk.DISABLED)
    
    # Scroll to the end of the conversation
    global_conversation_box.yview(tk.END)

# function to handle enter key press
def on_enter_key(event):
    """
    respond similar to pressing submit button
    """
    send_message()


# Function that will be triggered when the button is clicked or the Enter key is pressed
def on_send_message():
    print("Send clicked or Enter key pressed !")

def main():
    """
    main function of the chatbot
    initialize dataset, setup UI and invoke functions
    """

    # declare all the global variables (mainly UI fields)
    # required to be accessed across functions
    global global_input_field
    global global_conversation_box
    global global_window
    global global_submit_button

    # declare global QA and SQuAD dataframe required
    # to be acdessed across functions
    global global_qa_df
    global global_squad_df

    # Main logic of the program 
    print("Executing main function ...")
    global_qa_df, global_squad_df = Init()
    print("Init completed successfully")

    # Initialize the main window
    global_window = tk.Tk()
    global_window.title("Sci-Tech Yoda - GenAI Chatbot")


    # Change the background color using the 'configure' method
    global_window.configure(bg='lightblue')

    # Disable the maximize button by preventing resizing in both directions
    global_window.resizable(False, False)

    # Create a scrollable text area for the conversation
    global_conversation_box = scrolledtext.ScrolledText(global_window, wrap=tk.WORD, width=70, height=40, state=tk.DISABLED)
    global_conversation_box.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    # Get the default font for a widget (e.g., Label)
    default_font = tk.Label(global_window).cget("font")

    # Create a Label widget
    label = tk.Label(global_window, text="Input your query here", bg="lightblue", anchor="w")
    label.grid(row=1,column=0,columnspan=2)

     # Create a text input field
    global_input_field = tk.Entry(global_window, width=70)
    global_input_field.grid(row=2, column=0, padx=10)

    # Bind the Enter key to the input field to trigger 'on_enter_key' function
    global_input_field.bind('<Return>', on_enter_key)

    # Load the image (make sure the image file is in the same directory or provide a full path)
    btnImage = PhotoImage(file="./Btn_Arrow.png")

    # Resize the image using the subsample method (scale it down by a factor)
    resized_image = btnImage.subsample(4, 4)  # This scales down the image by 50% in both width and height

    # Create a button with an image
    image_button = tk.Button(global_window, image=resized_image, command=send_message)
    image_button.grid(row=2, column=1, padx=5, pady=10)

    # Bind the Enter key to the button click function
    global_window.bind('<Return>', lambda event: on_send_message())

    # Start the main event loop
    global_window.mainloop()

# The special if-block ensures this part only runs when the script is executed directly
if __name__ == "__main__":
    main()
