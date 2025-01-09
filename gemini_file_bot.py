import streamlit as st
import google.generativeai as genai
import os
import json
import io
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
from dotenv import load_dotenv
import time
from tabulate import tabulate
import logging
import google.api_core.exceptions # Import the specific exception

load_dotenv()  # Load environment variables from .env file

# Configure API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini Model
model = genai.GenerativeModel('gemini-pro')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Predefined responses
PREDEFINED_RESPONSES = {
    "hello": "Hello! How can I help you with your overseas education journey today?",
    "how are you": "I am doing great. How are you today?",
    "goodbye": "Goodbye! Feel free to come back if you have any more questions",
    "country": "Sure, what country are you interested in studying in?",
    "program": "Sure, what program are you looking into?",
     "clear": "Alright, let's start over!",
}
def read_pdf(file_content):
    pdf_text = ""
    try:
        pdf_reader = PdfReader(BytesIO(file_content))
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
    return pdf_text

def read_docx(file_content):
    docx_text = ""
    try:
        document = Document(BytesIO(file_content))
        for paragraph in document.paragraphs:
           docx_text += paragraph.text + "\n"
    except Exception as e:
        logging.error(f"Error reading DOCX: {e}")
    return docx_text

def read_txt(file_content):
    try:
        return file_content.decode("utf-8")
    except Exception as e:
        logging.error(f"Error reading TXT: {e}")
        return ""

def read_excel(file_content):
    try:
        df = pd.read_excel(BytesIO(file_content))
        excel_text = df.to_string()
        return excel_text
    except Exception as e:
        logging.error(f"Error reading Excel: {e}")
        return ""

def extract_tables(text):
    tables = []
    try:
        # Split the text into lines
        lines = text.splitlines()
        # Attempt to identify table start and end based on common patterns
        table_start_indices = [i for i, line in enumerate(lines) if any(x in line for x in ["Sl No", "University Name", "Country","City/State","Commission to Associate","Country University Name Commission to Associate"]) ]

        if table_start_indices:
           for start_index in table_start_indices:
                table_end_index = next((i for i in range(start_index + 1, len(lines)) if lines[i].strip() == ""), len(lines))
                table_lines = lines[start_index:table_end_index]
                table_data = [line.strip() for line in table_lines]
                tables.append(tabulate([line.split("  ") for line in table_data],headers="firstrow", tablefmt="grid") )
    except Exception as e:
         logging.error(f"Error extracting tables: {e}")
    return "\n".join(tables)


def process_documents_gemini(uploaded_files):
    combined_content = ""
    combined_tables = ""
    for uploaded_file in uploaded_files:
       try:
          file_content = uploaded_file.read()
          if uploaded_file.type == "application/pdf":
            content = read_pdf(file_content)
          elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" :
             content = read_docx(file_content)
          elif uploaded_file.type == "text/plain" :
            content = read_txt(file_content)
          elif  uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
             content = read_excel(file_content)
          else:
             content = " "
          combined_tables += extract_tables(content)
          combined_content += content
       except Exception as e:
           logging.error(f"Error processing {uploaded_file.name}: {e}")
    return combined_content, combined_tables

def generate_gemini_response(user_input, chat_history, combined_content, combined_tables):
   enriched_prompt = f"""
     You are an expert in overseas education. Your role is to guide students by understanding their academic background, educational goals, and all the information in all the documents that they provide and recommending universities or programs that match their profile.
     Here is the transcript of previous conversation:
     {chat_history}
    Here is the information from all the documents provided by the user (if any) and some of them may be in the form of tables: {combined_content}
    Here are the extracted tables from the provided documents:{combined_tables}
     Based on this information from all the documents, please:
    1. **Provide Descriptive Responses**: Respond clearly and thoroughly to the user's queries, making sure you take all the documents into consideration and providing answers based on that.
    2. **Answer Specific Questions**: Pay attention to the details from the tables from all the documents and respond appropriately to the user's question, if a question is related to the information in the table please provide the information in a well formatted way.
    3. **Ask Specific Follow-up Questions**: If needed, ask relevant questions to gather further details such as preferred study level (Bachelor's, Master's, PhD), specific program interests, budget, or location preferences.
    4. **Provide Recommendations**: If the user has provided enough information in the past messages and/or documents recommend universities or programs if possible
    The current query from the user is: {user_input}
    """
   max_retries = 3
   retry_delay = 1  # seconds
   for attempt in range(max_retries):
        try:
            if GOOGLE_API_KEY:
                response = model.generate_content(enriched_prompt, stream=True)
                full_response = ""
                with st.chat_message("assistant"):
                   message_placeholder = st.empty()
                   for chunk in response:
                      full_response += chunk.text
                      message_placeholder.markdown(full_response + "▌")
                   message_placeholder.markdown(full_response)
                return full_response
            else:
                st.error("Google API key is missing. Please check your `.env` file and make sure `GOOGLE_API_KEY` is present.")
                return "I'm unable to process your request, as I do not have access to Google API."
        except google.api_core.exceptions.ResourceExhausted as e:
            logging.warning(f"Resource Exhausted error: {e}. Retrying in {retry_delay} seconds (Attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
            if attempt == max_retries - 1:
                  logging.error(f"Gemini API failed after max retries: {e}")
                  st.error(f"Gemini API encountered an error, please try again later")
                  return "I'm facing issues, please try again."
        except Exception as e:
           logging.error(f"An unexpected error occurred: {e}")
           st.error(f"An unexpected error occurred, please try again.")
           return "I'm facing issues, please try again."


def generate_response(user_message):
   user_message = user_message.lower()
   for key, response in PREDEFINED_RESPONSES.items():
        if key in user_message:
             if key == "clear":
                  if "chat_history" in st.session_state:
                      st.session_state.chat_history = []
                      st.session_state.chat_history.append({"role": "bot", "message": "Alright, let's start over!"})
                      st.rerun()
             return response
   return "I'm here to assist with your overseas education journey. Please provide more details or ask a specific question."


# Streamlit App
def main():
      # Custom CSS for styling
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #1a2a6c, #b21f1f);
            color: #fff;
            padding: 20px;
           }
        .stApp {
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
                backdrop-filter: blur(5px);
            }
        .chat-container {
              width: 100%;
            margin-bottom: 20px;
                border-radius: 15px;
                 overflow: hidden;
              display: flex;
             flex-direction: column;
                background: rgba(255, 255, 255, 0.05);
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
             }
        .chat-log {
               flex-grow: 1;
                padding: 20px;
                overflow-y: auto;
               display: flex;
               flex-direction: column;
                gap: 15px;
          }
        .user-message, .bot-message {
             padding: 15px 20px;
            border-radius: 20px;
            display: inline-block;
              max-width: 75%;
             word-wrap: break-word;
             margin-bottom: 15px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
            }
        .user-message {
             background: linear-gradient(145deg, #4a148c, #2196f3);
             color: #fff;
             align-self: flex-end;
            }
        .bot-message {
             background-color: #333;
            color: #fff;
             align-self: flex-start;
          }
        .stTextInput > div > div > input {
             border-radius: 10px;
           border: 1px solid #fff;
             padding: 12px;
             margin-bottom: 15px;
            transition: border-color 0.3s ease;
             background: rgba(255, 255, 255, 0.1);
            color: #fff;
           }
        .stTextInput > div > div > input:focus {
            border-color: #2196f3;
          }
        .stButton {
             text-align: right;
            margin-bottom: 15px;
            }
       .stButton > button{
              background: linear-gradient(45deg, #2196f3, #66bb6a);
              color: white;
            border: none;
             border-radius: 10px;
             cursor: pointer;
               padding: 12px 20px;
             transition: background 0.4s ease, transform 0.2s;
           }
        .stButton > button:hover {
              background: linear-gradient(45deg, #2196f3, #66bb6a);
              transform: translateY(-2px);
            }
        .stFileUploader>div {
              border: 2px dashed rgba(255, 255, 255, 0.3);
                border-radius: 10px;
                padding: 20px;
                 margin-bottom: 20px;
              background: rgba(255, 255, 255, 0.1);
            }
        .stFileUploader>div>label {
            text-align: center;
           display: block;
             font-size: 1.2em;
            font-weight: bold;
              color: #fff
            }
       h1 {
                color: #fff;
            text-align: center;
                margin-bottom: 25px;
           }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("EDUBOT: Your Overseas Education Guide")

    # Chat history initialization
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "bot", "message": "Hi there! I'm EDUBOT, your guide to overseas education. Let's start exploring your options."})

    # Display chat history within a chat container
    with st.container():
        st.markdown('<div class="chat-container"><div class="chat-log" id="chat-log"></div></div>',unsafe_allow_html = True)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["message"]}</div>', unsafe_allow_html = True)
            elif message["role"] == "bot":
                st.markdown(f'<div class="bot-message">{message["message"]}</div>',unsafe_allow_html = True)

    # File uploader
    uploaded_files = st.file_uploader("Upload your documents (Transcripts, Test scores, etc.)", accept_multiple_files=True, type = ["pdf", "docx", "txt", "xlsx"])

    # User Input
    user_message = st.chat_input("Type your message...")
    if user_message:
         st.session_state.chat_history.append({"role": "user", "message": user_message})
         combined_content, combined_tables = process_documents_gemini(uploaded_files) if uploaded_files else ("","")
         bot_response = generate_gemini_response(
            user_input = user_message,
            chat_history = "\n".join([f"{msg['role']}: {msg['message']}" for msg in st.session_state.chat_history]),
            combined_content = combined_content,
            combined_tables = combined_tables,
         )
         st.session_state.chat_history.append({"role": "bot", "message": bot_response})


def generate_response(user_message):
   user_message = user_message.lower()
   for key, response in PREDEFINED_RESPONSES.items():
        if key in user_message:
             if key == "clear":
                  if "chat_history" in st.session_state:
                      st.session_state.chat_history = []
                      st.session_state.chat_history.append({"role": "bot", "message": "Alright, let's start over!"})
                      st.rerun()
             return response
   return "I'm here to assist with your overseas education journey. Please provide more details or ask a specific question."

if __name__ == "__main__":
    main()