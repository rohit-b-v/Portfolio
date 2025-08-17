import os
from dotenv import load_dotenv
import streamlit as st
import google.api_core.exceptions as google_exceptions

import create_map # your module to get documents
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the model
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)

# Define prompt template
prompt_template = PromptTemplate(
    template="""
You are Rohit's personal agent and are responsible for providing information on work experience, projects, education, yourself and any other synonym questions. You have to provide the information in a maximum of 4 bullet points for each heading by summarizing the given context. If you are unsure about any question you can just answer that you don't know that information, don't make up new information.

Format the information you want to display with appropriate headings and sub headings. Usually the line with all capitals are headings 

Use the context given below for your questions: {context}
---
Answer this question based on the information given above: {question}
""",
    input_variables=["question", "context"],
)

# --- Streamlit UI ---

st.title("Hello There 👋")
st.subheader("Welcome to Rohit's Portfolio")

st.write(
    """
I am an LLM here to answer your questions about Rohit. Since I am built using a free version of Google Gemini, I have usage limitations. 
If I am not able to answer your question, please check the "About Me" section to get all the information about Rohit.
"""
)

st.markdown(
    """
**Try asking or clicking on the buttons below:**  
- Give me a summary of the projects done by Rohit  
- Tell me about Rohit's work experience  
- Tell me about Rohit's education background  
- Tell me about Rohit's skills  
"""
)

# Initialize session state variables
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""
if "result_docs" not in st.session_state:
    st.session_state.result_docs = []
if "output" not in st.session_state:
    st.session_state.output = ""

def generate_response():
    if not st.session_state.result_docs or not st.session_state.user_prompt.strip():
        st.session_state.output = ""
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc in st.session_state.result_docs])
    print("context", context_text)
    with st.spinner("Getting that info..."):
        try:
            response = model.invoke(
                prompt_template.format(context=context_text, question=st.session_state.user_prompt)
            )
            st.session_state.output = response.content

        except google_exceptions.ResourceExhausted:
            st.session_state.output = "⚠️ Sorry, max limit reached. Please check the About Me page for more information."

        except google_exceptions.ServiceUnavailable:
            st.session_state.output = "🚧 Gemini service is temporarily unavailable. Please try again later."

        except google_exceptions.DeadlineExceeded:
            st.session_state.output = "⏳ The request timed out. Try again in a moment."

        except google_exceptions.GoogleAPICallError as e:
            st.session_state.output = f"API call error: {e}"

        except Exception as e:
            st.session_state.output = f"Unexpected error: {e}"
    st.session_state.result_docs = []
    st.session_state.user_prompt = ""
    

def fetch_documents_and_respond(topic: str, question: str):
    if topic == "":
        st.session_state.result_docs = create_map.get_all_document()
        st.session_state.user_prompt = question + " see if this question matches with information provided else tell you don't know about it"
    else:
        st.session_state.result_docs = create_map.get_documents(topic)
        st.session_state.user_prompt = question
    print("generated prompt", st.session_state.user_prompt)
    generate_response()

# --- Quick buttons section ---
st.markdown("### Quick Topics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Projects", key="btn_projects"):
        fetch_documents_and_respond("projects", "Give me a summary of Rohit's projects")

with col2:
    if st.button("Skills", key="btn_skills"):
        fetch_documents_and_respond("skills", "Give me info about Rohit's skills in a categorized format")

with col3:
    if st.button("Work Experience", key="btn_work_exp"):
        fetch_documents_and_respond("work", "Give me a summary of Rohit's work experience")

with col4:
    if st.button("Education", key="btn_education"):
        fetch_documents_and_respond("education", "Give me a summary of Rohit's education")

user_input = st.text_input("Type your question here:", key="user_input")

if st.button("Ask", key="btn_ask"):
    if user_input.strip():
        lower_input = user_input.lower()
        if "project" in lower_input:
            topic = "projects"
        elif "work" in lower_input:
            topic = "work"
        elif "education" in lower_input:
            topic = "education"
        elif lower_input.__contains__("skill"):
            topic = "skills"
        else:
            topic=""
        
        fetch_documents_and_respond(topic, user_input)
    else:
        st.warning("Please enter a question or click one of the buttons above.")

# --- Single output container ---

if st.session_state.output:
    st.write(st.session_state.output)
else:
    st.write("No response yet. Click a button or ask a question above.")
