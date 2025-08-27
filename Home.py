import os
from dotenv import load_dotenv
import streamlit as st
import google.api_core.exceptions as google_exceptions
import create_map  # your module to get documents
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
google_api_key = st.secrets["GOOGLE_API_KEY"]

# Initialize LLM model
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Rohit's Portfolio Assistant",
    page_icon="🤖",
    layout="wide"
)

# --------------------------
# Custom Styling
# --------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #2c3e50;
        }
        .subtitle {
            font-size: 18px;
            color: #34495e;
            margin-bottom: 20px;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
            margin-top: 10px;
        }
        .stButton>button {
            background-color: #4b7bec;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 8px 16px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #3867d6;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Prompt Template
# --------------------------
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

# --------------------------
# App Title
# --------------------------
st.markdown("<div class='title'>👋 Hello There!</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Welcome to Rohit's Portfolio Assistant — Ask me anything about Rohit's career, projects, and skills. </div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Since I am built using free tools, I have usage limitations. If I am not able to answer your question, please check the About Me section to get all the information about Rohit. </div>", unsafe_allow_html=True)

st.markdown("""
💡 **Sample Questions:**
- Give me a summary of the projects done by Rohit  
- Tell me about Rohit's work experience  
- Tell me about Rohit's education background  
- Tell me about Rohit's skills  
""")

# --------------------------
# Session State
# --------------------------
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = ""
if "result_docs" not in st.session_state:
    st.session_state.result_docs = []
if "output" not in st.session_state:
    st.session_state.output = ""

# --------------------------
# Functions
# --------------------------
def generate_response():
    if not st.session_state.result_docs or not st.session_state.user_prompt.strip():
        st.session_state.output = ""
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc in st.session_state.result_docs])
    with st.spinner("🤔 Thinking..."):
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
    generate_response()

# --------------------------
# Quick Buttons
# --------------------------
st.markdown("### 🚀 Quick Topics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📂 Projects"):
        fetch_documents_and_respond("projects", "Give me a summary of Rohit's projects")

with col2:
    if st.button("🛠 Skills"):
        fetch_documents_and_respond("skills", "Give me info about Rohit's skills in a categorized format")

with col3:
    if st.button("💼 Work Experience"):
        fetch_documents_and_respond("work", "Give me a summary of Rohit's work experience")

with col4:
    if st.button("🎓 Education"):
        fetch_documents_and_respond("education", "Give me a summary of Rohit's education")

# --------------------------
# User Input
# --------------------------
st.markdown("---")
user_input = st.text_input("💬 Type your question here:")

if st.button("Ask"):
    if user_input.strip():
        lower_input = user_input.lower()
        if "project" in lower_input:
            topic = "projects"
        elif "work" in lower_input:
            topic = "work"
        elif "education" in lower_input:
            topic = "education"
        elif "skill" in lower_input:
            topic = "skills"
        else:
            topic = ""
        fetch_documents_and_respond(topic, user_input)
    else:
        st.warning("⚠️ Please enter a question or click a quick topic button.")

# --------------------------
# Output Section
# --------------------------
st.markdown("### 📜 Response")
if st.session_state.output:
    st.markdown(f"<div class='card'>{st.session_state.output}</div>", unsafe_allow_html=True)
else:
    st.info("No response yet. Try asking a question above or use a quick topic.")
