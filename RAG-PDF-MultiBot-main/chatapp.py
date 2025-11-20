# 📚 Multi-PDF Chatbot + Multi-Bot Chat
# Author : Suraj Borkute   |   GitHub : https://github.com/Surajkecode
# -----------------------------------------------------------------------------
# 1️⃣  Imports
# -----------------------------------------------------------------------------
import os
import streamlit as st
from dotenv           import load_dotenv
from PyPDF2           import PdfReader
from langchain.text_splitter            import RecursiveCharacterTextSplitter
from langchain_community.vectorstores   import FAISS
from langchain_google_genai             import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts                  import PromptTemplate
import google.generativeai as genai

# -----------------------------------------------------------------------------
# 2️⃣  Environment / Config
# -----------------------------------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED_MODEL   = "models/embedding-001"
PDF_VECTOR_DIR = "faiss_index"      # local FAISS directory

# 🧠 Special-Bot catalogue
BOT_CATALOG = {
    "💻 Code Bot"      : "gemini-1.5-flash-latest",
    "🤖 General Bot"   : "gemini-1.5-flash-latest",
    "📚 Study Bot"     : "gemini-1.5-flash-8b-latest",
    "💝 Emotional Bot" : "gemini-1.5-flash",
}

# System prompts – tuned for richer answers
BOT_PROMPTS = {
    "💻 Code Bot": """
You are an expert programming assistant.
• Give production-ready code, fully commented.
• Provide step-by-step reasoning before the final answer when helpful.
• Use concise bullet points for explanations.
""",
    "🤖 General Bot": """
You are a knowledgeable, articulate assistant.
• Deliver clear, well-structured answers (~200 words max unless asked).
• When lists fit, format them as bullet points.
""",
    "📚 Study Bot": """
You are a patient academic tutor.
• Break concepts into small chunks with frequent examples.
• End each answer with a short 'Quick Recap'.
""",
    "💝 Emotional Bot": """
You are an empathetic companion.
• Respond with warmth and validation.
• Offer practical self-care suggestions when appropriate.
• Keep language supportive and non-judgemental.
""",
}

# -----------------------------------------------------------------------------
# 3️⃣  RAG (PDF-QA) — UNTOUCHED CORE LOGIC
# -----------------------------------------------------------------------------
def get_pdf_text(pdf_files) -> str:
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=50_000,
        chunk_overlap=1_000
    )
    return splitter.split_text(text)

def build_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vectordb   = FAISS.from_texts(chunks, embedding=embeddings)
    vectordb.save_local(PDF_VECTOR_DIR)

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(
        PDF_VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

def qa_chain():
    template = """
Answer as precisely as possible using ONLY the provided context.
If the answer is missing, say: "Answer is not available in the context."
Context:
{context}

Question: {question}
Answer:
"""
    prompt = PromptTemplate(template=template,
                            input_variables=["context", "question"])
    llm    = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash",
                                    temperature=0.3)
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def answer_pdf_question(user_q: str):
    vectordb = load_vector_store()
    docs     = vectordb.similarity_search(user_q, k=6)
    chain    = qa_chain()
    reply    = chain(
        {"input_documents": docs, "question": user_q},
        return_only_outputs=True
    )["output_text"]
    st.markdown(f"""
<div style="background:#1e1e1e;border-radius:10px;padding:15px;margin-top:10px;color:white;">
<strong>🤖 RAG Bot:</strong><br>{reply}
</div>""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4️⃣  Specialised-Bot chat helpers
# -----------------------------------------------------------------------------
def chat_special_bot(bot_name: str,
                     user_q: str,
                     history: list[str]) -> str:
    model_name = BOT_CATALOG[bot_name]
    system_prompt = BOT_PROMPTS[bot_name].strip()

    # Build rolling prompt (last 8 messages to keep context light)
    convo = "\n".join(history[-8:])
    full_prompt = f"{system_prompt}\n\nConversation so far:\n{convo}\nUser: {user_q}\nAssistant:"
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.6)
        resp = llm.invoke(full_prompt)
        return resp.content
    except Exception as err:
        return f"⚠️ Error: {err}"

def init_sessions():
    for bot in BOT_CATALOG:
        key = f"{bot}_history"
        if key not in st.session_state:
            st.session_state[key] = []

# -----------------------------------------------------------------------------
# 5️⃣  Streamlit App
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config("Multi-PDF Chatbot + Bots", page_icon="📚", layout="wide")
    init_sessions()

    # ---- Header --------------------------------------------------------------
    st.markdown("<h1 style='text-align:center;color:#33C3F0;'>📚 Multi-PDF Chatbot + 🧠 Special Bots</h1>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;'>Chat with your PDFs using RAG • or talk to specialised free Gemini bots</p>",
        unsafe_allow_html=True)
    st.write("---")

    # ---- Tabs ----------------------------------------------------------------
    tab_pdf, tab_bots = st.tabs(["📄 PDF Chat", "🧠 Bots"])

    # === TAB 1 : PDF ==========================================================
    with tab_pdf:
        st.markdown("### 💬 Ask your PDF")
        question = st.text_input(
            "Type a question about the uploaded PDFs:",
            key="pdf_q")
        if question:
            if os.path.exists(PDF_VECTOR_DIR):
                with st.spinner("Searching PDFs…"):
                    answer_pdf_question(question)
            else:
                st.warning("❗ Upload & process PDFs first in the sidebar.")

    # === TAB 2 : Bots =========================================================
    with tab_bots:
        st.markdown("### 🧠 Pick a bot")
        bot_choice = st.selectbox(
            "Specialised bot:",
            list(BOT_CATALOG.keys())
        )

        # Simple bot info without usage details
        st.info(f"**{bot_choice}** ready to help!", icon="🤖")

        # Chat input
        bot_q = st.text_input(f"Chat with {bot_choice}:", key=f"q_{bot_choice}")
        col_send, col_clear = st.columns([1, 4])

        with col_send:
            if st.button("🚀 Send", key=f"send_{bot_choice}") and bot_q:
                hist_key = f"{bot_choice}_history"
                with st.spinner("Thinking…"):
                    reply = chat_special_bot(
                        bot_choice, bot_q, st.session_state[hist_key])

                st.session_state[hist_key].append(f"User: {bot_q}")
                st.session_state[hist_key].append(f"Bot: {reply}")

                # Trim history
                if len(st.session_state[hist_key]) > 24:
                    st.session_state[hist_key] = st.session_state[hist_key][-24:]

        with col_clear:
            if st.button("🗑️ Clear Chat", key=f"clr_{bot_choice}"):
                st.session_state[f"{bot_choice}_history"] = []
                st.rerun()  # ✅ FIXED: Changed from st.experimental_rerun()

        # Conversation history rendering
        hist_key = f"{bot_choice}_history"
        if st.session_state[hist_key]:
            st.markdown("### 💬 Conversation")
            for entry in st.session_state[hist_key][-10:]:
                if entry.startswith("User:"):
                    st.success(entry[5:].strip())
                else:
                    st.markdown(f"""
<div style="background:#1e1e1e;color:white;border-radius:8px;padding:10px;margin:6px 0;">
{entry[4:].strip()}
</div>""", unsafe_allow_html=True)

    # ---- Sidebar -------------------------------------------------------------
    with st.sidebar:
        st.image("img/Robot.jpg", use_container_width=True)
        st.markdown("---")

        st.markdown("### 📁 Upload PDFs")
        pdfs = st.file_uploader(
            "Select PDF files", accept_multiple_files=True, type=["pdf"])

        if st.button("🚀 Submit & Process"):
            if pdfs:
                with st.spinner("Indexing…"):
                    raw  = get_pdf_text(pdfs)
                    chunks = split_into_chunks(raw)
                    build_vector_store(chunks)
                st.success("✅ PDFs indexed!")
            else:
                st.warning("No PDFs selected.")

        st.markdown("---")
        st.markdown("### 🧠 Bots Available")
        for bot in BOT_CATALOG.keys():
            st.markdown(f"- **{bot}**")

        st.markdown("---")
        st.image("img/gkj.jpg", use_container_width=True)
        st.markdown("<p style='text-align:center;font-size:14px;'>Created by <b>@Suraj Borkute</b></p>",
                    unsafe_allow_html=True)

    # ---- Footer --------------------------------------------------------------
    st.markdown("""
<style>
.footer{
    position:fixed;left:0;bottom:0;width:100%;
    background:#0E1117;color:white;text-align:center;
    padding:12px;font-size:14px;z-index:100;
}
.footer a{color:#33C3F0;text-decoration:none;}
</style>
<div class="footer">
🚀 Built with LangChain + Gemini | <a href="https://github.com/Surajkecode" target="_blank">GitHub</a>
</div>""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6️⃣  Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
