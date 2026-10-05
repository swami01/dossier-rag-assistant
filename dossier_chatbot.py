"""Run: streamlit run dossier_chatbot.py"""
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from pipeline import GREETINGS, NO_ANSWER, OFFICES, admin, build_prompt, office_user, out_of_scope, retrieve

load_dotenv()                                   # reads GOOGLE_API_KEY from .env
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.title("Dossier Assistant")

# DEMO login picker. In production the backend sets this from the JWT, never the browser.
login = st.sidebar.selectbox("Login as", ["admin"] + OFFICES)
user = admin if login == "admin" else office_user(login)

if "messages" not in st.session_state:
    st.session_state.messages = []
for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

    
def standalone(question, history):
    """Rewrite a follow-up ('give me their names') into a self-contained question using recent chat."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-15:])
    prompt = ("Rewrite the user's last question as one self-contained question, resolving words like "
            "'their', 'them', 'those', 'it' using the conversation. Keep every office, place, nationality "
            "and number mentioned. If it is already self-contained, repeat it unchanged. "
            "Output only the question.\n\n"
            f"Conversation:\n{convo}\n\nLast question: {question}\nSelf-contained question:")
    return llm.invoke(prompt).text.strip() or question

question = st.chat_input("Ask about dossiers")
if question:
    history = list(st.session_state.messages)             # earlier turns only
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)
    
    g = question.lower().strip(" !?.,")
    if g in GREETINGS:                                    # canned reply, no search, no LLM call
        first = g.title() if g.startswith("good") else g.split(",")[0].title()
        answer = f"{first}! How can I help you with dossiers today?"
    else:
        question = standalone(question, history)
        st.caption(f"Searching for: {question}")
        docs, metas = retrieve(question, user, "rerank")  # access control + search happen here
        if out_of_scope(question, user):
            answer = f"Your login ({login}) can only see its own office's dossiers."
        elif docs:
            answer = llm.invoke(build_prompt(question, docs)).text
            answer += "\n\n**Sources:** " + ", ".join(m["dossier_no"] for m in metas)
        else:
            answer = NO_ANSWER                             # no context -> never call the LLM

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)
