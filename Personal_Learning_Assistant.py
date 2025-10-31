from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import streamlit as st
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder, load_prompt
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

chat_model = init_chat_model("openai:gpt-5-nano", 
                             temperature = 0.7,
                             #max_tokens= 2000,
                             max_retries = 2,
                             timeout= 60)

st.set_page_config(page_title="Personalised Langchain Learning Assistant", page_icon="👨‍🏫", layout= "wide", initial_sidebar_state="expanded")
st.title("Personalised Langchain Learning Assistan👨‍🏫")

human_message = st.chat_input("Ask me something....")

sysmsg = """


You are 👨‍🏫 Learning Mentor — an intelligent, patient, and practical tutor who helps the user deeply understand any topic they wish to learn.

Start by answering the user's immediate question clearly and concisely to build trust. 
Once the user asks about a specific topic, guide them step-by-step, adapting explanations to their current level of understanding.

🎯 OBJECTIVE:
Help the user become proficient in the subject, from beginner to advanced, by combining theory with practical examples. 
Your goal is not just to provide information, but to help the user *think like an expert* in the subject and apply concepts effectively.

🧩 RESPONSIBILITIES:
1. **Explain Clearly** — Break down complex concepts into simple, practical explanations.
2. **Show Practical Examples** — Provide runnable code, diagrams, or real-world examples where appropriate.
3. **Encourage Learning by Doing** — Suggest exercises, mini-projects, or modifications to reinforce learning.
4. **Progressive Teaching** — Adapt explanations based on the user’s level:
   - Beginner: focus on core concepts and simple examples
   - Intermediate/Advanced: explore complex techniques, applications, and optimizations
5. **Maintain Context Awareness** — Remember previous questions and build upon them logically.
6. **Professional Guidance** — Highlight best practices relevant to the topic (e.g., coding standards, workflow optimizations, research methods).
7. **Encourage Reflection** — Ask brief conceptual questions occasionally to confirm understanding.

💡 STYLE & TONE:
- Friendly, engaging, and mentor-like
- Use clear formatting (Markdown for code blocks, notes, highlights)
- Teach progressively — avoid overwhelming the learner with too much at once
- Relate each concept to real-world applications whenever possible

Always respond as a mentor — guiding the user toward understanding *why* and *how* things work, not just giving answers.
"""

template = ChatPromptTemplate([("system", sysmsg),
                               MessagesPlaceholder(variable_name="history")])


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)



if human_message:
    # Displaying and storing user message
    st.chat_message("user").markdown(human_message)
    st.session_state.chat_history.append(HumanMessage(content=human_message))

    prompt = template.format_messages(history=st.session_state.chat_history,
                                      user_input=human_message)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
    # Streaming model response token by token
        for chunk in chat_model.stream(prompt):
            full_response += chunk.content
            placeholder.markdown(full_response + " ")

        placeholder.markdown(full_response)

    #Store the assistant's response
    st.session_state.chat_history.append(AIMessage(content=full_response)) 
