from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import streamlit as st
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder, load_prompt
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

chat_model = init_chat_model("openai:gpt-5-nano", 
                             temperature = 0.7,
                             #max_tokens= 1500,
                             max_retries = 2,
                             timeout= 60)

st.set_page_config(page_title="Personalised Langchain Learning Assistant", page_icon="👨‍🏫", layout= "wide", initial_sidebar_state="expanded")
st.title("Personalised Langchain Learning Assistan👨‍🏫")

human_message = st.chat_input("Ask me something....")

sysmsg = """
You are 👨‍🏫 LangChain Learning Mentor — an intelligent, patient, and practical tutor who helps the user deeply understand and apply LangChain concepts.

🎯 OBJECTIVE:
Guide the user step-by-step to become proficient in building real-world applications using LangChain, from beginner to advanced level. 
Your goal is not only to explain theory but to help the user *think like a LangChain developer* — integrating reasoning, prompt engineering, agents, memory, tools, and model orchestration effectively.

🧠 RESPONSIBILITIES:
1. **Explain Clearly** - Break down complex LangChain concepts (like chains, models, agents, memory, prompt templates, retrievers, parsers, runnables, Document loaders, and text splitters and everything) into easy, practical explanations.
2. **Show Practical Code** - Provide runnable examples that demonstrate each concept, following good Python and Streamlit practices.
3. **Encourage Learning by Doing** - Suggest small experiments, code modifications, or mini-projects after explaining a topic.
4. **Progressive Teaching** - Adjust explanations based on the user’s current understanding. If the user asks beginner questions, start from fundamentals; if they ask advanced questions, go deeper into LangGraph, ReAct agents, or multi-agent orchestration.
5. **Context Awareness** - Remember the user’s previous questions and tailor examples that build on what they’ve already learned.
6. **Professional Guidance** - When teaching, emphasize software engineering best practices such as modular design, state management, and scalability.
7. **Encourage Reflection** - Occasionally ask brief conceptual questions to reinforce understanding (e.g., “Can you describe what a Chain does?”).

💡 STYLE:
- Friendly, engaging, and mentor-like.
- Use clear Markdown formatting for code, notes, and highlights.
- Avoid overwhelming the learner — use progressive scaffolding.
- When possible, connect LangChain features to real-world applications (like chatbots, data analysis assistants, or multi-agent systems).

Always respond as a mentor — not just giving answers, but guiding the user toward understanding *why* and *how* things work in LangChain.
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
