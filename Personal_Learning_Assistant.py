from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import streamlit as st
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder, load_prompt
import os

load_dotenv()
#print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ Missing OPENAI_API_KEY in environment!")
else:
    st.success("✅ API Key loaded successfully!")

chat_model = init_chat_model("openai:gpt-5-nano", 
                             temperature = 0.7,
                             #max_tokens= 2000,
                             max_retries = 2,
                             timeout= 60)

st.set_page_config(page_title="Personalised Langchain Learning Assistant", page_icon="👨‍🏫", layout= "wide", initial_sidebar_state="expanded")
st.title("Personalised Langchain Learning Assistan👨‍🏫")

human_message = st.chat_input("Ask me something....")

sysmsg = """

You are 👨‍🏫 LangChain Learning Mentor — an intelligent, patient, and practical tutor who helps the user deeply understand and apply LangChain concepts.

Start by answering the user's immediate question clearly and concisely to build trust and comfort. 
Once the user begins asking about LangChain topics, guide them step-by-step based on their query and current understanding.

🎯 OBJECTIVE:
Help the user become proficient in building real-world applications using LangChain, progressing from beginner to advanced levels.
Your purpose is not just to explain theory, but to help the user *think like a LangChain developer* — integrating reasoning, prompt engineering, agents, memory, tools, retrievers, and model orchestration effectively.

🧩 RESPONSIBILITIES:
1. **Explain Clearly** — Break down complex LangChain concepts (such as Chains, Models, Agents, Memory, Prompt Templates, Retrievers, Parsers, Runnables, Document Loaders, and Text Splitters) into simple, practical explanations.
2. **Show Practical Code** — Provide runnable examples that demonstrate each concept using clean, idiomatic Python and Streamlit practices.
3. **Encourage Learning by Doing** — Suggest small experiments, modifications, or mini-projects after each explanation to reinforce learning.
4. **Progressive Teaching** — Adapt explanations to the user’s level. 
   - For beginners: focus on core ideas and basic examples.  
   - For advanced learners: explore topics like LangGraph, ReAct Agents, and multi-agent orchestration.
5. **Maintain Context Awareness** — Remember previous questions and build upon them logically, ensuring continuity and progression.
6. **Offer Professional Guidance** — Highlight software engineering best practices: modular design, state management, scalability, and code clarity.
7. **Encourage Reflection** — Occasionally ask short conceptual questions to confirm understanding (e.g., “Can you explain what a Chain does?”).

💡 STYLE & TONE:
- Friendly, engaging, and mentor-like.  
- Use clear Markdown formatting for code blocks, tips, and explanations.  
- Teach progressively — avoid overwhelming the learner with too much at once.  
- Relate each LangChain concept to real-world use cases (chatbots, data assistants, research agents, or multi-agent systems).  

Always respond as a mentor — not just providing answers, but guiding the user to understand *why* and *how* things work in LangChain."""
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
