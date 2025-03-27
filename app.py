import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMChain
import wikipedia
import re
import math

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def wikipedia_search(query: str) -> str:
    """Search Wikipedia for information."""
    try:
        return wikipedia.summary(query, sentences=2)
    except:
        return "Could not find information on Wikipedia."

def math_calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        # Remove any non-math characters except basic operators and numbers
        expression = re.sub(r'[^0-9+\-*/(). ]', '', expression)
        result = eval(expression)
        return f"The result is: {result}"
    except:
        return "Invalid mathematical expression."

def reasoning_tool(query: str) -> str:
    """Use LLM for reasoning and analysis."""
    return "This is a reasoning tool response."

# Create tools
tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia_search,
        description="Useful for searching general knowledge and facts on Wikipedia. Input should be a search query."
    ),
    Tool(
        name="Math Calculator",
        func=math_calculator,
        description="Useful for calculating mathematical expressions. Input should be a mathematical expression."
    ),
    Tool(
        name="Reasoning Tool",
        func=reasoning_tool,
        description="Useful for logical reasoning and analysis. Input should be a question or statement requiring reasoning."
    )
]

# Streamlit UI
st.set_page_config(page_title="AI Assistant with Groq", layout="wide")

# Sidebar for API key
with st.sidebar:
    st.title("Settings")
    groq_api_key = st.text_input("Enter your Groq API Key:", type="password")
    if groq_api_key:
        st.success("API Key configured!")
    else:
        st.warning("Please enter your Groq API Key to proceed.")

# Main content
st.title("🤖 AI Assistant with Groq")
st.write("Ask me anything! I can help with calculations, search Wikipedia, and provide reasoning.")

# Initialize chat model and agent
if groq_api_key:
    try:
        # Initialize the LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="Gemma2-9b-It",
            temperature=0.7,
            max_tokens=1000
        )

        # Create a simple prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. You can help with calculations, search Wikipedia, and provide reasoning. Always provide clear and concise responses."),
            ("human", "{input}")
        ])

        # Create the chain
        chain = LLMChain(llm=llm, prompt=prompt)

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt_text := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt_text})
            with st.chat_message("user"):
                st.markdown(prompt_text)

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Create a container for the thought process
                        thought_container = st.container()
                        
                        # Create a callback handler for Streamlit with expanded thoughts
                        callback_handler = StreamlitCallbackHandler(
                            thought_container,
                            expand_new_thoughts=True,
                            collapse_completed_thoughts=True
                        )

                        # First, try to determine which tool to use
                        if any(char in prompt_text for char in ['+', '-', '*', '/', '(', ')']):
                            with thought_container:
                                st.write("🤔 Using Math Calculator...")
                            response = math_calculator(prompt_text)
                        elif "wikipedia" in prompt_text.lower():
                            with thought_container:
                                st.write("🔍 Searching Wikipedia...")
                            query = prompt_text.replace("wikipedia", "").strip()
                            response = wikipedia_search(query)
                        else:
                            # Use the LLM for general responses
                            with thought_container:
                                st.write("💭 Processing your question...")
                            response = chain.run(
                                input=prompt_text,
                                callbacks=[callback_handler]
                            )
                        
                        # Display response
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.error("Please try again with a different question.")
                        st.exception(e)

    except Exception as e:
        st.error(f"Error initializing the agent: {str(e)}")
        st.exception(e)

else:
    st.warning("Please enter your Groq API Key in the sidebar to start using the assistant.") 