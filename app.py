import os 
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import certifi  # Import certifi

#---------------------------------------------------
#for steamlit deployment only 
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] #Comment both key when runing on vs code 
groq_api_key = st.secrets["GROQ_API_KEY"] #Comment both key when runing on vs code 
#---------------------------------------------------

# Set SSL certificate file path using certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper 
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
from langchain.agents import AgentType, initialize_agent #define type, Creates and sets up an AI agent with tools
from langchain.callbacks import StreamlitCallbackHandler   #Streamlit's callback handler for real-time interaction tracking

# map your envirnment api key with the api key present in dotenv file 
openai_api_key=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

#---------------#  Tools Used in App  #------------------------#


## Arxiv and wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wikipedia=WikipediaQueryRun(api_wrapper=api_wrapper)

# initialise the duckducksearch
search = DuckDuckGoSearchRun(name="Search")



#---------------#  Setting up the Streamlit app  #---------------#

# App title
st.title("🔎 AI-Powered Web & Research Assistant")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

#side bar name settings
st.sidebar.title("Settings")

# input the Groq api key 
api_key=st.sidebar.text_input("Enter your Groq API key:", type="password",value=groq_api_key ) #display on app

# Show a warning if the API key is missing
if not api_key:
    st.sidebar.warning("⚠️ Groq API key is not set. Please insert Groq API key") #display on app


#Check if the "messages" key exists in session state
if "messages" not in st.session_state:
    
    #If msg do not exist, initialize a list with a default chatbot message 
    st.session_state["messages"] = [
    {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]


#Loop through all messages in session state and display them
#chat_message- built-in function in Streamlit used to display chat messages

for msg in st.session_state.messages:
    # 'role' shows if its from the assistant or the user
    st.chat_message(msg["role"]).write(msg["content"])

#Wait for user input in the chat box
if prompt:= st.chat_input(placeholder="What is Machine Learning?"):
    
    #Save the user's message to session state (so we can track chat history)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    
    st.chat_message("user").write(prompt) #Display the user's message in the chat UI

    #call the llm model 
    #streaming-output will be received gradually instead of waiting for the entire response
    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True,temperature=0.5)

    
    tools=[search,arxiv,wikipedia] #get all the tools create above

    ## Create an AGENT
    
    # ZERO_SHOT_REACT_DESCRIPTION - Defines the agent's reasoning strategy think step-by-step before acting
    # handling_parsing_errors- Enables better error handling for parsing mistakes
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)
    
    #openai parsing is better.Use when required 
    #search_agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS,handle_parsing_errors=True)


    # Get the latest user message
    #user_query = st.session_state.messages[-1]["content"]

    
    with st.chat_message("assistant"): #when role is assistant'create chat message container 

        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False) #Set up a callback handler to track the agent's thought process in Streamlit
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb]) #Run the agent using stored chat messages and return a response
        st.session_state.messages.append({"role":"assistant","content":response}) #append the response to sessionstate
        st.write(response)






