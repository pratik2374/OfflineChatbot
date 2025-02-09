import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage
import time


## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful massistant . Please  repsonse to the user queries"),
        ("user","Question:{question}")
    ]
)

if "store" not in st.session_state:
    st.session_state["store"] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]

def generate_response(question,llm):
    model=Ollama(model=llm)
    output_parser=StrOutputParser()

    prompt=ChatPromptTemplate.from_messages(
       [
           ("system","You are a helpful assistant.Amnswer all the question to the nest of your ability"),
            MessagesPlaceholder(variable_name="messages")
       ]
    )

    chain=prompt|model|output_parser

    #chain.invoke({"messages":[HumanMessage(content="Hi My name is Pratik")]})

    with_message_history=RunnableWithMessageHistory(chain,get_session_history)

    config = {"configurable": {"session_id": "chat3"}}
    response=with_message_history.invoke(
        [HumanMessage(content=question)],
        config=config
    )

    return response

def response_generator(prompt,llm):
    result=generate_response(prompt,llm)
    for word in result.split():
        yield word + " "
        time.sleep(0.05)


st.title("Enhanced Q&A Chatbot With OpenSource Llama Model")


llm=st.sidebar.selectbox("Select Open Source model",["gemma2:2b","llama3.2:1b"])

temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Goe ahead and ask any question")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
#user_input=st.text_input("You:")
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response=st.write_stream(response_generator(prompt,llm))
    st.session_state.messages.append({"role": "assistant", "content": response})


# if user_input :
#     response=generate_response(user_input,llm,temperature,max_tokens)
#     st.write(response)
# else:
#     st.write("Please provide the user input")


