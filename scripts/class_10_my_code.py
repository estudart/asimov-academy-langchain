from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

memory = InMemoryChatMessageHistory()

memory.add_user_message('Hello, model')
memory.add_ai_message('Hello, user')

# print(memory.messages)
agent_name = "Satoshi"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"Your are a crypto research assistent called {agent_name}. Answer de following questions on a pedagogical approach"),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ]
)
chain = prompt | ChatOpenAI()

store = {}
def get_by_session_id(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key='question',
    history_messages_key='history'
)

config = {
    'configurable': {'session_id': 'usuario_a'}
}

while True:
    question = input(f"Talk to {agent_name}: ")
    response = chain_with_memory.invoke({'question': question}, config=config)
    print(f"{agent_name}'s answer: {response.content} \n")