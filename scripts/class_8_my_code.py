from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv



load_dotenv()
"""
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("Create a sentence about the subject: {subject}")
chain = prompt | model
subject = "Jiu-jitsu"
response = chain.invoke(subject)

print(response)

for stream in chain.stream(subject):
    print(stream.content, end='')

response = chain.batch(
    [
        {"subject": "jiu-jitsu"},
        {"subject": "boxing"},
        {"subject": "mma"}
    ]
)

print(response)
"""

model = ChatOpenAI()
history_prompt = ChatPromptTemplate.from_template("Tell the history behind the following city: {city}")
history_chain = history_prompt | model | StrOutputParser()

activities_prompt = ChatPromptTemplate.from_template("Tell the best activities to do in the given city: {city}")
activities_chain = activities_prompt | model | StrOutputParser()

combined_prompt = ChatPromptTemplate.from_template(
    """
    Given the history of the city and the best activities to do in it,
    generate a nice touristic post for my travels website

    History of the city: {city_history}
    Best activities on the city: {city_best_activities}
    """
)

parralel = RunnableParallel(
    {
        "city_history": history_chain,
        "city_best_activities": activities_chain
    }
)

# response = parralel.invoke({"city": "Copenhagen"})
# print(response)

combined_chain = parralel | combined_prompt | model | StrOutputParser()
response = combined_chain.invoke({"city": "Rio de Janeiro"})
print(response)
