from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from text_in_english import cointelegraph_text

load_dotenv()

"""
1) Create a chain to translate any text to portuguese
"""

# Step 1: Define a model
model = ChatOpenAI()
# Step 2: Generate a good prompt with the commands to the model
translate_prompt = ChatPromptTemplate.from_template("Traduza o texto a seguir para a lingua portuguesa: {text}")
# Step 3: Define the output you wanna get
output = StrOutputParser()
# Step 4: Create a chain with the previous steps
translate_chain = translate_prompt | model | output
# Step 5: invoke the chain with the variable
# translate_response = translate_chain.invoke({"text": "Hi, my name is Goku"})

# print(translate_response)


"""
Create a chain to summarize the text
"""

summarize_prompt = ChatPromptTemplate.from_template("Resuma o texto a seguir: {text}")
summarize_chain = summarize_prompt | model | output
# summarize_response = summarize_chain.invoke({"text": translate_response})

# print(summarize_response)

"""
A chain the is a combination of chain 1 and 2
"""

combination_chain = translate_chain | summarize_chain
combination_response = combination_chain.invoke({"text": cointelegraph_text})

print(combination_response)