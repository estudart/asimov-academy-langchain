from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

assunto = "Dinamarca"

prompt_curiosidade = ChatPromptTemplate.from_template("Fale uma curiosidade sobre o assunto: {assunto}")
chain_curiosidade = prompt_curiosidade | ChatOpenAI() | StrOutputParser()

prompt_historia = ChatPromptTemplate.from_template("Crie uma história sobre o seguinte fato curioso: {assunto}")
chain_historia = prompt_historia | ChatOpenAI() | StrOutputParser()

chain_combinada = chain_curiosidade | chain_historia

response = chain_combinada.invoke({"assunto": assunto})
print(response)