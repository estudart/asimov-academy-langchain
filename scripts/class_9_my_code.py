from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt_back_end = ChatPromptTemplate.from_template(
    """
    Voce é um professor de desenvolvimento back-end em um curso de programação com graduação no MIT. 
    Ao longo de sua carreira trabalhou em grandes empresas tech como Google e Amazon. Você é capaz de 
    tirar duvidas a respeito de desenvolvimento back-end de maneira detalhada e super didática. 
    Responda a seguinte pergunta do aluno:
    Pergunta {pergunta}
    """
)
chain_back_end = prompt_back_end | model

prompt_front_end = ChatPromptTemplate.from_template(
    """
    Voce é um professor de front end em um curso de programação com graduação em Harvard. Ao longo de 
    sua carreira teve a oportunidade de trabalhar com os frameworks mais conhecidos no mercado como: 
    React, Vue e Angular. Você é capaz de tirar duvidas a respeito de front-end de maneira 
    detalhada e super didática. Responda a seguinte pergunta do aluno:
    Pergunta {pergunta}
    """
)
chain_front_end = prompt_front_end | model

prompt_computacao_nuvem = ChatPromptTemplate.from_template(
    """
    Voce é um professor de computação em nuvem em um curso de programação com graduação em Yale. Ao longo de 
    sua carreira teve a oportunidade de trabalhar com os maiores provedores de serviços em numve como: 
    Amazon, Google Cloud e Azure. Você é capaz de tirar duvidas a respeito de computação em nuvem de maneira 
    detalhada e super didática. Responda a seguinte pergunta do aluno:
    Pergunta {pergunta}
    """
)
chain_computacao_nuvem = prompt_computacao_nuvem | model

promt_generico = ChatPromptTemplate.from_template(
    """
    {pergunta}
    """
)
chain_generico = promt_generico | model

prompt_estruturado = ChatPromptTemplate.from_template("Voce deve categorizar a seguinte pergunta: {pergunta}")

class Categorizador(BaseModel):
    """
    Categoriza as perguntas de alunos do curso de programação
    """
    area_conhecimento: str = Field(
        description="A área de conhecimento da pergunta feita pelo aluno. \
            Deve ser 'desenvolvimento front-end', 'computação em nuvem' ou 'desenvolvimento back-end'. \
            Caso não se encaixe em nenhuma delas, retorno 'outro'")
    
model_estruturado = prompt_estruturado | model.with_structured_output(Categorizador)
# response = model_estruturado.invoke({"pergunta": "Como criar uma api usando Python?"})
# print(response)

def route(input):
    if input['categoria'] == 'desenvolvimento front-end':
        return chain_front_end
    elif input['categoria'] == 'computação em nuvem':
        return chain_computacao_nuvem
    elif input['categoria'] == 'desenvolvimento back-end':
        return chain_back_end
    return chain_generico

chain = RunnablePassthrough().assign(categoria=model_estruturado) | route
response = chain.invoke({"pergunta": "Como criar uma api usando Python?"})
print(response.content)