from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.chains.question_answering import load_qa_chain
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


chat = ChatOpenAI(model='gpt-3.5-turbo-0125')
chain = load_qa_chain(llm=chat, chain_type='stuff', verbose=True)

# Carregando pdfs
"""
path = "C:/Users/erico.studart_hashde/Desktop/erico_personal/courses/asimov-academy-langchain/scripts/arquivos/Explorando o Universo das IAs com Hugging Face.pdf"
loader = PyPDFLoader(path)
documents = loader.load()

# print(len(documents))
# print(documents[5].page_content)

chain = load_qa_chain(llm=chat, chain_type='stuff', verbose=True)
pergunta = 'Quais os assuntos tratados no documento?'
response = chain.run(input_documents=documents[:10], question=pergunta)
print(response)
"""

# Carregando csv
"""
path = "C:/Users/erico.studart_hashde/Desktop/erico_personal/courses/asimov-academy-langchain/scripts/arquivos/Top 1000 IMDB movies.csv"
loader = CSVLoader(path)
documents = loader.load()
chain = load_qa_chain(llm=chat, chain_type='stuff', verbose=True)
pergunta = 'Qual o filme com a maior duração?'
response = chain.run(input_documents=documents[:10], question=pergunta)
print(response)
"""

# Carregando Youtube
"""
url = 'https://www.youtube.com/watch?v=0I15MDx-7Ag'

save_dir='docs/youtube/'
loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir),
    OpenAIWhisperParser()
)
documents = loader.load()
pergunta = "Qual o sentiment geral desse video?"
response = chain.run(input_documents=documents[0], question=pergunta)
print(response)
"""

# Carregando da Web
url = 'https://cointelegraph.com/news/bitcoin-traders-turn-to-93k-yearly-open-btc-price-hits-6-week-high'
loader = WebBaseLoader(url)
documents = loader.load()
pergunta = "Qual são os principais pontos da noticia?"
response = chain.run(input_documents=documents[0:], question=pergunta)
print(response)
