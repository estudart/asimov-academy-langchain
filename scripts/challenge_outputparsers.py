from typing import Optional
from pydantic import BaseModel, Field

from client_reviews import client_review_1
from extensions import chat



class Review(BaseModel):
    descricao: str = Field(description="Dê uma breve descrição em relação ao produto citado no texto")
    cliente_satisfeito_entrega: bool = Field(description="Diga se o cliente ficou satisfeito com a entrega no formato: true ou false")
    cliente_satisfeito_produto: bool = Field(description="Diga se o cliente ficou satisfeito com o produto no formato: true ou false")
    nota_atendimento: Optional[int] = Field(description="Retorne uma nota de 0 a 10 para o atendimento fornecido pela loja")
    nota_satisfacao: Optional[int] = Field(description="Retorne uma nota de 0 a 10 para a satisfação geral do cliente em relação a compra")


llm_estruturada = chat.with_structured_output(Review)
resposta = llm_estruturada.invoke(client_review_1)
print(resposta)