from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore, InMemoryByteStore
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langsmith import traceable
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
import uuid
from dotenv import load_dotenv
import os
import getpass

from utils import (
    get_qa_template,
    get_table_str_template,
    prep_table,
    extract_json_from_string,
)
from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Union
import textwrap

load_dotenv()


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


os.environ["OLLAMA_API_KEY"] = os.getenv("OLLAMA_API_KEY")
os.environ["OLLAMA_URL"] = os.getenv("OLLAMA_URL")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
_set_env("LANGCHAIN_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


class Resp(BaseModel):
    answer: Union[str, float, int] = Field(description="the answer to the question")
    contexts: List[str] = Field(description="list of facts from the context")


class FinRAG:
    def __init__(self, llm_model: str, context: dict, seed: int = 42) -> None:
        self.llm = ChatOllama(model=llm_model, temperature=0, seed=seed)
        self.embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.prompt = ChatPromptTemplate.from_template(get_qa_template())
        self.context = context
        self.retriever = self.get_retriever()

    @traceable
    def get_retriever(self):
        table_summ = self.elements_to_summ_chain(self.context["table"], "table")
        all_docs = self.context["pre_text"] + [table_summ] + self.context["post_text"]
        vstore = Chroma(collection_name="refs", embedding_function=self.embedder)
        store = InMemoryStore()
        byte_store = InMemoryByteStore()
        id_key = "id"
        retriever = MultiVectorRetriever(
            vectorstore=vstore, store=store, id_key=id_key, byte_store=byte_store
        )
        doc_ids = [str(uuid.uuid4()) for _ in all_docs]
        summary_texts = [
            Document(page_content=text, metadata={id_key: doc_ids[i]})
            for i, text in enumerate(all_docs)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, summary_texts)))
        return retriever

    @traceable
    def qa(self, question: str) -> str:
        parser = PydanticOutputParser(pydantic_object=Resp)
        qa_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        result = qa_chain.invoke(question)
        try:
            json_str = extract_json_from_string(result)
            json_output = parser.invoke(json_str).dict()
        except Exception as e:
            print(f"Result: {result} is not able to be parsed: {e}")
            json_output = {"answer": result, "contexts": ["Error"]}
        return json_output

    # @traceable
    def elements_to_summ_chain(self, element: list, type: str) -> str:
        summ_promt = ChatPromptTemplate.from_template(get_table_str_template())
        summ_chain = (
            {"element": lambda x: x} | summ_promt | self.llm | StrOutputParser()
        )
        if type == "table":
            element = prep_table(element)
            result = summ_chain.invoke(element)
        else:
            result = summ_chain.batch(element)
        return result


class RagEvaluator:
    def __init__(self, dataset_name, model) -> None:
        self.llm = ChatOllama(model=model, temperature=0)
        self.dataset_name = dataset_name
        self.qa_evaluator = LangChainStringEvaluator(
            "cot_qa",
            config={"llm": self.llm},
            prepare_data=lambda run, example: {
                "prediction": run.outputs["prediction"],
                "reference": example.outputs["reference"],
                "input": example.inputs["question"],
            },
        )

        self.hallucination_evaluator = LangChainStringEvaluator(
            "labeled_score_string",
            config={
                "criteria": {
                    "accuracy": """Is the Assistant's Answer grounded in the Ground Truth documentation? A score of [[1]] means that the
                        Assistant answer contains is not at all based upon / grounded in the Groun Truth documentation. A score of [[5]] means 
                        that the Assistant answer contains some information (e.g., a hallucination) that is not captured in the Ground Truth 
                        documentation. A score of [[10]] means that the Assistant answer is fully based upon the in the Ground Truth documentation."""
                },
                "llm": self.llm,
                # If you want the score to be saved on a scale from 0 to 1
                "normalize_by": 10,
            },
            prepare_data=lambda run, example: {
                "prediction": run.outputs["prediction"],
                "reference": run.outputs["reference"],
                "input": example.inputs["question"],
            },
        )
        self.doc_relevence_evaluator = LangChainStringEvaluator(
            "score_string",
            config={
                "criteria": {
                    "document_relevance": textwrap.dedent(
                        """The response is a set of documents retrieved from a vectorstore. The input is a question
                        used for retrieval. You will score whether the Assistant's response (retrieved docs) is relevant to the Ground Truth 
                        question. A score of [[1]] means that none of the  Assistant's response documents contain information useful in answering or addressing the user's input.
                        A score of [[5]] means that the Assistant answer contains some relevant documents that can at least partially answer the user's question or input. 
                        A score of [[10]] means that the user input can be fully answered using the content in the first retrieved doc(s)."""
                    )
                },
                "llm": self.llm,
                # If you want the score to be saved on a scale from 0 to 1
                "normalize_by": 10,
            },
            prepare_data=lambda run, example: {
                "prediction": run.outputs["reference"],
                "input": example.inputs["question"],
            },
        )

    def evaluate(self, rag: FinRAG, data: str) -> float:
        qa_evaluation = evaluate(
            rag.qa,
            data=data,
            evaluators=[
                self.qa_evaluator,
                self.hallucination_evaluator,
                self.doc_relevence_evaluator,
            ],
        )
        return qa_evaluation
