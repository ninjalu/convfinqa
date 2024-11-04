import json
from utils import split_questions
from rag import FinRAG
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    SemanticSimilarity,
    LLMContextPrecisionWithReference,
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from ragas import EvaluationDataset
from ragas import SingleTurnSample
import pickle
import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


def data_to_samples(eval_data, qa_llm, seed=42):
    eval_samples = []
    for i, item in enumerate(eval_data):
        finrag = FinRAG(llm_model=qa_llm, context=item, seed=seed)
        try:
            result = finrag.qa(item["qa"]["question"])
        except Exception as e:
            print(e, f"Skipping data {i}")
            continue
        item["rag_answer"] = result["answer"]
        item["rag_reference"] = result["contexts"]
        try:
            eval_samples.append(
                SingleTurnSample(
                    user_input=item["qa"]["question"],
                    retrieved_contexts=item["rag_reference"],
                    reference_contexts=list(item["qa"]["gold_inds"].values()),
                    response=str(item["rag_answer"]),
                    reference=item["qa"]["answer"],
                )
            )
        except Exception as e:
            print(
                f"response is {str(item['rag_answer'])}, type: {type(item['rag_answer'])}"
            )
            print(
                f"reference is {item['qa']['answer']}, type: {type(item['qa']['answer'])}"
            )
            print(item)
            print(result)
            print(e, f"Skipping data {i}")

    with open(f"eval_samples_{qa_llm}.pkl", "wb") as f:
        pickle.dump(eval_samples, f)
    return eval_samples


def eval(eval_samples: list, eval_llm: str, qa_llm: str = None):
    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(model=eval_llm, temperature=0, seed=42)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBEDDING_MODEL)
    )
    metrics = [
        LLMContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        SemanticSimilarity(llm=evaluator_embeddings),
    ]
    eval_dataset = EvaluationDataset(samples=eval_samples)
    eval_results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    results_df = eval_results.to_pandas()
    if qa_llm:
        with open(f"eval_results_{qa_llm}.pkl", "wb") as f:
            pickle.dump(results_df, f)
    return results_df
