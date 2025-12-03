"""
DeepEval test suite for the FastAPI question-answering application
"""
import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    CoherenceMetric
)
import requests

# Base URL for the FastAPI application
BASE_URL = os.getenv("API_URL", "http://localhost:8000")

def test_question_answering_relevancy():
    """Test answer relevancy for question-answering"""
    
    # Test case 1
    context = "Hugging Face is a technology company that provides open-source NLP libraries and tools for machine learning practitioners."
    question = "What does Hugging Face provide?"
    
    # Make API call
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"question": question, "context": context}
    )
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get("answer", "")
        
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            context=context
        )
        
        # Test answer relevancy
        relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
        assert_test(test_case, [relevancy_metric])
        
        # Test faithfulness
        faithfulness_metric = FaithfulnessMetric(threshold=0.7)
        assert_test(test_case, [faithfulness_metric])
    else:
        print(f"API request failed with status {response.status_code}")
        assert False, f"API request failed: {response.text}"

def test_question_answering_faithfulness():
    """Test answer faithfulness to the provided context"""
    
    context = "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum."
    question = "Who created Python?"
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"question": question, "context": context}
    )
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get("answer", "")
        
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            context=context
        )
        
        faithfulness_metric = FaithfulnessMetric(threshold=0.7)
        assert_test(test_case, [faithfulness_metric])
    else:
        print(f"API request failed with status {response.status_code}")
        assert False, f"API request failed: {response.text}"

def test_prompt_evaluation_contextual_relevancy():
    """Test contextual relevancy of prompts and context"""
    
    context = "FastAPI is a modern, fast web framework for building APIs with Python. It is based on standard Python type hints and supports async operations."
    question = "What is FastAPI?"
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"question": question, "context": context}
    )
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get("answer", "")
        
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            context=context
        )
        
        # Test contextual relevancy
        contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7)
        assert_test(test_case, [contextual_relevancy_metric])
    else:
        print(f"API request failed with status {response.status_code}")
        assert False, f"API request failed: {response.text}"

def test_prompt_evaluation_contextual_precision():
    """Test contextual precision of the prompt"""
    
    context = "Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed. Deep learning is a subset of machine learning using neural networks."
    question = "What is machine learning?"
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"question": question, "context": context}
    )
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get("answer", "")
        
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            context=context
        )
        
        # Test contextual precision
        contextual_precision_metric = ContextualPrecisionMetric(threshold=0.7)
        assert_test(test_case, [contextual_precision_metric])
    else:
        print(f"API request failed with status {response.status_code}")
        assert False, f"API request failed: {response.text}"

def test_prompt_evaluation_contextual_recall():
    """Test contextual recall of relevant information"""
    
    context = "Docker is a platform for containerization. It allows developers to package applications with dependencies. Kubernetes is an orchestration tool for managing containers at scale."
    question = "What is Docker used for?"
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"question": question, "context": context}
    )
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get("answer", "")
        
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            context=context
        )
        
        # Test contextual recall
        contextual_recall_metric = ContextualRecallMetric(threshold=0.7)
        assert_test(test_case, [contextual_recall_metric])
    else:
        print(f"API request failed with status {response.status_code}")
        assert False, f"API request failed: {response.text}"

def test_prompt_evaluation_coherence():
    """Test coherence of the generated answer"""
    
    context = "Git is a distributed version control system. It tracks changes in source code during software development. GitHub is a platform that hosts Git repositories."
    question = "What is Git and how does it relate to GitHub?"
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"question": question, "context": context}
    )
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get("answer", "")
        
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            context=context
        )
        
        # Test coherence
        coherence_metric = CoherenceMetric(threshold=0.7)
        assert_test(test_case, [coherence_metric])
    else:
        print(f"API request failed with status {response.status_code}")
        assert False, f"API request failed: {response.text}"

def test_comprehensive_prompt_evaluation():
    """Comprehensive prompt evaluation with multiple metrics"""
    
    context = "REST API stands for Representational State Transfer Application Programming Interface. It uses HTTP methods like GET, POST, PUT, DELETE. RESTful APIs are stateless and follow REST principles."
    question = "What is a REST API?"
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"question": question, "context": context}
    )
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get("answer", "")
        
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            context=context
        )
        
        # Run comprehensive evaluation with multiple metrics
        metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
            ContextualRelevancyMetric(threshold=0.7),
            CoherenceMetric(threshold=0.7)
        ]
        
        assert_test(test_case, metrics)
    else:
        print(f"API request failed with status {response.status_code}")
        assert False, f"API request failed: {response.text}"

if __name__ == "__main__":
    test_question_answering_relevancy()
    test_question_answering_faithfulness()
    test_prompt_evaluation_contextual_relevancy()
    test_prompt_evaluation_contextual_precision()
    test_prompt_evaluation_contextual_recall()
    test_prompt_evaluation_coherence()
    test_comprehensive_prompt_evaluation()
    print("All DeepEval tests including prompt evaluation completed!")

