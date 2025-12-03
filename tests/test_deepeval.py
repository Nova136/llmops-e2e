"""
DeepEval test suite for the FastAPI question-answering application
"""
import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric
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
            retrieval_context=[context]
        )
        
        # Test answer relevancy
        relevancy_metric = AnswerRelevancyMetric(threshold=0.4)
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
            retrieval_context=[context]
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
            retrieval_context=[context]
        )
        
        # Test contextual relevancy
        contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7)
        assert_test(test_case, [contextual_relevancy_metric])
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
            retrieval_context=[context]
        )
        
        # Run comprehensive evaluation with multiple metrics
        metrics = [
            AnswerRelevancyMetric(threshold=0.4),
            FaithfulnessMetric(threshold=0.7),
            ContextualRelevancyMetric(threshold=0.7)
        ]
        
        assert_test(test_case, metrics)
    else:
        print(f"API request failed with status {response.status_code}")
        assert False, f"API request failed: {response.text}"

if __name__ == "__main__":
    test_question_answering_relevancy()
    test_question_answering_faithfulness()
    test_prompt_evaluation_contextual_relevancy()
    test_comprehensive_prompt_evaluation()
    print("All DeepEval tests including prompt evaluation completed!")

