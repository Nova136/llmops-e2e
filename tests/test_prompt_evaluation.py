"""
Dedicated prompt evaluation test suite using DeepEval metrics
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

# Test cases for prompt evaluation
TEST_CASES = [
    {
        "name": "Technology Question",
        "context": "Kubernetes is an open-source container orchestration platform. It automates deployment, scaling, and management of containerized applications.",
        "question": "What is Kubernetes used for?",
        "expected_keywords": ["orchestration", "container", "deployment", "scaling"]
    },
    {
        "name": "Definition Question",
        "context": "Microservices architecture is an approach where applications are built as a collection of small, independent services that communicate over well-defined APIs.",
        "question": "What is microservices architecture?",
        "expected_keywords": ["small", "independent", "services", "APIs"]
    },
    {
        "name": "Comparison Question",
        "context": "SQL databases are relational and use structured schemas. NoSQL databases are non-relational and offer flexible schemas. Both have their use cases.",
        "question": "What is the difference between SQL and NoSQL databases?",
        "expected_keywords": ["relational", "non-relational", "schemas"]
    }
]

def evaluate_prompt_with_metrics(context, question, answer, metrics):
    """Helper function to evaluate a prompt with multiple metrics"""
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        context=context
    )
    
    try:
        assert_test(test_case, metrics)
        return True
    except AssertionError as e:
        print(f"Evaluation failed: {e}")
        return False

def test_prompt_evaluation_suite():
    """Run comprehensive prompt evaluation on multiple test cases"""
    results = []
    
    for test_case in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['name']}")
        print(f"Question: {test_case['question']}")
        print(f"{'='*60}")
        
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "question": test_case["question"],
                "context": test_case["context"]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "")
            print(f"Answer: {answer}")
            
            # Define metrics for prompt evaluation
            metrics = [
                AnswerRelevancyMetric(threshold=0.7),
                FaithfulnessMetric(threshold=0.7),
                ContextualRelevancyMetric(threshold=0.7),
                ContextualPrecisionMetric(threshold=0.7),
                ContextualRecallMetric(threshold=0.7),
                CoherenceMetric(threshold=0.7)
            ]
            
            # Evaluate with all metrics
            success = evaluate_prompt_with_metrics(
                test_case["context"],
                test_case["question"],
                answer,
                metrics
            )
            
            results.append({
                "test_name": test_case["name"],
                "success": success,
                "answer": answer
            })
        else:
            print(f"API request failed with status {response.status_code}")
            results.append({
                "test_name": test_case["name"],
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("Prompt Evaluation Summary")
    print(f"{'='*60}")
    passed = sum(1 for r in results if r.get("success", False))
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for result in results:
        status = "✓" if result.get("success", False) else "✗"
        print(f"{status} {result['test_name']}")
    
    return results

def test_single_prompt_evaluation():
    """Test a single prompt with all evaluation metrics"""
    context = "Artificial Intelligence (AI) is the simulation of human intelligence by machines. Machine Learning is a subset of AI that enables systems to learn from data."
    question = "What is the relationship between AI and Machine Learning?"
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"question": question, "context": context},
        timeout=30
    )
    
    assert response.status_code == 200, f"API request failed: {response.status_code}"
    
    result = response.json()
    answer = result.get("answer", "")
    
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        context=context
    )
    
    # Comprehensive prompt evaluation
    all_metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
        CoherenceMetric(threshold=0.7)
    ]
    
    assert_test(test_case, all_metrics)
    print(f"✓ All prompt evaluation metrics passed for: {question}")

if __name__ == "__main__":
    test_single_prompt_evaluation()
    test_prompt_evaluation_suite()
    print("\nAll prompt evaluation tests completed!")

