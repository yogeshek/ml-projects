"""
LLM-based spam classifier using Ollama
"""
import ollama
from typing import Dict, Optional
import os

# Set Ollama models path to your custom directory
os.environ['OLLAMA_MODELS'] = r'D:\Ollama-models'

class OllamaSpamClassifier:
    """Spam classifier using local Ollama LLM"""

    def __init__(self, model: str = "llama3.2:1b"):
        """
        Initialize Ollama classifier

        Args:
            model: Ollama model name (default: llama3.2:1b from your installation)
        """
        self.model = model
        self.name = f"ollama_{model.replace(':', '_')}"

    def predict(self, text: str) -> Dict[str, any]:
        """
        Classify text as spam or ham using LLM

        Args:
            text: SMS message to classify

        Returns:
            Dict with prediction, confidence, and reasoning
        """
        prompt = f"""Classify this SMS message as either 'spam' or 'ham' (legitimate message).

Message: "{text}"

Respond in this exact format:
Classification: [spam or ham]
Confidence: [0-100]
Reason: [brief explanation]"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.1,  # Low temperature for consistent classification
                    'num_predict': 100   # Limit response length
                }
            )

            response_text = response['message']['content']

            # Parse response
            result = self._parse_response(response_text, text)
            return result

        except Exception as e:
            print(f"[ERROR] Ollama prediction failed: {e}")
            return {
                'prediction': 'ham',  # Default to ham on error
                'confidence': 50.0,
                'reasoning': f'Error: {str(e)}',
                'raw_response': None
            }

    def _parse_response(self, response_text: str, original_text: str) -> Dict:
        """Parse LLM response to extract prediction and confidence"""
        response_lower = response_text.lower()

        # Extract classification
        lines = response_lower.split('\n')
        first_line = lines[0] if lines else ""

        if 'classification: spam' in response_lower or 'spam' in first_line:
            prediction = 'spam'
        elif 'classification: ham' in response_lower or 'ham' in response_lower:
            prediction = 'ham'
        else:
            # Fallback: look for spam/ham anywhere in response
            prediction = 'spam' if 'spam' in response_lower else 'ham'

        # Extract confidence (look for number between 0-100)
        confidence = 75.0  # Default
        import re
        conf_match = re.search(r'confidence:\s*(\d+)', response_lower)
        if conf_match:
            confidence = float(conf_match.group(1))

        # Extract reasoning
        reasoning = "No reasoning provided"
        if 'reason:' in response_lower:
            try:
                parts = response_text.split('reason:', 1)
                if len(parts) > 1:
                    reasoning = parts[1].strip()[:200]
            except Exception:
                reasoning = "Could not extract reasoning"

        return {
            'prediction': prediction,
            'confidence': confidence,
            'reasoning': reasoning,
            'raw_response': response_text
        }

# Quick test function
def test_classifier():
    """Test the Ollama classifier"""
    classifier = OllamaSpamClassifier()

    test_messages = [
        "WINNER! You've won $1000! Click here now!!!",
        "Hey mom, I'll be home late tonight",
        "FREE iPhone! Limited time offer! Act now!",
        "Meeting is at 3pm in conference room B"
    ]

    print(f"Testing Ollama Spam Classifier ({classifier.model})")
    print("=" * 70)

    for msg in test_messages:
        result = classifier.predict(msg)
        print(f"\nMessage: {msg[:50]}...")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Reasoning: {result['reasoning'][:100]}")
        print("-" * 70)

if __name__ == "__main__":
    test_classifier()
