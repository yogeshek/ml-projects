"""
LLM- Based spam classifier using ollama"""

import ollama
from typing import Dict, Optional
import os

# set ollama models path
os.environ['OLLAMA_MODELS'] = r'D:\Ollama-models'

class OllamaSpamClassifier:
    """spam classifier using local Ollama LLM"""
    def __init__(self, model: str ="llama3.2"):
        """
        Initialize ollama classifier
        
        Arg:
        model: ollama models name (defulat: llama3.2 from the instalation)
        """
        self.model = model
        self.name = f"ollama_{model}"
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Classify text as spam or ham using LLM
        
        Args:
            text: SMS message to classifiy
        Returns:
            Dict with prediction, confidence and reasoning
        """
        
        prompt = f""" Classify this SMS message as either 'spam' or 'ham' (legitimate message).
        
Message: "{text}"

Respond in this exact format:
Classification: [spam or ham]
Confidence: [0-100]
Reason: [brief explanation]"""

        try:
            response = ollama.chat(
                model=self.model,
                message = [{
                    'role':'user',
                    'content': prompt
                }],
                options={
                    'temprature':0.1,
                    'num_predict':100
                }
            )
            
            response_text = response['message']['content']
            #parse response
            result = self._parse_response(response_text, text)
            return result
        
        except Exception as e:
            print(f"[ERROR] ollama prediction failed: {e}")
            return{
                'prediciton':'ham',
                'confidence': 50.0,
                'reasoning': f'Error: {str(e)}',
                'raw_response': None
            }
    def _parse_response(self, response_text: str, original_text:str) -> dict:
        """ Parse LLM response to extract prediction"""
        response_lower = response_text.lower()
        
        if 'classification: spam' in response_lower or 'spam' in response_lower.split('\n')[0]:
            prediction = 'spam'
        elif 'classification: ham' in response_lower or 'ham' in response_lower:
            prediction = 'ham'
        else:
            # Fallback: look for spam/ham anywhere in response
            prediction = 'spam' if 'spam' in response_lower else 'ham'
        # Extract confidence (look for number between 0-100)
        confidence = 75.0 #Default
        import re
        conf_match = re.search(r'confidence:\s*(\d+)', response_lower)
        if conf_match:
            confidence = float(conf_match.group(1))
            
        # Extract reasoning
        reasoning = "No reasoning provided"
        if 'reason:' in response_lower:
            reasoning = response_text.split('reason', 1)[1].strip()[:200]
            
            return{
                'prediction': prediction,
                'confidence': confidence,
                'reasoning': reasoning,
                'raw_response': response_text
            }
            
            
        
        
        
        
        
    



