import requests
import json
import logging
import re
from typing import Dict, Optional
import time

logger = logging.getLogger(__name__)

class DeepSeekAnalyzer:
    """
    Local DeepSeek integration for advanced sentiment and content analysis
    Uses natural language understanding instead of forced JSON parsing
    """
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.get("deepseek.enabled", True)
        self.base_url = config.get("deepseek.base_url", "http://localhost:11434")
        self.model = config.get("deepseek.model", "deepseek-chat")
        self.timeout = config.get("deepseek.timeout", 10)
        self.max_retries = config.get("deepseek.max_retries", 2)
        
        if self.enabled:
            self._test_connection()
    
    def _test_connection(self):
        """Test connection to local DeepSeek server"""
        try:
            # Test Ollama connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                if any(self.model in name for name in model_names):
                    logger.info(f"✅ DeepSeek model '{self.model}' found and ready")
                    print(f"🤖 DeepSeek R1 8B connected successfully!")
                    print(f"🔗 Server: {self.base_url}")
                    
                    # Test with a simple query
                    self._test_simple_query()
                else:
                    logger.warning(f"⚠️  DeepSeek model '{self.model}' not found. Available models: {model_names}")
                    logger.warning(f"Run: ollama pull {self.model}")
                    print(f"❌ DeepSeek model '{self.model}' not found!")
                    print(f"📋 Available models: {model_names}")
                    print(f"💡 Run: ollama pull {self.model}")
                    self.enabled = False
            else:
                logger.error(f"❌ Failed to connect to Ollama server at {self.base_url}")
                print(f"❌ Cannot connect to Ollama server at {self.base_url}")
                print(f"💡 Make sure Ollama is running: ollama serve")
                self.enabled = False
                
        except Exception as e:
            logger.error(f"❌ DeepSeek connection test failed: {e}")
            print(f"❌ DeepSeek connection failed: {e}")
            print(f"💡 Make sure Ollama is running: ollama serve")
            self.enabled = False
    
    def _test_simple_query(self):
        """Test DeepSeek with a simple query"""
        try:
            test_result = self.analyze_content("This is a test message")
            if test_result and test_result.get("source") == "deepseek":
                print(f"✅ DeepSeek test successful: {test_result.get('sentiment', 'unknown')} sentiment")
            else:
                print(f"⚠️  DeepSeek test failed, falling back to keywords")
        except Exception as e:
            print(f"⚠️  DeepSeek test error: {e}")
            logger.warning(f"DeepSeek test failed: {e}")
    
    def analyze_content(self, transcript: str) -> Dict:
        """
        Analyze transcript with DeepSeek using natural language understanding
        
        Args:
            transcript: The speech transcript to analyze
            
        Returns:
            Dict with analysis results
        """
        if not self.enabled or not transcript or len(transcript.strip()) < 3:
            return self._fallback_analysis(transcript)
        
        try:
            # Create natural language prompt
            prompt = self._create_analysis_prompt(transcript)
            
            # Call DeepSeek
            response = self._call_deepseek(prompt)
            
            if response:
                return self._parse_deepseek_response(response, transcript)
            else:
                return self._fallback_analysis(transcript)
                
        except Exception as e:
            logger.error(f"DeepSeek analysis failed: {e}")
            return self._fallback_analysis(transcript)
    
    # def _create_analysis_prompt(self, transcript: str) -> str:
    #     """Create natural language prompt for DeepSeek R1"""
    #     prompt = f"""You are an expert in audio content analysis. Analyze this speech transcript:

	# 	TRANSCRIPT: "{transcript}"

	# 	You are an experienced conversation-analysis assistant.  
	# 	Your job is to scan each utterance in a speech-to-text transcript and judge whether it contains **negative language** (criticism, hostility, threats, profanity, sarcasm, self-deprecation, etc.).  
	# 	Because some snippets may be out of context, rely only on the words you see; do not invent extra context.  
	# 	For every utterance produce:

	# 	1. "sentiment": one of {{negative, neutral/unclear, positive}}.  
	# 	2. "negativity_subtype" (if sentiment == negative): choose the most relevant label from  
	# 	   ─ complaint / dissatisfaction  
	# 	   ─ insult / ad-hominem  
	# 	   ─ threat / intimidation  
	# 	   ─ profanity / vulgarity  
	# 	   ─ self-loathing / depressive  
	# 	   ─ other  
	# 	3. "confidence": 0–100 (your confidence in the sentiment judgment).  
	# 	4. "explanation": one or two short sentences quoting the words that triggered your decision."""
    #     return prompt

    def _create_analysis_prompt(self, transcript: str) -> str:
    	
        """
        Build a prompt that **forces** DeepSeek to answer with JSON only.
        """
        return (
            "<|system|>\n"
            "You are an experienced conversation-analysis assistant. "
            "Return ONLY valid JSON — no markdown, no commentary.\n"
            "Schema:\n"
            "[\n"
            "  {\n"
            "    \"utterance\": \"...\",\n"
            "    \"sentiment\": \"negative | neutral/unclear | positive\",\n"
            "    \"negativity_subtype\": \"complaint / insult / threat / profanity / self-loathing / other | null\",\n"
            "    \"confidence\": 0-100,\n"
            "    \"explanation\": \"...\"\n"
            "  }\n"
            "]\n"
            "\n"
            "<|user|>\n"
            "<<< TRANSCRIPT >>>\n"
            f"{transcript}\n"
            "<<< END TRANSCRIPT >>>"
        )
    
    def _call_deepseek(self, prompt: str) -> Optional[str]:
        """Call local DeepSeek API"""
        for attempt in range(self.max_retries):
            try:
                # Ollama API format
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Slightly higher for more natural responses
                        "top_p": 0.9,
                        "num_predict": 800,  # More tokens for detailed thinking
                        "stop": []  # Let it think naturally
                    }
                }
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    deepseek_response = ( result.get("response")
					    or result.get("message", {}).get("content", "")
					    or "" )

                    logger.debug(f"DeepSeek raw response: {deepseek_response[:200]}...")
                    return deepseek_response
                else:
                    logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                logger.error(f"DeepSeek API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Brief delay before retry
        
        return None
    
    def _parse_deepseek_response(self, response: str, transcript: str) -> Dict:
        """Parse DeepSeek natural language response"""
        try:
            # Check if response is empty or None
            if not response or response.strip() == "":
                logger.warning("DeepSeek returned empty response")
                return self._fallback_analysis(transcript)
            
            # Log the response for debugging
            logger.info(f"🧠 DeepSeek thinking response: '{response[:300]}...' (truncated)")
            
            # Extract the thinking content if present
            thinking_content = response
            if "<think>" in response:
                think_start = response.find("<think>")
                think_end = response.find("</think>")
                if think_start != -1 and think_end != -1:
                    thinking_content = response[think_start + 7:think_end]
                    logger.info(f"🧠 Extracted thinking: {thinking_content[:200]}...")
                
                # Also include any content after </think>
                if think_end != -1:
                    after_think = response[think_end + 8:].strip()
                    if after_think:
                        thinking_content += "\n" + after_think
            
            # Extract insights from the natural language response
            analysis = self._extract_insights_from_thinking(thinking_content, transcript)
            
            # Add metadata
            analysis["source"] = "deepseek"
            analysis["model"] = self.model
            analysis["original_transcript"] = transcript
            analysis["raw_thinking"] = thinking_content[:500]  # Store first 500 chars for debugging
            
            logger.info(f"✅ DeepSeek analysis successful: {analysis['sentiment']} ({analysis['confidence']:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing DeepSeek response: {e}")
            logger.error(f"Raw response was: '{response[:200]}...'")
            return self._fallback_analysis(transcript)
    
    def _extract_insights_from_thinking(self, thinking_content: str, transcript: str) -> Dict:
        """Extract insights from DeepSeek's natural language thinking"""
        # Convert to lowercase for easier matching
        content_lower = thinking_content.lower()
        
        # Extract sentiment
        sentiment = self._extract_sentiment(content_lower)
        
        # Extract emotional tone
        emotional_tone = self._extract_emotional_tone(content_lower)
        
        # Extract conversation topic
        conversation_topic = self._extract_conversation_topic(thinking_content, transcript)
        
        # Extract communication style
        communication_style = self._extract_communication_style(content_lower)
        
        # Extract concerns
        concerns = self._extract_concerns(content_lower)
        
        # Extract confidence level
        confidence = self._extract_confidence(content_lower)
        
        # Extract key themes
        key_themes = self._extract_key_themes(content_lower)
        
        # Create summary from thinking content
        summary = self._create_summary(thinking_content, sentiment, emotional_tone)
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "emotional_tone": emotional_tone,
            "mental_state": self._extract_mental_state(content_lower),
            "conversation_topic": conversation_topic,
            "content_analysis": thinking_content[:200] + "..." if len(thinking_content) > 200 else thinking_content,
            "toxicity": self._extract_toxicity(content_lower),
            "profanity": self._extract_profanity(content_lower),
            "emergency": self._extract_emergency(content_lower),
            "threat": self._extract_threat(content_lower),
            "distress_level": self._extract_distress_level(content_lower),
            "communication_style": communication_style,
            "key_themes": key_themes,
            "concerns": concerns,
            "summary": summary
        }
    
    def _extract_sentiment(self, content_lower: str) -> str:
        """Extract sentiment from thinking content"""
        positive_indicators = ["positive", "happy", "good", "pleased", "cheerful", "upbeat", "optimistic", "joyful"]
        negative_indicators = ["negative", "sad", "frustrated", "angry", "upset", "disappointed", "worried", "stressed"]
        neutral_indicators = ["neutral", "balanced", "calm", "matter-of-fact", "objective"]
        mixed_indicators = ["mixed", "conflicted", "ambivalent", "both positive and negative"]
        
        positive_count = sum(1 for word in positive_indicators if word in content_lower)
        negative_count = sum(1 for word in negative_indicators if word in content_lower)
        neutral_count = sum(1 for word in neutral_indicators if word in content_lower)
        mixed_count = sum(1 for word in mixed_indicators if word in content_lower)
        
        if mixed_count > 0:
            return "mixed"
        elif negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        elif neutral_count > 0:
            return "neutral"
        else:
            return "neutral"
    
    def _extract_emotional_tone(self, content_lower: str) -> str:
        """Extract emotional tone from thinking content"""
        tones = [
            "calm", "excited", "frustrated", "cheerful", "worried", "confident", 
            "anxious", "relaxed", "energetic", "tired", "enthusiastic", "bored",
            "angry", "happy", "sad", "neutral", "dismissive", "casual"
        ]
        
        for tone in tones:
            if tone in content_lower:
                return tone
        
        return "neutral"
    
    def _extract_conversation_topic(self, thinking_content: str, transcript: str) -> str:
        """Extract conversation topic from thinking content"""
        # Look for topic indicators in the thinking
        topic_patterns = [
            r"about (.+?)(?:\.|,|;|\n)",
            r"discussing (.+?)(?:\.|,|;|\n)",
            r"topic.*?is (.+?)(?:\.|,|;|\n)",
            r"conversation.*?about (.+?)(?:\.|,|;|\n)",
            r"talking about (.+?)(?:\.|,|;|\n)"
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, thinking_content, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # Fallback: analyze the transcript directly
        if len(transcript.split()) < 5:
            return "brief or incomplete conversation"
        else:
            return "general conversation"
    
    def _extract_communication_style(self, content_lower: str) -> str:
        """Extract communication style from thinking content"""
        styles = [
            "formal", "informal", "casual", "professional", "technical", "conversational",
            "repetitive", "hesitant", "confident", "unclear", "direct", "indirect"
        ]
        
        for style in styles:
            if style in content_lower:
                return style
        
        return "conversational"
    
    def _extract_concerns(self, content_lower: str) -> list:
        """Extract concerns from thinking content"""
        concerns = []
        # Only flag actual concerns, not discussion of concepts
        concern_phrases = [
            "concerning content", "worried about", "alarming behavior", 
            "distressing message", "troubling speech", "actual problem", 
            "real issue", "concerning elements present"
        ]
        
        for phrase in concern_phrases:
            if phrase in content_lower:
                concerns.append(phrase.split()[0])  # Just the main word
        
        return concerns
    
    def _extract_confidence(self, content_lower: str) -> float:
        """Extract confidence level from thinking content"""
        high_confidence = ["very confident", "quite sure", "definitely", "clearly", "obviously"]
        medium_confidence = ["somewhat", "moderately", "fairly", "reasonably"]
        low_confidence = ["not sure", "uncertain", "unclear", "difficult to", "hard to tell"]
        
        for phrase in high_confidence:
            if phrase in content_lower:
                return 0.9
        
        for phrase in medium_confidence:
            if phrase in content_lower:
                return 0.7
        
        for phrase in low_confidence:
            if phrase in content_lower:
                return 0.5
        
        return 0.8  # Default confidence
    
    def _extract_key_themes(self, content_lower: str) -> list:
        """Extract key themes from thinking content"""
        themes = []
        theme_indicators = [
            "work", "family", "technology", "education", "health", "social", "personal",
            "business", "casual", "technical", "emotional", "daily life", "routine"
        ]
        
        for theme in theme_indicators:
            if theme in content_lower:
                themes.append(theme)
        
        return themes if themes else ["general"]
    
    def _extract_mental_state(self, content_lower: str) -> str:
        """Extract mental state from thinking content"""
        states = [
            "stressed", "relaxed", "confused", "focused", "distracted", "alert",
            "tired", "energetic", "calm", "agitated", "content", "frustrated"
        ]
        
        for state in states:
            if state in content_lower:
                return state
        
        return "normal"
    
    def _extract_toxicity(self, content_lower: str) -> str:
        """Extract toxicity level from thinking content"""
        # Only flag if explicitly mentioned as present
        if "toxic content" in content_lower or "offensive language" in content_lower:
            return "moderate"
        elif "profanity present" in content_lower or "inappropriate language" in content_lower:
            return "mild"
        else:
            return "none"
    
    def _extract_profanity(self, content_lower: str) -> bool:
        """Extract profanity presence from thinking content"""
        # Only flag if explicitly mentioned as present
        profanity_phrases = [
            "profanity present", "swear words", "curse words", 
            "profanity detected", "contains profanity"
        ]
        return any(phrase in content_lower for phrase in profanity_phrases)
    
    def _extract_emergency(self, content_lower: str) -> bool:
        """Extract emergency indicators from thinking content"""
        # Only flag if explicitly mentioned as present, not just discussed
        emergency_phrases = [
            "emergency present", "urgent situation", "actual emergency", 
            "real emergency", "emergency detected", "emergency situation"
        ]
        return any(phrase in content_lower for phrase in emergency_phrases)
    
    def _extract_threat(self, content_lower: str) -> bool:
        """Extract threat indicators from thinking content"""
        # Only flag if explicitly mentioned as present, not just discussed
        threat_phrases = [
            "threat present", "actual threat", "real threat", 
            "threat detected", "threatening behavior", "violence present"
        ]
        return any(phrase in content_lower for phrase in threat_phrases)
    
    def _extract_distress_level(self, content_lower: str) -> str:
        """Extract distress level from thinking content"""
        if "severe" in content_lower or "extreme" in content_lower:
            return "severe"
        elif "moderate" in content_lower or "distress" in content_lower:
            return "moderate"
        elif "mild" in content_lower or "slight" in content_lower:
            return "mild"
        else:
            return "none"
    
    def _create_summary(self, thinking_content: str, sentiment: str, emotional_tone: str) -> str:
        """Create a summary from the thinking content"""
        # Extract the first sentence or key insight
        sentences = thinking_content.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 20:
                return f"DeepSeek analysis: {sentiment} sentiment with {emotional_tone} tone - {first_sentence[:100]}..."
        
        return f"DeepSeek analysis: {sentiment} sentiment with {emotional_tone} emotional tone"
    
    def _fallback_analysis(self, transcript: str) -> Dict:
        """Fallback to simple keyword-based analysis"""
        # Simple keyword matching as fallback
        negative_keywords = ["help", "emergency", "fire", "police", "danger", "problem", "wrong"]
        positive_keywords = ["thank", "good", "great", "wonderful", "happy", "excellent"]
        
        text_lower = transcript.lower() if transcript else ""
        
        found_negative = [kw for kw in negative_keywords if kw in text_lower]
        found_positive = [kw for kw in positive_keywords if kw in text_lower]
        
        if found_negative and not found_positive:
            sentiment = "negative"
        elif found_positive and not found_negative:
            sentiment = "positive"
        elif found_negative and found_positive:
            sentiment = "mixed"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "confidence": 0.6,  # Lower confidence for fallback
            "emotional_tone": "unknown",
            "mental_state": "unknown",
            "conversation_topic": "unknown topic",
            "content_analysis": f"Fallback keyword analysis - {sentiment} sentiment detected",
            "toxicity": "none",
            "profanity": False,
            "emergency": "emergency" in text_lower or "help" in text_lower,
            "threat": False,
            "distress_level": "none",
            "communication_style": "unknown",
            "key_themes": ["general"],
            "concerns": found_negative if found_negative else [],
            "summary": f"Fallback keyword analysis: {sentiment} sentiment detected",
            "source": "fallback",
            "model": "keyword-based",
            "original_transcript": transcript
        }
    
    def is_concerning_content(self, analysis: Dict) -> bool:
        """Check if content analysis indicates concerning speech"""
        if not analysis:
            return False
        
        # Check various concerning indicators
        concerning_factors = [
            analysis.get("sentiment") in ["negative"],
            analysis.get("toxicity") in ["moderate", "severe"],
            analysis.get("profanity", False),
            analysis.get("emergency", False),
            analysis.get("threat", False),
            analysis.get("distress_level") in ["moderate", "severe"],
            analysis.get("confidence", 0) > 0.7  # High confidence in negative assessment
        ]
        
        # Return True if multiple concerning factors are present
        return sum(concerning_factors) >= 2
    
    def get_analysis_summary(self, analysis: Dict) -> str:
        """Get a human-readable summary of the analysis"""
        if not analysis:
            return "No analysis available"
        
        summary_parts = []
        
        # Sentiment and tone
        sentiment = analysis.get("sentiment", "unknown")
        tone = analysis.get("emotional_tone", "unknown")
        confidence = analysis.get("confidence", 0)
        summary_parts.append(f"Sentiment: {sentiment} ({tone} tone, {confidence:.1%} confidence)")
        
        # Topic
        topic = analysis.get("conversation_topic", "unknown")
        if topic != "unknown topic":
            summary_parts.append(f"Topic: {topic}")
        
        # Concerns
        concerns = analysis.get("concerns", [])
        if concerns:
            summary_parts.append(f"Concerns: {', '.join(concerns)}")
        
        # Emergency/threat indicators
        if analysis.get("emergency", False):
            summary_parts.append("⚠️ Emergency indicators detected")
        if analysis.get("threat", False):
            summary_parts.append("🚨 Threat indicators detected")
        
        # Distress level
        distress = analysis.get("distress_level", "none")
        if distress != "none":
            summary_parts.append(f"Distress: {distress}")
        
        return " | ".join(summary_parts)