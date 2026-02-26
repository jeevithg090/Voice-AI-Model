"""
Filler and Back-channel Manager for Layer 2 - Voice AI Pipeline
Provides natural conversational behaviors to make the AI more human-like.

Features:
    - Multilingual fillers (English, Hindi, Tamil, Telugu, Kannada)
    - Context-aware back-channel selection
    - Integration with turn detector and LLM pipeline
"""

import random
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import time


class ConversationState(Enum):
    """States for conversation flow."""
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    USER_PAUSED = "user_paused"  # Short pause during speech
    TURN_COMPLETE = "turn_complete"
    AI_THINKING = "ai_thinking"
    AI_SPEAKING = "ai_speaking"


class FillerType(Enum):
    """Types of fillers based on context."""
    THINKING = "thinking"      # "um", "let me see"
    HESITATION = "hesitation"  # "uh", "well"
    TRANSITION = "transition"  # "so", "okay so"
    ACKNOWLEDGMENT = "ack"     # "okay", "right"


@dataclass
class FillerConfig:
    """Configuration for filler behavior."""
    min_processing_time_ms: int = 1500  # Trigger filler if LLM takes longer
    max_fillers_per_turn: int = 2       # Don't overuse
    backchannel_pause_ms: int = 200     # Min pause to trigger backchannel
    backchannel_cooldown_ms: int = 3000 # Cooldown between backchannels
    use_script_detection_fallback: bool = True  # Only if session language unknown


# ========================
# Multilingual Filler Definitions
# ========================

FILLERS: Dict[str, Dict[str, List[str]]] = {
    "en": {
        "thinking": ["um", "let me see", "let me think"],
        "hesitation": ["uh", "well", "hmm"],
        "transition": ["so", "okay so", "alright"],
        "acknowledgment": ["okay", "right", "got it"],
    },
    "hi": {  # Hindi
        "thinking": ["देखिए", "सोचता हूँ", "एक मिनट"],
        "hesitation": ["अच्छा", "वैसे", "हम्म"],
        "transition": ["तो", "तो देखिए", "अच्छा तो"],
        "acknowledgment": ["ठीक है", "समझा", "जी हाँ"],
    },
    "ta": {  # Tamil
        "thinking": ["பாருங்க", "யோசிக்கிறேன்", "ஒரு நிமிஷம்"],
        "hesitation": ["சரி", "அது", "ம்ம்"],
        "transition": ["அப்போ", "சரி அப்போ", "ஆமா"],
        "acknowledgment": ["சரி", "புரியுது", "ஓகே"],
    },
    "te": {  # Telugu
        "thinking": ["చూడండి", "ఆలోచిస్తున్నా", "ఒక్క క్షణం"],
        "hesitation": ["అది", "బాగా", "హ్మ్"],
        "transition": ["అయితే", "సరే అయితే", "అవును"],
        "acknowledgment": ["సరే", "అర్థమైంది", "ఓకే"],
    },
    "kn": {  # Kannada
        "thinking": ["ನೋಡಿ", "ಯೋಚಿಸ್ತೀನಿ", "ಒಂದು ನಿಮಿಷ"],
        "hesitation": ["ಅದು", "ಹಾಂ", "ಹ್ಮ್ಮ್"],
        "transition": ["ಹಾಗಾದ್ರೆ", "ಸರಿ ಹಾಗಾದ್ರೆ", "ಹೌದು"],
        "acknowledgment": ["ಸರಿ", "ಅರ್ಥವಾಯಿತು", "ಓಕೆ"],
    },
}

BACKCHANNELS: Dict[str, List[str]] = {
    "en": ["mm-hmm", "I see", "right", "okay", "uh-huh", "got it", "yes"],
    "hi": ["अच्छा", "हाँ हाँ", "समझा", "ठीक है", "हम्म", "जी"],
    "ta": ["ஆமா ஆமா", "சரி சரி", "புரியுது", "ஓகே", "ம்ம்", "ஆமா"],
    "te": ["అవును అవును", "సరే సరే", "అర్థమైంది", "ఓకే", "హ్మ్", "అవును"],
    "kn": ["ಹೌದು ಹೌದು", "ಸರಿ ಸರಿ", "ಅರ್ಥವಾಯಿತು", "ಓಕೆ", "ಹ್ಮ್ಮ್", "ಹೌದು"],
}

# Language detection patterns (simple heuristics)
LANGUAGE_PATTERNS = {
    "hi": ["क", "ख", "ग", "घ", "च", "छ", "ज", "झ"],  # Hindi/Devanagari
    "ta": ["க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந"],  # Tamil
    "te": ["క", "ఖ", "గ", "ఘ", "చ", "ఛ", "జ", "ఝ"],  # Telugu
    "kn": ["ಕ", "ಖ", "ಗ", "ಘ", "ಚ", "ಛ", "ಜ", "ಝ"],  # Kannada
}


class FillerBackchannelManager:
    """
    Manages fillers and backchannels for natural conversation flow.
    
    IMPORTANT Voice UX Rules:
    1. Language is set via session (from STT), NOT script detection
    2. Fillers are spoken separately, NEVER inserted into LLM text
    3. Backchannels NEVER overlap with AI speech
    
    Usage:
        manager = FillerBackchannelManager()
        manager.set_session_language("hi")  # Set from STT detection
        
        # Get a filler before LLM response  
        filler = manager.get_filler()  # Uses session language
        
        # Check if backchannel appropriate (checks AI not speaking)
        if manager.should_backchannel(turn_state):
            backchannel = manager.get_backchannel()
    """
    
    def __init__(self, config: Optional[FillerConfig] = None):
        self.config = config or FillerConfig()
        self._filler_count = 0
        self._last_backchannel_time = 0
        self._last_filler_time = 0
        self._session_language = "en"
        self._conversation_state = ConversationState.IDLE
    
    def detect_language(self, text: str) -> str:
        """
        Get language for the session.
        
        IMPORTANT: Prefers session language (set from STT) over script detection.
        Script detection is only a fallback for edge cases.
        
        Args:
            text: Input text (used only as fallback)
            
        Returns:
            Language code (en, hi, ta, te, kn)
        """
        # Primary: Use session language (should be set from STT)
        if self._session_language:
            return self._session_language
        
        # Fallback: Script-based detection (for edge cases only)
        if self.config.use_script_detection_fallback:
            for lang, patterns in LANGUAGE_PATTERNS.items():
                if any(char in text for char in patterns):
                    return lang
        
        return "en"  # Default to English
    
    def set_session_language(self, language: str):
        """
        Set the language for the current session.
        Should be called with STT-detected language at session start.
        """
        if language in FILLERS:
            self._session_language = language
    
    def set_conversation_state(self, state: ConversationState):
        """
        Update conversation state. Critical for preventing backchannel overlap.
        """
        self._conversation_state = state
    
    def get_filler(
        self, 
        language: Optional[str] = None, 
        filler_type: FillerType = FillerType.THINKING
    ) -> str:
        """
        Get a contextually appropriate filler.
        
        Args:
            language: Language code (en, hi, ta, te, kn)
            filler_type: Type of filler needed
            
        Returns:
            Filler phrase
        """
        lang = language or self._session_language
        if lang not in FILLERS:
            lang = "en"
        
        fillers = FILLERS[lang].get(filler_type.value, FILLERS[lang]["thinking"])
        filler = random.choice(fillers)
        
        self._last_filler_time = time.time() * 1000
        self._filler_count += 1
        
        return filler
    
    def get_backchannel(self, language: Optional[str] = None) -> str:
        """
        Get a backchannel phrase for active listening.
        
        Args:
            language: Language code
            
        Returns:
            Backchannel phrase
        """
        lang = language or self._session_language
        if lang not in BACKCHANNELS:
            lang = "en"
        
        backchannel = random.choice(BACKCHANNELS[lang])
        self._last_backchannel_time = time.time() * 1000
        
        return backchannel
    
    def should_use_filler(self, processing_time_ms: int) -> bool:
        """
        Determine if a filler should be used based on processing time.
        
        Args:
            processing_time_ms: Expected or elapsed processing time
            
        Returns:
            True if filler is appropriate
        """
        # Don't overuse fillers
        if self._filler_count >= self.config.max_fillers_per_turn:
            return False
        
        # Only use filler if processing takes long enough
        return processing_time_ms >= self.config.min_processing_time_ms
    
    def should_backchannel(self, turn_state: Dict) -> bool:
        """
        Determine if a backchannel is appropriate based on turn state.
        
        CRITICAL: NEVER backchannel while AI is speaking!
        
        Args:
            turn_state: Output from turn detector
            
        Returns:
            True if backchannel opportunity detected
        """
        # HARD RULE: Never backchannel while AI is speaking
        if self._conversation_state == ConversationState.AI_SPEAKING:
            return False
        
        if self._conversation_state == ConversationState.AI_THINKING:
            return False
        
        current_time = time.time() * 1000
        
        # Check cooldown
        if (current_time - self._last_backchannel_time) < self.config.backchannel_cooldown_ms:
            return False
        
        # Check for backchannel opportunity from turn detector
        if turn_state.get("backchannel_opportunity", False):
            return True
        
        # Check for pause during speech
        silence_ms = turn_state.get("silence_ms", 0)
        is_speaking = turn_state.get("is_speaking", True)
        
        # Short pause during speech = backchannel opportunity
        if not is_speaking and self.config.backchannel_pause_ms <= silence_ms < 500:
            return True
        
        return False
    
    def get_response_prefix(
        self, 
        query: str, 
        processing_time_ms: int,
        language: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Get an optional filler to speak BEFORE LLM response.
        
        IMPORTANT: This returns a filler to be SPOKEN SEPARATELY.
        Do NOT concatenate this with LLM text! The correct flow is:
        
        1. Speak filler via TTS
        2. Brief pause
        3. Speak LLM response via TTS
        
        Args:
            query: User's query
            processing_time_ms: Expected processing time
            language: Language code (uses session language if not provided)
            
        Returns:
            (filler_text, was_filler_used) - filler to speak separately
        """
        lang = language or self._session_language
        
        if not self.should_use_filler(processing_time_ms):
            return "", False
        
        # Select filler type based on query
        if "?" in query:
            filler_type = FillerType.THINKING
        elif len(query) > 100:
            filler_type = FillerType.ACKNOWLEDGMENT
        else:
            filler_type = FillerType.HESITATION
        
        filler = self.get_filler(lang, filler_type)
        return filler, True
    
    def reset_turn(self):
        """Reset counters for a new conversation turn."""
        self._filler_count = 0
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            "fillers_used": self._filler_count,
            "session_language": self._session_language,
            "conversation_state": self._conversation_state.value,
            "last_filler_time": self._last_filler_time,
            "last_backchannel_time": self._last_backchannel_time,
        }


# ========================
# LLM Integration Helper
# ========================

def get_enhanced_system_prompt(base_prompt: str, language: str = "en") -> str:
    """
    Enhance system prompt with filler instructions for LLM.
    
    Args:
        base_prompt: Original system prompt
        language: Target language
        
    Returns:
        Enhanced prompt with filler guidance
    """
    filler_examples = FILLERS.get(language, FILLERS["en"])
    
    filler_guidance = """

CONVERSATIONAL STYLE:
- For complex questions, you may naturally start with a brief thinking phrase
- Keep responses conversational and warm
- Match the user's language and formality level
- Avoid robotic or overly formal responses

AVAILABLE THINKING PHRASES (use sparingly):
- Thinking: {thinking}
- Transitions: {transition}
""".format(
        thinking=", ".join(filler_examples.get("thinking", ["let me see"])[:2]),
        transition=", ".join(filler_examples.get("transition", ["so"])[:2]),
    )
    
    return base_prompt + filler_guidance
