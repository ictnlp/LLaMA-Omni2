from .speech_generator import LLMSpeechGenerator


def build_speech_generator(config):    
    return LLMSpeechGenerator(config)
