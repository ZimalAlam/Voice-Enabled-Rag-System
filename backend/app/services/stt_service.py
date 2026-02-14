import asyncio
import io
import logging
import time
import wave
from typing import Optional, AsyncGenerator, List
import tempfile
import os
import json

from google.cloud import speech
import speech_recognition as sr
from pydub import AudioSegment
import numpy as np

from app.config import settings
from app.models.schemas import STTResponse

logger = logging.getLogger(__name__)

class STTService:
    """Speech-to-Text service with streaming capabilities using Google Cloud Speech-to-Text"""
    
    def __init__(self):
        self.google_speech_client = None
        self.speech_recognizer = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize STT clients based on configuration"""
        try:
            # Initialize Google Cloud Speech client
            if settings.google_cloud_service_account_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_cloud_service_account_path
                self.google_speech_client = speech.SpeechClient()
                logger.info("Google Cloud Speech-to-Text client initialized")
            elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                self.google_speech_client = speech.SpeechClient()
                logger.info("Google Cloud Speech-to-Text client initialized from environment")
            else:
                logger.warning("Google Cloud credentials not found, STT may not work properly")
            
            # Initialize speech recognition for real-time processing
            self.speech_recognizer = sr.Recognizer()
            self.speech_recognizer.energy_threshold = 300
            self.speech_recognizer.dynamic_energy_threshold = True
            self.speech_recognizer.pause_threshold = 0.8
            self.speech_recognizer.operation_timeout = 1.0
            
        except Exception as e:
            logger.error(f"Failed to initialize STT clients: {e}")
            raise
    
    async def transcribe_stream(self, audio_data: bytes, language: str = "en") -> STTResponse:
        """
        Transcribe streaming audio data
        
        Args:
            audio_data: Raw audio bytes
            language: Language code for transcription
            
        Returns:
            STTResponse with transcription results
        """
        start_time = time.time()
        
        try:
            # Process audio data
            text, confidence = await self._process_audio_chunk(audio_data, language)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return STTResponse(
                text=text,
                confidence=confidence,
                language=language,
                is_partial=True,  # Streaming results are partial
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return STTResponse(
                text="",
                confidence=0.0,
                language=language,
                is_partial=True,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def transcribe_file(self, audio_file: io.BytesIO, language: str = "en") -> STTResponse:
        """
        Transcribe a complete audio file
        
        Args:
            audio_file: Audio file data
            language: Language code for transcription
            
        Returns:
            STTResponse with complete transcription
        """
        start_time = time.time()
        
        try:
            if self.google_speech_client:
                result = await self._transcribe_with_google_speech(audio_file, language)
            else:
                result = await self._transcribe_with_speech_recognition(audio_file, language)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return STTResponse(
                text=result["text"],
                confidence=result.get("confidence", 0.9),
                language=result.get("language", language),
                is_partial=False,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            raise
    
    async def _process_audio_chunk(self, audio_data: bytes, language: str) -> tuple[str, float]:
        """Process a chunk of audio data for streaming transcription"""
        try:
            # Convert bytes to audio segment
            audio_segment = AudioSegment(
                data=audio_data,
                sample_width=2,  # 16-bit
                frame_rate=16000,  # 16kHz
                channels=1  # Mono
            )
            
            # Check if audio is long enough and has enough energy
            if len(audio_segment) < 500:  # Less than 500ms
                return "", 0.0
            
            # Convert to numpy array for processing
            audio_array = np.array(audio_segment.get_array_of_samples())
            
            # Check for silence (simple energy-based detection)
            energy = np.sqrt(np.mean(audio_array ** 2))
            if energy < 100:  # Threshold for silence
                return "", 0.0
            
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_segment.export(temp_file.name, format="wav")
                
                if self.google_speech_client:
                    result = await self._transcribe_chunk_google_speech(temp_file.name, language)
                else:
                    result = await self._transcribe_chunk_speech_recognition(temp_file.name)
                
                # Clean up temp file
                os.unlink(temp_file.name)
                
                return result
            
        except Exception as e:
            logger.error(f"Audio chunk processing failed: {e}")
            return "", 0.0
    
    async def _transcribe_chunk_google_speech(self, audio_file_path: str, language: str) -> tuple[str, float]:
        """Transcribe audio chunk using Google Cloud Speech-to-Text"""
        try:
            # Read audio file
            with open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()
            
            # Configure recognition
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self._convert_language_code(language),
                model=settings.google_speech_model,
                use_enhanced=True,
                enable_automatic_punctuation=True,
            )
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.google_speech_client.recognize(config=config, audio=audio)
            )
            
            if response.results:
                alternative = response.results[0].alternatives[0]
                text = alternative.transcript.strip()
                confidence = alternative.confidence
                return text, confidence
            else:
                return "", 0.0
            
        except Exception as e:
            logger.error(f"Google Speech transcription failed: {e}")
            return "", 0.0
    
    async def _transcribe_chunk_speech_recognition(self, audio_file_path: str) -> tuple[str, float]:
        """Transcribe audio chunk using speech_recognition library as fallback"""
        try:
            # Use speech_recognition library
            with sr.AudioFile(audio_file_path) as source:
                audio = self.speech_recognizer.record(source)
            
            # Try Google Speech Recognition (free tier)
            try:
                text = self.speech_recognizer.recognize_google(audio)
                return text.strip(), 0.8  # Estimated confidence
            except sr.UnknownValueError:
                return "", 0.0
            except sr.RequestError as e:
                logger.error(f"Google Speech Recognition error: {e}")
                return "", 0.0
                
        except Exception as e:
            logger.error(f"Speech recognition fallback failed: {e}")
            return "", 0.0
    
    async def _transcribe_with_google_speech(self, audio_file: io.BytesIO, language: str) -> dict:
        """Transcribe complete audio file using Google Cloud Speech-to-Text"""
        try:
            content = audio_file.read()
            
            # Configure recognition for longer audio
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self._convert_language_code(language),
                model=settings.google_speech_model,
                use_enhanced=True,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                audio_channel_count=1,
            )
            
            # Use long running recognize for files > 1 minute
            if len(content) > 1024 * 1024:  # 1MB threshold
                # For long audio, use async operation
                operation = self.google_speech_client.long_running_recognize(
                    config=config, audio=audio
                )
                response = operation.result(timeout=300)  # 5 minute timeout
            else:
                # For short audio, use synchronous recognition
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.google_speech_client.recognize(config=config, audio=audio)
                )
            
            if response.results:
                # Combine all transcripts
                full_transcript = ""
                total_confidence = 0.0
                
                for result in response.results:
                    alternative = result.alternatives[0]
                    full_transcript += alternative.transcript + " "
                    total_confidence += alternative.confidence
                
                avg_confidence = total_confidence / len(response.results)
                
                return {
                    "text": full_transcript.strip(),
                    "confidence": avg_confidence,
                    "language": language
                }
            else:
                return {"text": "", "confidence": 0.0, "language": language}
                
        except Exception as e:
            logger.error(f"Google Speech file transcription failed: {e}")
            raise
    
    async def _transcribe_with_speech_recognition(self, audio_file: io.BytesIO, language: str) -> dict:
        """Fallback transcription using speech_recognition library"""
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_file.read())
                temp_file_path = temp_file.name
            
            try:
                with sr.AudioFile(temp_file_path) as source:
                    audio = self.speech_recognizer.record(source)
                
                text = self.speech_recognizer.recognize_google(audio, language=language)
                return {"text": text, "confidence": 0.8, "language": language}
                
            finally:
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Speech recognition fallback failed: {e}")
            raise
    
    async def detect_language(self, audio_file: io.BytesIO) -> str:
        """
        Detect the language of the audio
        
        Args:
            audio_file: Audio file data
            
        Returns:
            Detected language code
        """
        try:
            if not self.google_speech_client:
                return "en"  # Default to English
            
            content = audio_file.read()
            
            # Configure for language detection
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                alternative_language_codes=["en-US", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "ja-JP", "ko-KR", "zh-CN"],
            )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.google_speech_client.recognize(config=config, audio=audio)
            )
            
            if response.results:
                # Google returns the detected language in the result
                return response.results[0].language_code[:2]  # Return just the language part
            
            return "en"  # Default to English
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en"
    
    def _convert_language_code(self, language: str) -> str:
        """Convert language code to Google Speech format"""
        language_map = {
            "en": "en-US",
            "es": "es-ES", 
            "fr": "fr-FR",
            "de": "de-DE",
            "it": "it-IT",
            "pt": "pt-BR",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "zh": "zh-CN",
            "ru": "ru-RU",
            "ar": "ar-SA",
            "hi": "hi-IN",
        }
        return language_map.get(language, "en-US")
    
    def is_available(self) -> bool:
        """Check if STT service is available"""
        return self.google_speech_client is not None or self.speech_recognizer is not None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return [
            "en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh", "ru", "ar", "hi",
            "nl", "sv", "da", "no", "fi", "pl", "cs", "sk", "hu", "ro", "bg",
            "hr", "sl", "et", "lv", "lt", "mt", "ga", "cy", "eu", "ca", "gl",
            "is", "mk", "sr", "bs", "sq", "tr", "el", "he", "th", "vi", "ms",
            "id", "tl", "sw", "yo", "ig", "ha", "zu", "xh", "af", "st", "tn",
            "ts", "ss", "ve", "nr"
        ] 