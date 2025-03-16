#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speaker diarization module using pre-trained models.
"""

import os
import logging
import signal
import time
from pathlib import Path
import tempfile
import soundfile as sf
import multiprocessing
import contextlib

logger = logging.getLogger("voice-separation.diarization")

class TimeoutException(Exception):
    """Exception raised when a function call times out."""
    pass

@contextlib.contextmanager
def time_limit(seconds):
    """Context manager to limit execution time of a block of code."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # Set the timeout handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Cancel the timeout
        signal.alarm(0)

class SpeakerDiarizer:
    """Class for speaker diarization using pre-trained models."""
    
    def __init__(self, input_file, model_dir="models", timeout=300):
        """
        Initialize speaker diarizer.
        
        Args:
            input_file (str): Path to input audio file
            model_dir (str): Directory to save/load models
            timeout (int): Maximum time in seconds to wait for diarization (default: 5 minutes)
        """
        self.input_file = input_file
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        
        # Check if pyannote is available - safer import approach
        self.pyannote_available = False
        self.diarization_pipeline = None
        
        try:
            # First check if pyannote.audio is available
            import pyannote.audio
            
            try:
                # Then try to import Pipeline specifically
                from pyannote.audio import Pipeline
                self.pyannote_available = True
                self.diarization_pipeline = self._load_diarization_model()
                logger.debug("PyAnnote modules imported successfully")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not import Pipeline from pyannote.audio: {e}")
        except ImportError:
            logger.warning("PyAnnote not available for diarization.")
    
    def _load_diarization_model(self):
        """
        Load PyAnnote diarization model.
        
        Returns:
            Pipeline or None: PyAnnote diarization pipeline
        """
        try:
            # Import inside method to avoid top-level import issues
            from pyannote.audio import Pipeline
            
            # Try to load the pre-trained model
            logger.info("Loading diarization model, this may take a moment...")
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                              use_auth_token=os.environ.get('HF_TOKEN', None))
            logger.debug("PyAnnote diarization model loaded successfully")
            return pipeline
        except Exception as e:
            logger.warning(f"Could not load PyAnnote diarization model: {e}")
            return None
    
    def _diarize_with_timeout(self, input_file):
        """
        Run diarization with a timeout to prevent hanging.
        
        Args:
            input_file (str): Path to audio file
            
        Returns:
            diarization_result or None if timeout
        """
        def run_diarization(input_file, result_queue):
            try:
                result = self.diarization_pipeline(input_file)
                result_queue.put(result)
            except Exception as e:
                logger.error(f"Error in diarization process: {e}")
                result_queue.put(None)
        
        # Create a queue to get the result
        result_queue = multiprocessing.Queue()
        
        # Create and start the process
        process = multiprocessing.Process(
            target=run_diarization,
            args=(input_file, result_queue)
        )
        
        # Start the process
        process.start()
        
        # Wait for the process to finish or timeout
        process.join(timeout=self.timeout)
        
        # If process is still running after timeout, terminate it
        if process.is_alive():
            logger.warning(f"Diarization timed out after {self.timeout} seconds")
            process.terminate()
            process.join()
            return None
        
        # Get the result if available
        if not result_queue.empty():
            return result_queue.get()
        
        return None
    
    def perform_diarization(self):
        """
        Perform speaker diarization.
        
        Returns:
            dict: Dictionary of speaker_id -> list of (start_time, end_time) tuples
        """
        if not self.pyannote_available or self.diarization_pipeline is None:
            logger.error("PyAnnote diarization model not available")
            raise ValueError("PyAnnote diarization model not available")
        
        logger.info("Performing speaker diarization with pre-trained model...")
        logger.info(f"This process may take up to {self.timeout} seconds")
        
        # Create a simpler version of the audio file if it's an MP3
        if self.input_file.lower().endswith('.mp3'):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_path = tmp.name
                
            logger.info("Converting MP3 to WAV for better compatibility...")
            try:
                import librosa
                audio, sr = librosa.load(self.input_file, sr=16000, mono=True)
                sf.write(temp_path, audio, sr)
                input_file = temp_path
                logger.debug(f"Created temporary WAV file: {temp_path}")
            except Exception as e:
                logger.warning(f"Error converting MP3 to WAV: {e}")
                input_file = self.input_file
        else:
            input_file = self.input_file
        
        # Apply diarization with timeout
        try:
            diarization_result = self._diarize_with_timeout(input_file)
            
            # Clean up temporary file if created
            if input_file != self.input_file and os.path.exists(input_file):
                os.unlink(input_file)
            
            if diarization_result is None:
                logger.warning("Diarization failed or timed out")
                raise ValueError("Diarization process timed out")
            
            # Convert to speaker timeline
            speaker_timeline = {}
            
            for segment, track, speaker in diarization_result.itertracks(yield_label=True):
                if speaker not in speaker_timeline:
                    speaker_timeline[speaker] = []
                
                speaker_timeline[speaker].append((segment.start, segment.end))
            
            # Sort each speaker's segments by start time
            for speaker in speaker_timeline:
                speaker_timeline[speaker].sort(key=lambda x: x[0])
            
            # Merge overlapping or very close segments for each speaker
            for speaker in speaker_timeline:
                merged_segments = []
                if speaker_timeline[speaker]:
                    current_segment = speaker_timeline[speaker][0]
                    
                    for next_segment in speaker_timeline[speaker][1:]:
                        # If segments overlap or are very close (less than 0.5s apart), merge them
                        if next_segment[0] - current_segment[1] < 0.5:
                            current_segment = (current_segment[0], next_segment[1])
                        else:
                            merged_segments.append(current_segment)
                            current_segment = next_segment
                    
                    # Add the last segment
                    merged_segments.append(current_segment)
                    
                    speaker_timeline[speaker] = merged_segments
            
            logger.info(f"Diarization found {len(speaker_timeline)} speakers")
            
            # Log total duration for each speaker
            for speaker, segments in speaker_timeline.items():
                total_duration = sum(end - start for start, end in segments)
                logger.debug(f"Speaker {speaker}: {len(segments)} segments, {total_duration:.2f} seconds total")
            
            return speaker_timeline
            
        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            
            # Clean up temporary file if created
            if input_file != self.input_file and os.path.exists(input_file):
                os.unlink(input_file)
                
            raise ValueError(f"Failed to perform diarization: {str(e)}")