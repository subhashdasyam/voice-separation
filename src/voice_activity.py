#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voice activity detection module.
"""

import os
import numpy as np
import librosa
import torch
import webrtcvad
from pathlib import Path
import logging
import tempfile
import soundfile as sf

logger = logging.getLogger("voice-separation.vad")

class VoiceActivityDetector:
    """Class for voice activity detection."""
    
    def __init__(self, audio, sample_rate, device=None, vad_agg_level=3):
        """
        Initialize voice activity detector.
        
        Args:
            audio (ndarray): Audio data
            sample_rate (int): Sample rate
            device (str): Device to run models on ('cuda' or 'cpu')
            vad_agg_level (int): WebRTC VAD aggressiveness level (0-3)
        """
        self.audio = audio
        self.sample_rate = sample_rate
        self.vad_agg_level = vad_agg_level
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.debug(f"Initialized VAD with device: {self.device}")
        
        # Try to import pyannote for neural VAD
        self.pyannote_available = False
        self.vad_pipeline = None
        
        try:
            # Check if pyannote.audio is available before trying to import specific modules
            import pyannote.audio
            
            try:
                # Now try to load Pipeline
                from pyannote.audio import Pipeline
                self.pyannote_available = True
                self.vad_pipeline = self._load_pyannote_vad()
                logger.debug("PyAnnote VAD loaded successfully")
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not import Pipeline from pyannote.audio: {e}")
        except ImportError:
            logger.warning("PyAnnote not available. Using WebRTC VAD instead.")
    
    def _load_pyannote_vad(self):
        """
        Load PyAnnote VAD model.
        
        Returns:
            Pipeline or None: PyAnnote VAD pipeline
        """
        try:
            # Import inside the method to avoid top-level import issues
            from pyannote.audio import Pipeline
            
            # Try to load the pre-trained model
            pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                               use_auth_token=os.environ.get('HF_TOKEN', None))
            return pipeline
        except Exception as e:
            logger.warning(f"Could not load PyAnnote VAD model: {e}")
            return None
    
    def detect_with_pyannote(self):
        """
        Detect voice activity using PyAnnote VAD.
        
        Returns:
            list: List of (start_time, end_time) tuples
        """
        if self.vad_pipeline is None:
            logger.warning("PyAnnote VAD model not available.")
            return []
        
        # Import required modules inside the method
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            logger.warning("PyAnnote Pipeline not available.")
            return []
        
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name
        
        sf.write(temp_path, self.audio, self.sample_rate)
        
        try:
            # Apply VAD
            vad_result = self.vad_pipeline(temp_path)
            
            # Extract segments
            segments = []
            for segment, _, _ in vad_result.itertracks(yield_label=True):
                segments.append((segment.start, segment.end))
            
            logger.debug(f"PyAnnote VAD detected {len(segments)} segments")
            
            # Clean up
            os.unlink(temp_path)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error with PyAnnote VAD: {e}")
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return []
    
    def detect_with_webrtc(self, frame_duration=30, padding_ms=300, merge_threshold_ms=500):
        """
        Detect voice activity using WebRTC VAD.
        
        Args:
            frame_duration (int): Frame duration in ms
            padding_ms (int): Padding to add to segments in ms
            merge_threshold_ms (int): Threshold to merge segments in ms
            
        Returns:
            list: List of (start_time, end_time) tuples
        """
        logger.debug(f"Running WebRTC VAD with frame_duration={frame_duration}ms, agg_level={self.vad_agg_level}")
        
        # Create WebRTC VAD instance
        vad = webrtcvad.Vad(self.vad_agg_level)
        
        # Resample to 16kHz if needed (WebRTC VAD only supports 8, 16, 32, 48 kHz)
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            logger.debug(f"Resampling from {self.sample_rate}Hz to 16000Hz for WebRTC VAD")
            audio = librosa.resample(self.audio, orig_sr=self.sample_rate, target_sr=16000)
            sample_rate = 16000
        else:
            audio = self.audio
            sample_rate = self.sample_rate
        
        # Convert to PCM format
        pcm_data = (audio * 32767).astype(np.int16)
        
        # Calculate frame parameters
        frame_length = int(sample_rate * (frame_duration / 1000.0))
        frame_step = frame_length  # Non-overlapping frames
        
        # Frame the audio
        frames = librosa.util.frame(pcm_data, frame_length=frame_length, hop_length=frame_step)
        frames = frames.T  # Transpose to get frames as rows
        
        # Detect speech in each frame
        speech_frames = []
        for i, frame in enumerate(frames):
            # Ensure the frame is the right length
            if len(frame) != frame_length:
                continue
                
            try:
                is_speech = vad.is_speech(frame.tobytes(), sample_rate)
                speech_frames.append((i, is_speech))
            except Exception as e:
                logger.warning(f"Error processing frame {i}: {e}")
        
        # Group consecutive speech frames
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in speech_frames:
            if is_speech and not in_speech:
                # Speech start
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                # Speech end
                end_frame = i
                in_speech = False
                
                # Convert frames to time (seconds)
                start_time = start_frame * frame_duration / 1000.0
                end_time = end_frame * frame_duration / 1000.0
                
                # Only add if segment is long enough (at least 300ms)
                if end_time - start_time >= 0.3:
                    segments.append((start_time, end_time))
        
        # Don't forget the last segment if we end in speech
        if in_speech:
            end_frame = len(speech_frames)
            start_time = start_frame * frame_duration / 1000.0
            end_time = end_frame * frame_duration / 1000.0
            
            if end_time - start_time >= 0.3:
                segments.append((start_time, end_time))
        
        # Merge segments that are close together
        merged_segments = []
        if segments:
            current_segment = segments[0]
            
            for next_segment in segments[1:]:
                # If next segment starts soon after current ends, merge them
                if next_segment[0] - current_segment[1] < merge_threshold_ms / 1000.0:
                    current_segment = (current_segment[0], next_segment[1])
                else:
                    # Add padding to segment
                    padded_start = max(0, current_segment[0] - padding_ms / 1000.0)
                    padded_end = min(len(audio) / sample_rate, 
                                    current_segment[1] + padding_ms / 1000.0)
                    
                    merged_segments.append((padded_start, padded_end))
                    current_segment = next_segment
            
            # Add the last segment
            padded_start = max(0, current_segment[0] - padding_ms / 1000.0)
            padded_end = min(len(audio) / sample_rate, 
                            current_segment[1] + padding_ms / 1000.0)
            
            merged_segments.append((padded_start, padded_end))
        
        logger.debug(f"WebRTC VAD detected {len(merged_segments)} segments after merging")
        return merged_segments
    
    def detect_voice_activity(self):
        """
        Detect voice activity using the best available method.
        
        Returns:
            list: List of (start_time, end_time) tuples
        """
        # Try PyAnnote VAD first if available
        if self.pyannote_available and self.vad_pipeline is not None:
            try:
                logger.info("Detecting voice activity with PyAnnote VAD...")
                segments = self.detect_with_pyannote()
                if segments:
                    return segments
            except Exception as e:
                logger.warning(f"PyAnnote VAD failed: {e}")
                logger.info("Falling back to WebRTC VAD...")
        
        # Fall back to WebRTC VAD
        logger.info("Detecting voice activity with WebRTC VAD...")
        segments = self.detect_with_webrtc()
        
        # Post-process segments: remove very short segments
        segments = [(start, end) for start, end in segments if (end - start) >= 0.3]
        
        return segments