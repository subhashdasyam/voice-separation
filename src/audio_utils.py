#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utilities for audio processing and manipulation.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import collections
import logging

logger = logging.getLogger("voice-separation.audio")

class AudioProcessor:
    """Class for audio processing operations."""
    
    def __init__(self, input_file, target_sr=16000):
        """
        Initialize audio processor.
        
        Args:
            input_file (str): Path to input audio file
            target_sr (int): Target sample rate for processing
        """
        self.input_file = input_file
        self.target_sr = target_sr
        self.audio, self.sample_rate = self._load_audio()
        logger.debug(f"Loaded audio with shape {self.audio.shape} and sample rate {self.sample_rate}")
    
    def _load_audio(self):
        """
        Load audio file with proper conversion.
        
        Returns:
            tuple: (audio_data, sample_rate)
        """
        # Check file extension
        file_ext = os.path.splitext(self.input_file)[1].lower()
        
        try:
            if file_ext == '.mp3':
                # Convert mp3 to wav in memory using pydub
                audio_segment = AudioSegment.from_mp3(self.input_file)
                # Convert to mono
                audio_segment = audio_segment.set_channels(1)
                # Resample if needed
                if audio_segment.frame_rate != self.target_sr:
                    audio_segment = audio_segment.set_frame_rate(self.target_sr)
                # Export to numpy array
                samples = np.array(audio_segment.get_array_of_samples())
                # Convert to float32 and normalize
                samples = samples.astype(np.float32) / (1 << (8 * audio_segment.sample_width - 1))
                return samples, audio_segment.frame_rate
            
            elif file_ext == '.wav':
                # Load wav file directly using librosa
                audio, sr = librosa.load(self.input_file, sr=self.target_sr, mono=True)
                return audio, sr
            
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise
    
    def get_duration(self):
        """Get audio duration in seconds."""
        return len(self.audio) / self.sample_rate
    
    def extract_segments(self, segments):
        """
        Extract and concatenate audio segments.
        
        Args:
            segments (list): List of (start_time, end_time) tuples in seconds
            
        Returns:
            ndarray: Concatenated audio of all segments
        """
        concatenated_audio = np.array([])
        
        for start_time, end_time in segments:
            # Convert time to samples
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Handle edge cases
            start_sample = max(0, start_sample)
            end_sample = min(len(self.audio), end_sample)
            
            # Extract segment
            if end_sample > start_sample:
                segment = self.audio[start_sample:end_sample]
                
                # Add a small silence between segments (0.1 seconds)
                if len(concatenated_audio) > 0:
                    silence = np.zeros(int(0.1 * self.sample_rate))
                    concatenated_audio = np.concatenate([concatenated_audio, silence])
                
                # Concatenate
                concatenated_audio = np.concatenate([concatenated_audio, segment])
        
        return concatenated_audio
    
    def save_audio(self, audio_data, output_path):
        """
        Save audio data to file.
        
        Args:
            audio_data (ndarray): Audio data
            output_path (str): Output file path
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Get output format from path
        output_format = os.path.splitext(output_path)[1].lower().replace('.', '')
        
        if output_format == 'wav':
            # Save directly as WAV
            sf.write(output_path, audio_data, self.sample_rate)
            
        elif output_format == 'mp3':
            # Save as temporary WAV then convert to MP3
            temp_wav = output_path.replace('.mp3', '_temp.wav')
            sf.write(temp_wav, audio_data, self.sample_rate)
            
            # Convert to MP3
            try:
                AudioSegment.from_wav(temp_wav).export(output_path, format="mp3")
                # Remove temporary WAV
                os.remove(temp_wav)
            except Exception as e:
                logger.error(f"Error converting to MP3: {e}")
                # Keep WAV file as fallback
                os.rename(temp_wav, output_path.replace('.mp3', '.wav'))
                logger.warning(f"Saved as WAV instead at: {output_path.replace('.mp3', '.wav')}")
    
    def save_separated_speakers(self, speaker_labels, segment_times, segment_audio_samples, 
                               output_dir, output_format='wav'):
        """
        Save each speaker's segments to separate audio files.
        
        Args:
            speaker_labels (ndarray): Speaker label for each segment
            segment_times (list): List of segment times as (start_time, end_time)
            segment_audio_samples (list): List of audio segments
            output_dir (str): Output directory
            output_format (str): Output file format ('wav' or 'mp3')
        """
        # Group segments by speaker
        speaker_segments = collections.defaultdict(list)
        
        for i, (label, segment_time, segment_audio) in enumerate(zip(speaker_labels, segment_times, segment_audio_samples)):
            speaker_segments[label].append((segment_time, segment_audio))
        
        # Process each speaker
        for speaker_id, segments in speaker_segments.items():
            # Sort segments by start time
            segments.sort(key=lambda x: x[0][0])
            
            # Create empty audio for this speaker
            speaker_audio = np.array([])
            
            # Previous segment end time for gap calculation
            prev_end_time = 0
            
            # Concatenate all segments for this speaker
            for segment_time, segment_audio in segments:
                start_time, end_time = segment_time
                
                # Calculate gap from previous segment
                gap_duration = start_time - prev_end_time
                
                # If there's a reasonable gap, add a short silence
                if len(speaker_audio) > 0 and gap_duration > 0 and gap_duration < 5:  # Less than 5 seconds
                    # Add a shorter silence (proportional to gap but max 0.5 second)
                    silence_duration = min(gap_duration / 3, 0.5)
                    silence = np.zeros(int(silence_duration * self.sample_rate))
                    speaker_audio = np.concatenate([speaker_audio, silence])
                
                # Add this segment
                speaker_audio = np.concatenate([speaker_audio, segment_audio])
                
                # Update previous end time
                prev_end_time = end_time
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(speaker_audio)) > 0:
                speaker_audio = speaker_audio / np.max(np.abs(speaker_audio)) * 0.9
            
            # Output file path
            output_file = os.path.join(output_dir, f"voice{speaker_id+1}.{output_format}")
            
            # Save the audio
            self.save_audio(speaker_audio, output_file)
            
            logger.info(f"Saved speaker {speaker_id+1} audio to {output_file}")
            
    def trim_silence(self, audio_data, threshold=0.01, min_silence_duration=0.3):
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio_data (ndarray): Audio data
            threshold (float): Silence threshold
            min_silence_duration (float): Minimum silence duration in seconds
            
        Returns:
            ndarray: Trimmed audio
        """
        # Calculate energy
        energy = librosa.feature.rms(y=audio_data)[0]
        
        # Find indices above threshold
        frames_above_threshold = np.where(energy > threshold)[0]
        
        if len(frames_above_threshold) == 0:
            return audio_data  # No non-silent frames found
        
        # Get start and end frames
        start_frame = frames_above_threshold[0]
        end_frame = frames_above_threshold[-1]
        
        # Convert frames to samples
        hop_length = 512  # Default hop length in librosa
        start_sample = max(0, start_frame * hop_length - int(min_silence_duration * self.sample_rate))
        end_sample = min(len(audio_data), (end_frame + 1) * hop_length + int(min_silence_duration * self.sample_rate))
        
        return audio_data[start_sample:end_sample]