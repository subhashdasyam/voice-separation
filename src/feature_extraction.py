#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speaker feature extraction module.
"""

import os
import numpy as np
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import librosa

logger = logging.getLogger("voice-separation.features")

class FeatureExtractor:
    """Class for extracting speaker-discriminative features."""
    
    def __init__(self, audio, sample_rate, device=None, model_dir="models"):
        """
        Initialize feature extractor.
        
        Args:
            audio (ndarray): Audio data
            sample_rate (int): Sample rate
            device (str): Device to run models on ('cuda' or 'cpu')
            model_dir (str): Directory to save/load models
        """
        self.audio = audio
        self.sample_rate = sample_rate
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.debug(f"Initialized feature extractor with device: {self.device}")
        
        # Initialize models
        self.resemblyzer_available = self._init_resemblyzer()
        self.wav2vec2_available = self._init_wav2vec2()
        
        if not (self.resemblyzer_available or self.wav2vec2_available):
            logger.error("No embedding models available. Feature extraction will use fallback methods.")
    
    def _init_resemblyzer(self):
        """
        Initialize Resemblyzer model for d-vector extraction.
        
        Returns:
            bool: Whether initialization was successful
        """
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            self.voice_encoder = VoiceEncoder(device=self.device)
            self.preprocess_wav = preprocess_wav
            logger.debug("Resemblyzer VoiceEncoder loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Could not initialize Resemblyzer: {e}")
            return False
    
    def _init_wav2vec2(self):
        """
        Initialize Wav2Vec2 model for embedding extraction.
        
        Returns:
            bool: Whether initialization was successful
        """
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            
            # Load or download models
            logger.debug("Loading Wav2Vec2 model...")
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
            
            logger.debug("Wav2Vec2 model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Could not initialize Wav2Vec2: {e}")
            return False
    
    def extract_resemblyzer_embeddings(self, audio_segments):
        """
        Extract d-vector embeddings using Resemblyzer.
        
        Args:
            audio_segments (list): List of audio segment arrays
            
        Returns:
            ndarray: Array of embeddings
        """
        embeddings = []
        
        for segment in tqdm(audio_segments, desc="Extracting d-vectors"):
            try:
                # Process segment for Resemblyzer
                processed_segment = self.preprocess_wav(segment, source_sr=self.sample_rate)
                
                # Get d-vector
                embedding = self.voice_encoder.embed_utterance(processed_segment)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Error extracting d-vector: {e}")
                # Add a placeholder embedding to maintain alignment
                if embeddings:
                    embeddings.append(np.zeros_like(embeddings[0]))
                else:
                    # No previous embedding to reference shape, use a default
                    embeddings.append(np.zeros(256))
        
        return np.array(embeddings)
    
    def extract_wav2vec2_embeddings(self, audio_segments):
        """
        Extract embeddings using Wav2Vec2.
        
        Args:
            audio_segments (list): List of audio segment arrays
            
        Returns:
            ndarray: Array of embeddings
        """
        embeddings = []
        
        for segment in tqdm(audio_segments, desc="Extracting Wav2Vec2 embeddings"):
            try:
                # Process segment for Wav2Vec2
                inputs = self.wav2vec2_processor(segment, 
                                              sampling_rate=self.sample_rate, 
                                              return_tensors="pt").to(self.device)
                
                # Extract features
                with torch.no_grad():
                    outputs = self.wav2vec2_model(**inputs)
                    # Mean pooling of hidden states
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Error extracting Wav2Vec2 embedding: {e}")
                # Add a placeholder embedding
                if embeddings:
                    embeddings.append(np.zeros_like(embeddings[0]))
                else:
                    # Use a default size for Wav2Vec2 embeddings
                    embeddings.append(np.zeros(768))
        
        return np.array(embeddings)
    
    def extract_mfcc_features(self, audio_segments):
        """
        Extract MFCC features as fallback.
        
        Args:
            audio_segments (list): List of audio segment arrays
            
        Returns:
            ndarray: Array of feature vectors
        """
        features = []
        
        for segment in tqdm(audio_segments, desc="Extracting MFCC features"):
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=segment, sr=self.sample_rate, n_mfcc=20)
            
            # Compute statistics over the segment
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_delta = librosa.feature.delta(mfccs)
            delta_mean = np.mean(mfcc_delta, axis=1)
            delta_std = np.std(mfcc_delta, axis=1)
            
            # Combine features
            combined_features = np.concatenate([mfcc_mean, mfcc_std, delta_mean, delta_std])
            features.append(combined_features)
        
        return np.array(features)
    
    def extract_features(self, voice_segments):
        """
        Extract features from voice segments using the best available method.
        
        Args:
            voice_segments (list): List of (start_time, end_time) tuples
            
        Returns:
            tuple: (embeddings, segment_times, segment_audio)
        """
        # Extract audio for each segment
        segment_times = []
        segment_audio = []
        
        for start_time, end_time in voice_segments:
            # Convert time to samples
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Handle edge cases
            start_sample = max(0, start_sample)
            end_sample = min(len(self.audio), end_sample)
            
            # Skip if segment is too short (less than 0.5 seconds)
            if end_sample - start_sample < int(0.5 * self.sample_rate):
                continue
            
            # Extract segment audio
            audio_segment = self.audio[start_sample:end_sample]
            
            # Add to lists
            segment_times.append((start_time, end_time))
            segment_audio.append(audio_segment)
        
        # Check if any segments were extracted
        if not segment_audio:
            logger.warning("No valid audio segments extracted")
            return np.array([]), [], []
        
        # Try to extract embeddings with Resemblyzer
        if self.resemblyzer_available:
            try:
                logger.info("Extracting d-vector embeddings with Resemblyzer...")
                embeddings = self.extract_resemblyzer_embeddings(segment_audio)
                if len(embeddings) > 0 and not np.all(embeddings[0] == 0):
                    logger.info(f"Successfully extracted {len(embeddings)} d-vector embeddings")
                    return embeddings, segment_times, segment_audio
            except Exception as e:
                logger.warning(f"Resemblyzer extraction failed: {e}")
        
        # Try with Wav2Vec2 if Resemblyzer failed
        if self.wav2vec2_available:
            try:
                logger.info("Extracting embeddings with Wav2Vec2...")
                embeddings = self.extract_wav2vec2_embeddings(segment_audio)
                if len(embeddings) > 0 and not np.all(embeddings[0] == 0):
                    logger.info(f"Successfully extracted {len(embeddings)} Wav2Vec2 embeddings")
                    return embeddings, segment_times, segment_audio
            except Exception as e:
                logger.warning(f"Wav2Vec2 extraction failed: {e}")
        
        # Fall back to MFCC features
        logger.info("Falling back to MFCC feature extraction...")
        embeddings = self.extract_mfcc_features(segment_audio)
        logger.info(f"Extracted {len(embeddings)} MFCC feature vectors")
        
        return embeddings, segment_times, segment_audio