#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speaker clustering module.
"""

import numpy as np
import logging
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import librosa
import sklearn

logger = logging.getLogger("voice-separation.clustering")

class SpeakerClusterer:
    """Class for speaker clustering."""
    
    def __init__(self, embeddings):
        """
        Initialize speaker clusterer.
        
        Args:
            embeddings (ndarray): Speaker embeddings
        """
        self.embeddings = embeddings
        self.scaler = StandardScaler()
        
        # Log scikit-learn version
        logger.debug(f"scikit-learn version: {sklearn.__version__}")
        
        # Scale embeddings if there's sufficient data
        if len(embeddings) > 1:
            self.scaled_embeddings = self.scaler.fit_transform(embeddings)
        else:
            self.scaled_embeddings = embeddings
            
        logger.debug(f"Initialized clusterer with {len(embeddings)} embeddings of shape {embeddings.shape}")
        
        # Check if spectralcluster is available
        try:
            import spectralcluster
            self.spectral_available = True
            # Get version to adjust parameters accordingly
            self.spectral_version = spectralcluster.__version__
            logger.debug(f"Using spectralcluster version {self.spectral_version}")
        except (ImportError, AttributeError):
            self.spectral_available = False
            self.spectral_version = None
            logger.debug("spectralcluster not available. Will use sklearn clustering instead.")
    
    def _estimate_num_speakers(self, min_speakers=1, max_speakers=10):
        """
        Estimate the optimal number of speakers using silhouette score.
        
        Args:
            min_speakers (int): Minimum number of speakers
            max_speakers (int): Maximum number of speakers
            
        Returns:
            int: Estimated number of speakers
        """
        # Handle edge cases
        if len(self.scaled_embeddings) <= 1:
            return 1
            
        # Cannot have more clusters than samples
        max_speakers = min(max_speakers, len(self.scaled_embeddings) - 1)
        
        # If we have very few segments, limit the search
        if len(self.scaled_embeddings) < 5:
            return min(len(self.scaled_embeddings), 2)
        
        # If we have many segments, increase the minimum number of speakers
        if len(self.scaled_embeddings) > 20:
            min_speakers = max(min_speakers, 2)
            
        # Adjust max speakers based on segment count - rule of thumb
        if len(self.scaled_embeddings) > 30:
            # More segments typically means more potential speakers
            suggested_max = min(max_speakers, max(4, len(self.scaled_embeddings) // 10))
            max_speakers = suggested_max
            
        logger.debug(f"Estimating optimal speakers between {min_speakers} and {max_speakers}")
        
        # Try different numbers of clusters and compute silhouette score
        best_score = -1
        best_n_speakers = max(1, min_speakers)
        
        # Must have at least 2 clusters for silhouette score
        if min_speakers < 2:
            min_speakers = 2
        
        for n_speakers in range(min_speakers, max_speakers + 1):
            # Skip if we don't have enough samples
            if n_speakers >= len(self.scaled_embeddings):
                continue
                
            # Perform clustering - use compatible parameters for AgglomerativeClustering
            try:
                clusterer = AgglomerativeClustering(
                    n_clusters=n_speakers,
                    linkage='ward'
                )
                
                cluster_labels = clusterer.fit_predict(self.scaled_embeddings)
                
                # Compute silhouette score
                score = silhouette_score(self.scaled_embeddings, cluster_labels)
                
                logger.debug(f"Silhouette score for {n_speakers} speakers: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_n_speakers = n_speakers
                
            except Exception as e:
                logger.warning(f"Error during clustering with {n_speakers} speakers: {e}")
        
        logger.info(f"Estimated optimal number of speakers: {best_n_speakers} (score: {best_score:.4f})")
        return best_n_speakers
    
    def _cluster_with_spectral(self, n_speakers):
        """
        Perform spectral clustering.
        
        Args:
            n_speakers (int): Number of speakers
            
        Returns:
            ndarray: Speaker labels
        """
        try:
            from spectralcluster import SpectralClusterer
            
            # Configure spectral clusterer with version-appropriate parameters
            if self.spectral_version:
                clusterer_params = {
                    'min_clusters': n_speakers,
                    'max_clusters': n_speakers
                }
                
                # Add parameters based on spectralcluster version
                try:
                    # Try with newer parameters
                    clusterer = SpectralClusterer(
                        **clusterer_params,
                        p_percentile=0.95,
                        gaussian_blur_sigma=1
                    )
                except TypeError:
                    # Fall back to basic parameters for older versions
                    logger.debug("Using basic parameters for spectral clustering")
                    clusterer = SpectralClusterer(**clusterer_params)
            else:
                # Default with basic parameters
                clusterer = SpectralClusterer(
                    min_clusters=n_speakers,
                    max_clusters=n_speakers
                )
            
            labels = clusterer.predict(self.embeddings)
            return n_speakers, labels
            
        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}")
            return None, None
    
    def _cluster_with_gmm(self, n_speakers):
        """
        Perform clustering with Gaussian Mixture Model.
        
        Args:
            n_speakers (int): Number of speakers
            
        Returns:
            ndarray: Speaker labels
        """
        try:
            gmm = GaussianMixture(
                n_components=n_speakers,
                covariance_type='full',
                random_state=42,
                max_iter=100
            )
            
            labels = gmm.fit_predict(self.scaled_embeddings)
            return n_speakers, labels
            
        except Exception as e:
            logger.warning(f"GMM clustering failed: {e}")
            return None, None
    
    def _cluster_with_agglomerative(self, n_speakers=None):
        """
        Perform agglomerative clustering.
        
        Args:
            n_speakers (int, optional): Number of speakers
            
        Returns:
            tuple: (n_speakers, speaker_labels)
        """
        try:
            if n_speakers is None:
                # For older scikit-learn versions, we'll fall back to a fixed number of clusters
                # We can't use distance_threshold in older versions
                # Try to estimate a reasonable number
                logger.warning("Using fixed number of clusters rather than distance_threshold")
                n_speakers = min(5, max(2, len(self.scaled_embeddings) // 10))
                
            logger.debug(f"Using AgglomerativeClustering with n_clusters={n_speakers}, linkage='ward'")
            clusterer = AgglomerativeClustering(
                n_clusters=n_speakers,
                linkage='ward'
            )
            
            labels = clusterer.fit_predict(self.scaled_embeddings)
            actual_n_speakers = len(set(labels))
            
            return actual_n_speakers, labels
            
        except Exception as e:
            logger.warning(f"Agglomerative clustering failed: {e}")
            # Fall back to simpler approach
            if n_speakers is None:
                n_speakers = 2
            
            # Very simple fallback with KMeans
            logger.debug(f"Falling back to KMeans with n_clusters={n_speakers}")
            kmeans = KMeans(n_clusters=n_speakers, random_state=42)
            labels = kmeans.fit_predict(self.scaled_embeddings)
            
            return n_speakers, labels
    
    def cluster(self):
        """
        Perform speaker clustering.
        
        Returns:
            tuple: (n_speakers, speaker_labels)
        """
        # Handle edge cases
        if len(self.embeddings) <= 1:
            return 1, np.zeros(len(self.embeddings), dtype=int)
        
        # Estimate the number of speakers
        n_speakers = self._estimate_num_speakers(min_speakers=2, max_speakers=8)
        
        # Try different clustering methods
        
        # 1. Try spectral clustering if available
        if self.spectral_available:
            logger.info("Trying spectral clustering...")
            n_spk, labels = self._cluster_with_spectral(n_speakers)
            if labels is not None:
                return n_spk, labels
        
        # 2. Try GMM
        logger.info("Trying Gaussian Mixture Model clustering...")
        n_spk, labels = self._cluster_with_gmm(n_speakers)
        if labels is not None:
            return n_spk, labels
        
        # 3. Fall back to agglomerative clustering
        logger.info("Using agglomerative clustering...")
        return self._cluster_with_agglomerative(n_speakers)
    
    def _extract_pitch_features(self, audio_segments, sample_rate):
        """
        Extract pitch features from audio segments.
        
        Args:
            audio_segments (list): Audio segments
            sample_rate (int): Sample rate
            
        Returns:
            ndarray: Pitch features
        """
        pitch_features = []
        
        for audio in audio_segments:
            # Extract pitch (F0) using librosa
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
            
            # Find pitch with highest magnitude for each frame
            pitch = []
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch.append(pitches[index, i])
            
            # Filter zeros and get statistics
            pitch = np.array([p for p in pitch if p > 0])
            
            if len(pitch) > 0:
                pitch_stats = [np.mean(pitch), np.std(pitch), np.median(pitch)]
            else:
                pitch_stats = [0, 0, 0]
            
            pitch_features.append(pitch_stats)
        
        return np.array(pitch_features)
    
    def _extract_formant_features(self, audio_segments, sample_rate):
        """
        Extract approximate formant features from audio segments.
        
        Args:
            audio_segments (list): Audio segments
            sample_rate (int): Sample rate
            
        Returns:
            ndarray: Formant features
        """
        formant_features = []
        
        for audio in audio_segments:
            # Simple formant approximation using spectral envelope
            S = np.abs(librosa.stft(audio))
            # Get the spectral envelope (mean across frames)
            formant_approx = np.mean(S, axis=1)[:20]  # First 20 frequency bins
            formant_features.append(formant_approx)
        
        return np.array(formant_features)
    
    def refine_clusters(self, speaker_labels, embeddings, segment_times, audio_segments, sample_rate):
        """
        Refine speaker clusters by analyzing voice characteristics.
        
        Args:
            speaker_labels (ndarray): Initial speaker labels
            embeddings (ndarray): Speaker embeddings
            segment_times (list): Segment times
            audio_segments (list): Audio segments
            sample_rate (int): Sample rate
            
        Returns:
            ndarray: Refined speaker labels
        """
        # Check if refinement is needed
        num_speakers = len(set(speaker_labels))
        if num_speakers <= 1 or len(embeddings) < 3:
            return speaker_labels
        
        logger.info(f"Refining {num_speakers} speaker clusters...")
        
        # Extract acoustic features
        pitch_features = self._extract_pitch_features(audio_segments, sample_rate)
        formant_features = self._extract_formant_features(audio_segments, sample_rate)
        
        # Normalize acoustic features
        if pitch_features.shape[0] > 0 and np.sum(pitch_features) > 0:
            pitch_scaler = StandardScaler()
            pitch_features = pitch_scaler.fit_transform(pitch_features)
        
        if formant_features.shape[0] > 0 and np.sum(formant_features) > 0:
            formant_scaler = StandardScaler()
            formant_features = formant_scaler.fit_transform(formant_features)
        
        # Calculate embedding similarity matrix
        similarity_matrix = 1 - cdist(embeddings, embeddings, metric='cosine')
        
        # Analyze within-speaker and between-speaker similarities
        within_similarities = []
        between_similarities = []
        
        for i in range(len(speaker_labels)):
            for j in range(i + 1, len(speaker_labels)):
                sim = similarity_matrix[i, j]
                if speaker_labels[i] == speaker_labels[j]:
                    within_similarities.append(sim)
                else:
                    between_similarities.append(sim)
        
        # Calculate thresholds from distributions
        if within_similarities:
            within_threshold = np.percentile(within_similarities, 10)  # 10th percentile
        else:
            within_threshold = 0.7  # Default
        
        if between_similarities:
            between_threshold = np.percentile(between_similarities, 95)  # Increased from 90 to 95 percentile
        else:
            between_threshold = 0.5  # Increased from 0.3 to 0.5
        
        logger.debug(f"Within-speaker similarity threshold: {within_threshold:.4f}")
        logger.debug(f"Between-speaker similarity threshold: {between_threshold:.4f}")
        
        # Find potential merge candidates
        merge_candidates = []
        
        for spk1 in range(num_speakers):
            for spk2 in range(spk1 + 1, num_speakers):
                # Get segments for each speaker
                spk1_indices = np.where(speaker_labels == spk1)[0]
                spk2_indices = np.where(speaker_labels == spk2)[0]
                
                if len(spk1_indices) == 0 or len(spk2_indices) == 0:
                    continue
                
                # Calculate cross-similarities
                cross_similarities = []
                for i in spk1_indices:
                    for j in spk2_indices:
                        cross_similarities.append(similarity_matrix[i, j])
                
                avg_similarity = np.mean(cross_similarities)
                max_similarity = np.max(cross_similarities)
                
                # Make merging stricter - both conditions must be true
                if avg_similarity > between_threshold and max_similarity > 0.85: # Increased from 0.8 to 0.85
                    merge_candidates.append((spk1, spk2, avg_similarity))
                    logger.debug(f"Potential merge: speakers {spk1} and {spk2} (sim: {avg_similarity:.4f})")
        
        # Validate merge candidates with acoustic features
        final_labels = speaker_labels.copy()
        
        # Limit the number of merges to prevent over-merging
        max_merges = max(1, num_speakers // 4)  # At most merge 25% of speakers
        merge_count = 0
        
        for spk1, spk2, similarity in sorted(merge_candidates, key=lambda x: x[2], reverse=True):
            # Check if we've reached the merge limit
            if merge_count >= max_merges:
                logger.info(f"Reached maximum allowed merges ({max_merges})")
                break
                
            # Check if these clusters still exist
            if spk1 not in final_labels or spk2 not in final_labels:
                continue
            
            # Get indices for each speaker
            spk1_indices = np.where(final_labels == spk1)[0]
            spk2_indices = np.where(final_labels == spk2)[0]
            
            # Compare pitch distributions if available
            if len(pitch_features) > 0 and np.sum(pitch_features) > 0:
                spk1_pitch = pitch_features[spk1_indices]
                spk2_pitch = pitch_features[spk2_indices]
                
                pitch_distance = np.mean(cdist(spk1_pitch, spk2_pitch, metric='euclidean'))
                
                # If pitch difference is large, likely different speakers - stricter threshold
                if pitch_distance > 3.0:  # Increased from 1.5 to 3.0
                    logger.debug(f"Rejecting merge due to pitch distance: {pitch_distance:.4f}")
                    continue
            
            # Verify with formants if available
            if len(formant_features) > 0 and np.sum(formant_features) > 0:
                spk1_formants = formant_features[spk1_indices]
                spk2_formants = formant_features[spk2_indices]
                
                formant_distance = np.mean(cdist(spk1_formants, spk2_formants, metric='euclidean'))
                
                # If formant difference is large, likely different speakers - stricter threshold
                if formant_distance > 3.5:  # Increased from 2.0 to 3.5
                    logger.debug(f"Rejecting merge due to formant distance: {formant_distance:.4f}")
                    continue
            
            # If we get here, merge the clusters
            logger.info(f"Merging speakers {spk1} and {spk2}")
            final_labels[final_labels == spk2] = spk1
            
            # Increment merge count
            merge_count += 1
        
        # Re-index labels to be consecutive integers
        unique_labels = np.unique(final_labels)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        final_labels = np.array([label_map[label] for label in final_labels])
        
        num_final_speakers = len(set(final_labels))
        logger.info(f"Refinement complete: {num_speakers} initial speakers \u2192 {num_final_speakers} final speakers")
        
        return final_labels