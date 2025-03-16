#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization module for speaker diarization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logger = logging.getLogger("voice-separation.visualization")

class DiarizationVisualizer:
    """Class for visualizing speaker diarization results."""
    
    def __init__(self, output_dir):
        """
        Initialize visualizer.
        
        Args:
            output_dir (str): Output directory for visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_segments(self, speaker_labels, segment_times, audio_duration):
        """
        Plot speaker segments timeline.
        
        Args:
            speaker_labels (ndarray): Speaker label for each segment
            segment_times (list): List of segment times as (start_time, end_time)
            audio_duration (float): Audio duration in seconds
        """
        logger.info("Creating speaker segments visualization...")
        
        # Create a figure
        plt.figure(figsize=(14, 6))
        
        # Set colors for speakers
        num_speakers = len(set(speaker_labels))
        colors = plt.cm.get_cmap('tab10', num_speakers)
        
        # Plot each segment
        for i, (label, (start, end)) in enumerate(zip(speaker_labels, segment_times)):
            plt.barh(y=label, width=end-start, left=start, height=0.8, 
                     color=colors(label), alpha=0.8)
        
        # Add labels and title
        plt.yticks(range(num_speakers), 
                  [f"Speaker {i+1}" for i in range(num_speakers)])
        plt.xlabel("Time (seconds)")
        plt.ylabel("Speaker")
        plt.title("Speaker Diarization")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Set time axis limits
        plt.xlim(0, audio_duration)
        
        # Add legend for segments
        segment_count = len(segment_times)
        plt.figtext(0.02, 0.02, f"Total segments: {segment_count}", fontsize=9)
        
        # Save the figure
        output_path = self.output_dir / "speaker_segments.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
    
    def plot_diarization(self, speaker_timeline, audio_duration):
        """
        Plot diarization results from timeline.
        
        Args:
            speaker_timeline (dict): Dictionary of speaker_id -> list of (start_time, end_time)
            audio_duration (float): Audio duration in seconds
        """
        logger.info("Creating diarization visualization...")
        
        # Create a figure
        plt.figure(figsize=(14, 6))
        
        # Set colors for speakers
        num_speakers = len(speaker_timeline)
        colors = plt.cm.get_cmap('tab10', num_speakers)
        
        # Sort speaker IDs for consistent display
        sorted_speakers = sorted(speaker_timeline.keys())
        
        # Plot each speaker's segments
        for i, speaker_id in enumerate(sorted_speakers):
            segments = speaker_timeline[speaker_id]
            
            # Extract speaker number from ID (e.g., "speaker_1" -> 1)
            if isinstance(speaker_id, str) and '_' in speaker_id:
                try:
                    speaker_num = int(speaker_id.split('_')[1])
                except (IndexError, ValueError):
                    speaker_num = i + 1
            else:
                speaker_num = i + 1
            
            # Plot each segment for this speaker
            for start, end in segments:
                plt.barh(y=i, width=end-start, left=start, height=0.8, 
                         color=colors(i), alpha=0.8)
        
        # Add labels and title
        plt.yticks(range(num_speakers), 
                  [f"Speaker {i+1}" for i in range(num_speakers)])
        plt.xlabel("Time (seconds)")
        plt.ylabel("Speaker")
        plt.title("Speaker Diarization")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Set time axis limits
        plt.xlim(0, audio_duration)
        
        # Count total segments
        total_segments = sum(len(segments) for segments in speaker_timeline.values())
        plt.figtext(0.02, 0.02, f"Total segments: {total_segments}", fontsize=9)
        
        # Save the figure
        output_path = self.output_dir / "diarization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
    
    def plot_embedding_space(self, embeddings, speaker_labels):
        """
        Plot speaker embeddings in 2D space using t-SNE.
        
        Args:
            embeddings (ndarray): Speaker embeddings
            speaker_labels (ndarray): Speaker label for each embedding
        """
        if len(embeddings) < 3:
            logger.warning("Not enough embeddings to visualize")
            return
        
        try:
            from sklearn.manifold import TSNE
            
            logger.info("Creating speaker embedding space visualization...")
            
            # Reduce dimensions with t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create a figure
            plt.figure(figsize=(10, 8))
            
            # Set colors for speakers
            num_speakers = len(set(speaker_labels))
            colors = plt.cm.get_cmap('tab10', num_speakers)
            
            # Plot each speaker's embeddings
            for speaker_id in range(num_speakers):
                indices = np.where(speaker_labels == speaker_id)[0]
                plt.scatter(
                    embeddings_2d[indices, 0],
                    embeddings_2d[indices, 1],
                    c=[colors(speaker_id)],
                    label=f"Speaker {speaker_id+1}",
                    alpha=0.7,
                    s=80
                )
            
            plt.legend()
            plt.title("Speaker Embedding Space (t-SNE)")
            plt.grid(linestyle='--', alpha=0.3)
            
            # Save the figure
            output_path = self.output_dir / "embedding_space.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Embedding space visualization saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not create embedding space visualization: {e}")