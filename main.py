#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for voice separation tool.
"""

import os
import sys
import argparse
import logging
import time
import traceback
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voice_activity import VoiceActivityDetector
from src.feature_extraction import FeatureExtractor
from src.diarization import SpeakerDiarizer
from src.clustering import SpeakerClusterer
from src.audio_utils import AudioProcessor
from src.visualization import DiarizationVisualizer

def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level
    )
    return logging.getLogger("voice-separation")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Separate different speakers in an audio file."
    )
    parser.add_argument(
        "input_file", 
        help="Path to input audio file (mp3 or wav)"
    )
    parser.add_argument(
        "--output-dir", 
        default="output",
        help="Directory to save separated voice files"
    )
    parser.add_argument(
        "--output-format", 
        choices=["wav", "mp3"], 
        default="wav",
        help="Output file format"
    )
    parser.add_argument(
        "--device", 
        choices=["cuda", "cpu"], 
        default=None,
        help="Device to run models on"
    )
    parser.add_argument(
        "--model-dir", 
        default="models",
        help="Directory to save/load models"
    )
    parser.add_argument(
        "--no-visualize", 
        action="store_true",
        help="Disable visualization generation"
    )
    parser.add_argument(
        "--diarization-timeout",
        type=int,
        default=300,  # 5 minutes default
        help="Timeout for diarization in seconds"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--skip-diarization",
        action="store_true",
        help="Skip pre-trained diarization and use clustering directly"
    )
    parser.add_argument(
        "--disable-refinement",
        action="store_true",
        help="Disable cluster refinement to preserve initial speaker count"
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=2,
        help="Minimum number of speakers to consider in clustering"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=8,
        help="Maximum number of speakers to consider in clustering"
    )
    return parser.parse_args()

def process_audio(args, logger):
    """Process audio file to separate speakers."""
    start_time = time.time()
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create AudioProcessor instance
    logger.info(f"Loading audio file: {args.input_file}")
    audio_processor = AudioProcessor(args.input_file)
    
    # Step 1: Voice Activity Detection
    logger.info("Detecting voice activity...")
    vad = VoiceActivityDetector(audio_processor.audio, audio_processor.sample_rate, device=args.device)
    voice_segments = vad.detect_voice_activity()
    logger.info(f"Detected {len(voice_segments)} voice segments")
    
    # Check if any voice activity was detected
    if not voice_segments:
        logger.warning("No voice segments detected. Exiting.")
        return 0
    
    # Step 2: Feature Extraction
    logger.info("Extracting speaker features...")
    feature_extractor = FeatureExtractor(
        audio_processor.audio, 
        audio_processor.sample_rate,
        device=args.device,
        model_dir=args.model_dir
    )
    embeddings, segment_times, segment_audio = feature_extractor.extract_features(voice_segments)
    
    # Check if features were extracted successfully
    if embeddings.shape[0] == 0:
        logger.warning("Could not extract features. Exiting.")
        return 0
    
    # Step 3: Speaker Diarization (if available and not skipped)
    if not args.skip_diarization:
        try:
            logger.info("Attempting speaker diarization with pre-trained model...")
            diarizer = SpeakerDiarizer(
                args.input_file, 
                model_dir=args.model_dir,
                timeout=args.diarization_timeout
            )
            speaker_timeline = diarizer.perform_diarization()
            logger.info(f"Diarization successful, found {len(speaker_timeline)} speakers")
            
            # Save speaker audio based on diarization
            for speaker_id, segments in speaker_timeline.items():
                logger.info(f"Processing speaker {speaker_id}...")
                speaker_audio = audio_processor.extract_segments(segments)
                
                # Convert speaker_id to an integer index if needed
                if isinstance(speaker_id, str) and '_' in speaker_id:
                    try:
                        idx = int(speaker_id.split('_')[1])
                    except (IndexError, ValueError):
                        idx = list(speaker_timeline.keys()).index(speaker_id) + 1
                else:
                    idx = list(speaker_timeline.keys()).index(speaker_id) + 1
                
                output_path = os.path.join(args.output_dir, f"voice{idx}.{args.output_format}")
                audio_processor.save_audio(speaker_audio, output_path)
                logger.info(f"Saved to {output_path}")
            
            # Create visualization if requested
            if not args.no_visualize:
                logger.info("Creating visualization...")
                visualizer = DiarizationVisualizer(args.output_dir)
                visualizer.plot_diarization(speaker_timeline, audio_processor.get_duration())
            
            logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds")
            return len(speaker_timeline)
            
        except Exception as e:
            logger.warning(f"Pre-trained diarization failed: {str(e)}")
            logger.info("Falling back to custom clustering approach")
    else:
        logger.info("Skipping pre-trained diarization as requested")
    
    # Step 4: Speaker Clustering
    logger.info("Clustering speakers...")
    clusterer = SpeakerClusterer(embeddings)
    num_speakers, speaker_labels = clusterer.cluster()
    logger.info(f"Initial clustering found {num_speakers} speakers")
    
    # Step 5: Refine clustering (unless disabled)
    if args.disable_refinement:
        logger.info("Cluster refinement disabled, using initial clustering results")
        refined_labels = speaker_labels
        num_refined_speakers = num_speakers
    else:
        logger.info("Refining speaker clusters...")
        refined_labels = clusterer.refine_clusters(
            speaker_labels, embeddings, segment_times, segment_audio, 
            audio_processor.sample_rate
        )
        num_refined_speakers = len(set(refined_labels))
        logger.info(f"Refined to {num_refined_speakers} speakers")
    
    # Check if we have too few speakers after refinement
    if num_refined_speakers < 2 and len(embeddings) >= 10:
        logger.warning("Too few speakers detected after refinement, reverting to initial clustering")
        refined_labels = speaker_labels
        num_refined_speakers = num_speakers
    
    # Step 6: Save separated audio
    logger.info("Saving separated audio files...")
    audio_processor.save_separated_speakers(
        refined_labels, segment_times, segment_audio, 
        args.output_dir, args.output_format
    )
    
    # Step 7: Create visualization if requested
    if not args.no_visualize:
        logger.info("Creating visualization...")
        visualizer = DiarizationVisualizer(args.output_dir)
        visualizer.plot_segments(refined_labels, segment_times, audio_processor.get_duration())
        
        # Also visualize the embedding space if we have enough speakers
        if num_refined_speakers >= 2:
            visualizer.plot_embedding_space(embeddings, refined_labels)
    
    logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds")
    return num_refined_speakers

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.debug)
    
    # Print startup info
    logger.info("Voice Separation Tool")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Output format: {args.output_format}")
    
    # Process the audio file
    start_time = time.time()
    try:
        num_speakers = process_audio(args, logger)
        logger.info(f"Processing complete! Found {num_speakers} speakers")
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
        return 0
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        logger.info("Partial results may be available in the output directory")
        return 1
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        if args.debug:
            logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())