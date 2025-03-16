# Voice Separation Tool

A powerful speaker diarization and voice separation tool that automatically detects and isolates different speakers from audio files. This tool handles varied speaking styles, emotional states, and voice modulations from the same speaker.

## Features

- **Advanced Voice Activity Detection**: Identifies speech segments using neural and traditional methods
- **Speaker Embedding Extraction**: Uses state-of-the-art d-vectors and Wav2Vec2 embeddings
- **Automatic Speaker Detection**: Determines the optimal number of speakers without manual input
- **Voice Variation Handling**: Recognizes the same speaker across different voice characteristics
- **Acoustic Analysis**: Analyzes pitch, formants, and spectral properties for accurate speaker identification
- **Visualization**: Generates visual timelines of speaker activity

## Installation

### Prerequisites

- Python 3.7+
- ffmpeg (for audio format conversion)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/subhashdasyam/voice-separation.git
   cd voice-separation
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python main.py input_audio.mp3 --output-format mp3
```

This will:
- Detect and separate different speakers in the audio file
- Save each speaker's voice to separate files (voice1.mp3, voice2.mp3, etc.)
- Generate visualization of the diarization results

### Advanced Options

```bash
python main.py input_audio.mp3 --output-format mp3 --device cuda --skip-diarization
```

### HF Token needed

```
export HF_TOKEN=<GET READ ONLY HUGGING FACE TOKEN>
```

### Command-Line Arguments

- `input_file`: Path to input audio file (mp3 or wav)
- `--output-dir`: Directory to save separated voice files (default: 'output')
- `--output-format`: Output file format, either 'wav' or 'mp3' (default: 'wav')
- `--device`: Device to run models on ('cuda' or 'cpu')
- `--model-dir`: Directory to save/load models (default: 'models')
- `--no-visualize`: Disable visualization generation
- `--diarization-timeout`: Timeout for neural diarization in seconds (default: 300)
- `--skip-diarization`: Skip neural diarization and use clustering directly
- `--disable-refinement`: Disable cluster refinement to preserve initial speaker count
- `--min-speakers`: Minimum number of speakers to consider in clustering (default: 2)
- `--max-speakers`: Maximum number of speakers to consider in clustering (default: 8)
- `--debug`: Enable debug logging

## How It Works

1. **Voice Activity Detection**
   - Uses PyAnnote neural VAD model with WebRTC VAD as fallback
   - Identifies all speech segments in the audio

2. **Feature Extraction**
   - Extracts d-vector embeddings (voice prints) using Resemblyzer
   - Falls back to Wav2Vec2 or MFCC features if needed

3. **Speaker Diarization**
   - Uses either pretrained diarization model or custom clustering
   - Automatically determines optimal number of speakers

4. **Cluster Refinement**
   - Analyzes acoustic properties (pitch, formants) to validate speaker identity
   - Merges clusters that likely belong to the same speaker with different voice characteristics

5. **Audio Separation**
   - Separates and saves each speaker's segments as individual files
   - Adds appropriate silence between segments for natural listening

## Troubleshooting

### Common Issues

1. **Too Few Speakers Detected**:
   - Use `--disable-refinement` to prevent merging speaker clusters
   - Specify minimum speakers with `--min-speakers 4` if you know there are at least 4 speakers
   - Try `--skip-diarization` to use the custom clustering approach

2. **Diarization Takes Too Long**:
   - Use `--skip-diarization` to bypass neural diarization
   - Adjust timeout with `--diarization-timeout 180` (3 minutes)

3. **Out of Memory Errors**:
   - Use `--device cpu` to process on CPU instead of GPU

4. **Spectral Clustering Errors**:
   - Update spectralcluster library: `pip install spectralcluster --upgrade`
   - The code includes fallbacks if spectral clustering fails

## License

This project is licensed under the MIT License - see the LICENSE file for details.