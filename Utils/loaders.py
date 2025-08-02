import os
import json
import h5py
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union
import os
import json
import csv  

class UniversalDataLoader:
    """Universal loader that auto-detects and handles any data format with smart capabilities"""

    @staticmethod
    def detect_format(file_path: str) -> str:
        """Auto-detect file format from extension"""
        ext = Path(file_path).suffix.lower()
        format_map = {
            '.h5': 'h5', '.hdf5': 'h5',
            '.json': 'json', '.jsonl': 'jsonl',
            '.pt': 'pytorch', '.pth': 'pytorch',
            '.npy': 'numpy', '.npz': 'numpy',
            '.csv': 'csv', '.tsv': 'csv',
            '.txt': 'text',
            '.dat': 'dat', '.data': 'dat',
            '.wav': 'audio', '.mp3': 'audio', '.flac': 'audio', '.aac': 'audio', '.m4a': 'audio',
            '.mp4': 'video', '.avi': 'video', '.mov': 'video', '.mkv': 'video', '.wmv': 'video',
            '.jpg': 'image', '.jpeg': 'image', '.png': 'image', '.bmp': 'image', '.tiff': 'image', '.gif': 'image'
        }
        return format_map.get(ext, 'unknown')

    @staticmethod
    def load_h5_data(file_path: str) -> Any:
        """Load HDF5 files (like AVE dataset)"""
        try:
            with h5py.File(file_path, 'r') as f:
                # Auto-detect main data key
                keys = list(f.keys())
                if len(keys) == 1:
                    return f[keys[0]][:]
                else:
                    # Return dict of all datasets
                    return {key: f[key][:] for key in keys}
        except Exception as e:
            print(f"[Error] Loading H5 file {file_path}: {e}")
            return None

    @staticmethod
    def load_jsonl_data(file_path: str, max_items: int = None) -> List[Dict]:
        """Load JSONL files (like TVQA dataset)"""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_items and i >= max_items:
                        break
                    if line.strip():
                        data.append(json.loads(line.strip()))
            return data
        except Exception as e:
            print(f"[Error] Loading JSONL file {file_path}: {e}")
            return []

    @staticmethod
    def load_json_data(file_path: str) -> Any:
        """Load JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Error] Loading JSON file {file_path}: {e}")
            return None

    @staticmethod
    def load_pytorch_data(file_path: str) -> Any:
        """Load PyTorch tensor files with enhanced compatibility"""
        try:
            # Approach 1: Standard loading
            return torch.load(file_path, map_location='cpu')
        except Exception as e1:
            try:
                # Approach 2: Load with weights_only=False for older files
                return torch.load(file_path, map_location='cpu', weights_only=False)
            except Exception as e2:
                try:
                    # Approach 3: Load with pickle protocol
                    import pickle
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e3:
                    print(f"[Error] All PyTorch loading methods failed for {file_path}")
                    print(f"  Standard: {str(e1)[:100]}...")
                    print(f"  Weights_only=False: {str(e2)[:100]}...")
                    print(f"  Pickle: {str(e3)[:100]}...")
                    return None

    @staticmethod
    def load_numpy_data(file_path: str) -> Any:
        """Load NumPy files"""
        try:
            if file_path.endswith('.npz'):
                return np.load(file_path)
            else:
                return np.load(file_path)
        except Exception as e:
            print(f"[Error] Loading NumPy file {file_path}: {e}")
            return None

    @staticmethod
    def load_csv_data(file_path: str) -> pd.DataFrame:
        """Load CSV/TSV files"""
        try:
            separator = '\t' if file_path.endswith('.tsv') else ','
            return pd.read_csv(file_path, sep=separator)
        except Exception as e:
            print(f"[Error] Loading CSV file {file_path}: {e}")
            return None

    
    @staticmethod
    def load_text_data(path: Union[str, Path],
                       max_items: int = None,
                       verbose: bool = True) -> List[str]:
        """
        Robust text loader (file *or* folder).
        • Supports .txt / .json / .jsonl / .csv / .tsv
        • Recursively extracts every string, no matter how deeply nested.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[TextLoader] Path not found: {path}")

        # ------------------------------------------------------------------ #
        # 1️⃣  Collect files
        # ------------------------------------------------------------------ #
        if path.is_file():
            files = [path]
        else:
            patterns = ["*.txt", "*.json", "*.jsonl", "*.csv", "*.tsv"]
            files: List[Path] = []
            for pat in patterns:
                files.extend(path.rglob(pat))

        if not files:
            raise FileNotFoundError(f"[TextLoader] No supported text files under: {path}")

        # ------------------------------------------------------------------ #
        # 2️⃣  String extractor
        # ------------------------------------------------------------------ #
        def grab(obj):
            if isinstance(obj, str):
                return [obj.strip()] if obj.strip() else []
            if isinstance(obj, dict):
                out = []
                for v in obj.values():
                    out.extend(grab(v))
                return out
            if isinstance(obj, (list, tuple, set)):
                out = []
                for v in obj:
                    out.extend(grab(v))
                return out
            return []

        # ------------------------------------------------------------------ #
        # 3️⃣  Read files
        # ------------------------------------------------------------------ #
        texts: List[str] = []
        for fp in files:
            try:
                suf = fp.suffix.lower()

                # plain TXT -------------------------------------------------
                if suf == ".txt":
                    with fp.open("r", encoding="utf-8") as f:
                        texts.extend(line.strip() for line in f if line.strip())

                # JSON-Lines -----------------------------------------------
                elif suf == ".jsonl":
                    with fp.open("r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if max_items and len(texts) >= max_items:
                                break
                            if line.strip():
                                texts.extend(grab(json.loads(line)))

                # JSON -----------------------------------------------------
                elif suf == ".json":
                    with fp.open("r", encoding="utf-8") as f:
                        texts.extend(grab(json.load(f)))

                # CSV / TSV -----------------------------------------------
                elif suf in {".csv", ".tsv"}:
                    sep = "\t" if suf == ".tsv" else ","
                    with fp.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f, delimiter=sep)
                        for row in reader:
                            texts.extend(grab(row))

            except Exception as e:
                if verbose:
                    print(f"[TextLoader] Skipped {fp.name}: {e}")

            if max_items and len(texts) >= max_items:
                break

        if not texts:
            raise ValueError("[TextLoader] No valid text extracted.")

        if max_items:
            texts = texts[:max_items]

        if verbose:
            print(f"[TextLoader] Loaded {len(texts)} text snippets from {len(files)} file(s)")

        return texts


    @staticmethod
    def load_image_data(file_path: str) -> np.ndarray:
        """Load image files and convert to numpy arrays"""
        try:
            from PIL import Image
            
            # Load image
            with Image.open(file_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array
                img_array = np.array(img)
                
                print(f"[Loader] Loaded image: {img_array.shape} - {file_path}")
                return img_array
                
        except ImportError:
            print(f"[Error] PIL (Pillow) not installed. Please install: pip install Pillow")
            return None
        except Exception as e:
            print(f"[Error] Loading image {file_path}: {e}")
            return None

    @staticmethod
    def load_video_data(file_path: str) -> np.ndarray:
        """Load video files and extract frames"""
        try:
            import cv2
            
            # Open video file
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                print(f"[Error] Could not open video file: {file_path}")
                return None
            
            frames = []
            frame_count = 0
            max_frames = 30  # Limit to first 30 frames for memory efficiency
            
            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_count += 1
            
            cap.release()
            
            if frames:
                video_array = np.array(frames)
                print(f"[Loader] Loaded video: {video_array.shape} - {file_path}")
                return video_array
            else:
                print(f"[Error] No frames extracted from video: {file_path}")
                return None
                
        except ImportError:
            print(f"[Error] OpenCV not installed. Please install: pip install opencv-python")
            return None
        except Exception as e:
            print(f"[Error] Loading video {file_path}: {e}")
            return None

    @staticmethod
    def load_audio_data(file_path: str) -> np.ndarray:
        """Load audio files and convert to numpy arrays"""
        try:
            import librosa
            
            # Load audio file
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            
            print(f"[Loader] Loaded audio: {audio_data.shape} @ {sample_rate}Hz - {file_path}")
            return audio_data
            
        except ImportError:
            print(f"[Error] librosa not installed. Please install: pip install librosa")
            return None
        except Exception as e:
            print(f"[Error] Loading audio {file_path}: {e}")
            return None

    @staticmethod
    def load_dat_data(file_path: str) -> Any:
        """Load .dat files (like Opportunity dataset)"""
        try:
            # Approach 1: Try text-based data first (most common for sensor data)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    data_rows = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('%'):
                            # Try different separators
                            for sep in [' ', '\t', ',', ';']:
                                try:
                                    row = line.split(sep)
                                    if len(row) > 1:  # Must have multiple columns
                                        numeric_row = [float(x.strip()) for x in row if x.strip()]
                                        if len(numeric_row) > 0:
                                            data_rows.append(numeric_row)
                                            break
                                except:
                                    continue

                    if data_rows:
                        # Ensure all rows have same length (pad with zeros if needed)
                        max_len = max(len(row) for row in data_rows)
                        padded_rows = []
                        for row in data_rows:
                            if len(row) < max_len:
                                row.extend([0.0] * (max_len - len(row)))
                            padded_rows.append(row)
                        return np.array(padded_rows)
            except UnicodeDecodeError:
                pass  # File is binary, try binary approach

            # Approach 2: Binary data
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                    # Try different data types
                    for dtype in [np.float32, np.float64, np.int32, np.int16]:
                        try:
                            arr = np.frombuffer(data, dtype=dtype)
                            if len(arr) > 1:  # Must have multiple values
                                return arr
                        except:
                            continue
            except:
                pass

            # Approach 3: Pandas with multiple separators
            try:
                for sep in [' ', '\t', ',', ';']:
                    try:
                        df = pd.read_csv(file_path, sep=sep, header=None, comment='#')
                        if not df.empty and df.shape[1] > 1:
                            return df.values
                    except:
                        continue
            except:
                pass

            print(f"[Warning] Could not parse DAT file format: {file_path}")
            return None

        except Exception as e:
            print(f"[Error] Loading DAT file {file_path}: {e}")
            return None

    # ============================================================================
    # SMART CAPABILITIES
    # ============================================================================

    @staticmethod
    def download_dataset(url: str, cache_dir: str = "./cache/") -> str:
        """Download dataset from URL and cache locally"""
        import requests
        import zipfile
        import tarfile
        from urllib.parse import urlparse
        
        try:
            # Create cache directory
            os.makedirs(cache_dir, exist_ok=True)
            
            # Generate cache filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path) or "dataset"
            if not filename.endswith(('.zip', '.tar', '.gz', '.h5', '.jsonl')):
                filename += ".zip"
            
            cache_path = os.path.join(cache_dir, filename)
            extract_dir = os.path.join(cache_dir, filename.split('.')[0])
            
            # Check if already cached
            if os.path.exists(extract_dir):
                print(f"[Loader] Using cached dataset: {extract_dir}")
                return extract_dir
            
            print(f"[Loader] Downloading dataset from: {url}")
            
            # Download file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"[Loader] Downloaded to: {cache_path}")
            
            # Extract if compressed
            if cache_path.endswith('.zip'):
                with zipfile.ZipFile(cache_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    print(f"[Loader] Extracted to: {extract_dir}")
                    return extract_dir
            elif cache_path.endswith(('.tar', '.tar.gz')):
                with tarfile.open(cache_path, 'r') as tar_ref:
                    tar_ref.extractall(extract_dir)
                    print(f"[Loader] Extracted to: {extract_dir}")
                    return extract_dir
            else:
                # Single file, return as is
                return cache_path
                
        except Exception as e:
            print(f"[Error] Failed to download dataset: {e}")
            return None

    @staticmethod
    def handle_dataset_path(dataset_path: str) -> str:
        """Handle both local paths and URLs"""
        
        # Check if it's a URL
        if dataset_path.startswith(('http://', 'https://', 'ftp://')):
            return UniversalDataLoader.download_dataset(dataset_path)
        
        # Check if it's a HuggingFace dataset
        elif '/' in dataset_path and not os.path.exists(dataset_path):
            try:
                from datasets import load_dataset
                print(f"[Loader] Loading HuggingFace dataset: {dataset_path}")
                # This would require additional HuggingFace integration
                # For now, return None to indicate unsupported
                print(f"[Warning] HuggingFace datasets not yet supported: {dataset_path}")
                return None
            except ImportError:
                print(f"[Warning] HuggingFace datasets library not installed")
                return None
        
        # Local path
        elif os.path.exists(dataset_path):
            return dataset_path
        
        else:
            print(f"[Error] Dataset path not found: {dataset_path}")
            return None

    @staticmethod
    def auto_detect_modalities(dataset_path: str) -> Dict[str, List[str]]:
        """Scan dataset and auto-detect available modalities with file lists"""
        
        if not os.path.exists(dataset_path):
            print(f"[Warning] Dataset path not found for auto-detection: {dataset_path}")
            return {}
        
        detected_modalities = {}
        
        print(f"[Loader] Auto-detecting modalities in: {dataset_path}")
        
        # Scan all files recursively
        all_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if not file.startswith('.'):  # Skip hidden files
                    full_path = os.path.join(root, file)
                    all_files.append(full_path)
        
        # Group files by modality type
        video_files = []
        audio_files = []
        image_files = []
        text_files = []
        sensor_files = []
        
        for file_path in all_files:
            ext = Path(file_path).suffix.lower()
            filename = Path(file_path).name.lower()
            
            # Video files
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                video_files.append(file_path)
            
            # Audio files
            elif ext in ['.wav', '.mp3', '.flac', '.aac', '.m4a']:
                audio_files.append(file_path)
            
            # Image files
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                image_files.append(file_path)
            
            # Text files
            elif ext in ['.json', '.jsonl', '.txt', '.csv'] or 'text' in filename:
                text_files.append(file_path)
            
            # Sensor files (specific detection)
            elif ext in ['.dat', '.h5', '.npy'] or any(sensor in filename for sensor in ['sensor', 'temp', 'heart', 'brain', 'eye', 'emotion']):
                sensor_files.append(file_path)
        
        # Add detected modalities
        if video_files:
            detected_modalities['video'] = video_files
            print(f"[Loader] Detected VIDEO: {len(video_files)} files")
        
        if audio_files:
            detected_modalities['audio'] = audio_files
            print(f"[Loader] Detected AUDIO: {len(audio_files)} files")
        
        if image_files:
            detected_modalities['image'] = image_files
            print(f"[Loader] Detected IMAGE: {len(image_files)} files")
        
        if text_files:
            detected_modalities['text'] = text_files
            print(f"[Loader] Detected TEXT: {len(text_files)} files")
        
        if sensor_files:
            # Try to detect specific sensor types
            sensor_subtypes = UniversalDataLoader.detect_sensor_subtypes(sensor_files)
            detected_modalities.update(sensor_subtypes)
        
        print(f"[Loader] Auto-detection complete: {list(detected_modalities.keys())}")
        return detected_modalities

    @staticmethod
    def detect_sensor_subtypes(sensor_files: List[str]) -> Dict[str, List[str]]:
        """Detect specific sensor types from filenames and content"""
        
        sensor_types = {
            'temperature': [],
            'heart': [],
            'brain': [], 
            'eyes': [],
            'emotions': [],
            'sensor': []  # Generic sensors
        }
        
        for file_path in sensor_files:
            filename = Path(file_path).name.lower()
            
            # Temperature sensors
            if any(keyword in filename for keyword in ['temp', 'temperature', 'thermal']):
                sensor_types['temperature'].append(file_path)
            
            # Heart sensors
            elif any(keyword in filename for keyword in ['heart', 'ecg', 'hrv', 'cardiac', 'pulse']):
                sensor_types['heart'].append(file_path)
            
            # Brain sensors
            elif any(keyword in filename for keyword in ['brain', 'eeg', 'fmri', 'neural']):
                sensor_types['brain'].append(file_path)
            
            # Eye tracking
            elif any(keyword in filename for keyword in ['eye', 'gaze', 'pupil', 'fixation']):
                sensor_types['eyes'].append(file_path)
            
            # Emotion sensors
            elif any(keyword in filename for keyword in ['emotion', 'facial', 'sentiment', 'affect']):
                sensor_types['emotions'].append(file_path)
            
            # Generic sensor
            else:
                sensor_types['sensor'].append(file_path)
        
        # Only return non-empty sensor types
        return {k: v for k, v in sensor_types.items() if v}

    @classmethod
    def load_file(cls, file_path: str, format_hint: str = None) -> Any:
        """Universal file loader - auto-detects format and loads appropriately"""
        if not os.path.exists(file_path):
            print(f"[Warning] File not found: {file_path}")
            return None

        # Use format hint if provided, otherwise auto-detect
        file_format = format_hint or cls.detect_format(file_path)

        print(f"[Loader] Loading {file_format} file: {file_path}")

        loader_map = {
            'h5': cls.load_h5_data,
            'json': cls.load_json_data,
            'jsonl': cls.load_jsonl_data,
            'pytorch': cls.load_pytorch_data,
            'numpy': cls.load_numpy_data,
            'csv': cls.load_csv_data,
            'text': cls.load_text_data,
            'dat': cls.load_dat_data,
            'image': cls.load_image_data,
            'video': cls.load_video_data,
            'audio': cls.load_audio_data
        }

        loader = loader_map.get(file_format)
        if loader:
            return loader(file_path)
        else:
            print(f"[Warning] Unsupported format: {file_format}")
            return None

def load_modalities(modality_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Universal modality loader that handles any dataset configuration
    
    Args:
        modality_config: Dict with modality specifications
        
    Returns:
        Dict mapping modality names to loaded data
    """
    loader = UniversalDataLoader()
    loaded_modalities = {}

    for modality_name, config in modality_config.items():
        try:
            file_path = config.get('path')
            format_hint = config.get('format')
            max_items = config.get('max_items')  # For large datasets

            if not file_path:
                print(f"[Warning] No path specified for modality: {modality_name}")
                continue

            # Handle directory paths (multiple files)
            if os.path.isdir(file_path):
                print(f"[Loader] Loading directory: {file_path}")
                files = [os.path.join(file_path, f) for f in os.listdir(file_path)
                        if not f.startswith('.')]
                loaded_modalities[modality_name] = files[:max_items] if max_items else files

            # Handle single file paths
            else:
                data = loader.load_file(file_path, format_hint)
                if data is not None:
                    # Apply max_items limit for large datasets
                    if max_items and hasattr(data, '__len__'):
                        if isinstance(data, (list, np.ndarray)) and len(data) > max_items:
                            data = data[:max_items]

                    loaded_modalities[modality_name] = data
                    print(f"[Loader] ✅ Successfully loaded {modality_name}: {type(data)}")

                    # Print shape/size info for debugging
                    if hasattr(data, 'shape'):
                        print(f"[Loader]   Shape: {data.shape}")
                    elif hasattr(data, '__len__'):
                        print(f"[Loader]   Length: {len(data)}")
                else:
                    print(f"[Loader] ❌ Failed to load {modality_name}")

        except Exception as e:
            print(f"[Error] Loading modality {modality_name}: {e}")
            continue

    return loaded_modalities

def smart_load_dataset(dataset_path: str, modalities: List[str] = None, max_items: int = None) -> Dict[str, Any]:
    """
    Smart dataset loader that handles URLs, auto-detection, and user specifications
    
    Args:
        dataset_path: Local path, URL, or HuggingFace dataset identifier
        modalities: Optional list of specific modalities to load
        max_items: Optional limit on number of items per modality
        
    Returns:
        Dict mapping modality names to loaded data
    """
    
    print(f"[Smart Loader] Starting smart dataset loading...")
    print(f"[Smart Loader] Dataset: {dataset_path}")
    print(f"[Smart Loader] Requested modalities: {modalities or 'AUTO-DETECT'}")
    
    # Step 1: Handle dataset path (URL downloads, etc.)
    local_path = UniversalDataLoader.handle_dataset_path(dataset_path)
    if not local_path:
        print(f"[Error] Could not access dataset: {dataset_path}")
        return {}
    
    # Step 2: Auto-detect available modalities
    available_modalities = UniversalDataLoader.auto_detect_modalities(local_path)
    
    if not available_modalities:
        print(f"[Warning] No modalities detected in dataset")
        return {}
    
    # Step 3: Determine which modalities to load
    if modalities is None:
        # AUTO MODE: Load all detected modalities
        target_modalities = list(available_modalities.keys())
        print(f"[Smart Loader] AUTO MODE: Loading all detected modalities: {target_modalities}")
    else:
        # USER-SPECIFIED MODE: Load only requested modalities
        target_modalities = []
        for requested in modalities:
            if requested in available_modalities:
                target_modalities.append(requested)
            else:
                print(f"[Warning] Requested modality '{requested}' not found in dataset")
        
        if not target_modalities:
            print(f"[Error] None of the requested modalities found in dataset")
            return {}
        
        print(f"[Smart Loader] USER MODE: Loading requested modalities: {target_modalities}")
    
    # Step 4: Load the selected modalities
    loaded_data = {}
    loader = UniversalDataLoader()
    
      
    for modality_name in target_modalities:
        files = available_modalities[modality_name]
        print(f"[Smart Loader] Loading {modality_name}: {len(files)} files")
    
        if len(files) == 1:
            # ---------- single file ----------
            data = loader.load_file(files[0])
            if data is not None:
                if max_items and hasattr(data, '__len__') and len(data) > max_items:
                    if isinstance(data, (list, np.ndarray)):
                        data = data[:max_items]
                loaded_data[modality_name] = data
                print(f"[Smart Loader] ✅ Loaded {modality_name}: "
                      f"{type(data)} – {getattr(data, 'shape', len(data))}")
        else:
            # ---------- multiple files ----------
            imgs = []
            for fp in files:
                if max_items and len(imgs) >= max_items:
                    break
                arr = loader.load_file(fp, format_hint="image")  # force image loader
                if arr is not None:
                    imgs.append(arr)
    
            if imgs:
                data = np.stack(imgs)          # (N,H,W,3)
                loaded_data[modality_name] = data
                print(f"[Smart Loader] ✅ Loaded {modality_name}: {data.shape}")
            else:
                print(f"[Smart Loader] ❌ Failed to load any {modality_name} files")
    
    print(f"[Smart Loader] Smart loading complete: {list(loaded_data.keys())}")
    return loaded_data



# ────────────────────────────────────────────────────────────
# Helper for MM‑SCORE pre‑computed features
# Place this at the bottom of loader.py

def load_embeddings(emb_dir: Union[str, Path], modality: str):
    """
    Load the single .npy file produced by the pre‑processor.
    Returns a dict: {"ids": [...], "emb": ndarray (N, D)}
    """
    from pathlib import Path
    import numpy as np

    file_path = Path(emb_dir) / f"{modality}.npy"
    return np.load(file_path, allow_pickle=True).item()

