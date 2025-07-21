# ================================================================================
# Complete MM-Score Framework with CLIP-based Semantic Alignment + Debug Imbalance
# ================================================================================

from .base import Metric
import numpy as np
from scipy.signal import correlate
from scipy.stats import pearsonr, zscore
from scipy import signal
from typing import Dict, List, Any, Tuple
import torch
import clip
from PIL import Image

class ComprehensiveAlignmentMetric(Metric):
    """
    Complete MM-Score alignment metric with CLIP-based semantic alignment and debug imbalance.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.z_threshold = 3.0

        # Component weights for overall MM-Score
        self.weights = {
            'alignment': 0.50,
            'noise': 0.30,
            'imbalance': 0.20
        }

        # CLIP model initialization
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_device = None

    def name(self):
        return "alignment"

    # ============================================================================
    # CLIP-BASED SEMANTIC ALIGNMENT
    # ============================================================================

    def load_clip_model(self):
        """Load CLIP model for semantic alignment"""
        if not hasattr(self, 'clip_model') or self.clip_model is None:
            try:
                print("[CLIP] Loading CLIP model...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                self.clip_device = device
                print(f"[CLIP] Model loaded on {device}")
            except Exception as e:
                print(f"[Error] Failed to load CLIP model: {e}")
                self.clip_model = None
                self.clip_preprocess = None

    def is_image_text_pair(self, mod1: str, mod2: str) -> bool:
        """Check if this is an image-text modality pair"""
        image_modalities = {"image", "visual", "img", "picture", "photo"}
        text_modalities = {"text", "caption", "description", "annotation", "label"}
        
        mod1_lower, mod2_lower = mod1.lower(), mod2.lower()
        
        is_image_text = (any(img in mod1_lower for img in image_modalities) and 
                        any(txt in mod2_lower for txt in text_modalities))
        is_text_image = (any(txt in mod1_lower for txt in text_modalities) and 
                        any(img in mod2_lower for img in image_modalities))
        
        return is_image_text or is_text_image

    def is_text_text_pair(self, mod1: str, mod2: str) -> bool:
        """Check if this is a text-text modality pair"""
        text_modalities = {"text", "caption", "description", "annotation", "label", "transcript", "subtitle"}
        
        mod1_lower, mod2_lower = mod1.lower(), mod2.lower()
        
        return (any(txt in mod1_lower for txt in text_modalities) and 
                any(txt in mod2_lower for txt in text_modalities))

    def extract_image_features(self, image_data: Any) -> torch.Tensor:
        """Extract CLIP features from image data"""
        try:
            # Handle different image data formats
            if isinstance(image_data, np.ndarray):
                # Single image or batch of images
                if len(image_data.shape) == 3:  # Single image (H, W, C)
                    images = [image_data]
                elif len(image_data.shape) == 4:  # Batch of images (N, H, W, C)
                    images = [image_data[i] for i in range(min(5, image_data.shape[0]))]  # Limit to 5 images
                else:
                    print("[Warning] Unexpected image data shape")
                    return None
            elif isinstance(image_data, list):
                # List of image arrays
                images = image_data[:5]  # Limit to first 5 images
            else:
                print(f"[Warning] Unsupported image data type: {type(image_data)}")
                return None
            
            # Process images through CLIP
            processed_images = []
            for img_array in images:
                try:
                    # Convert numpy array to PIL Image
                    if isinstance(img_array, np.ndarray):
                        if img_array.dtype != np.uint8:
                            img_array = (img_array * 255).astype(np.uint8)
                        pil_image = Image.fromarray(img_array)
                    else:
                        pil_image = img_array
                    
                    # Preprocess for CLIP
                    processed_img = self.clip_preprocess(pil_image).unsqueeze(0).to(self.clip_device)
                    processed_images.append(processed_img)
                    
                except Exception as e:
                    print(f"[Warning] Failed to process image: {e}")
                    continue
            
            if not processed_images:
                return None
            
            # Batch process through CLIP
            image_batch = torch.cat(processed_images, dim=0)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_batch)
            
            return image_features
            
        except Exception as e:
            print(f"[Error] Image feature extraction: {e}")
            return None

    def extract_text_features(self, text_data: Any) -> torch.Tensor:
        """Extract CLIP features from text data"""
        try:
            # Handle different text data formats
            if isinstance(text_data, str):
                texts = [text_data]
            elif isinstance(text_data, list):
                # Extract text strings from various formats
                texts = []
                for item in text_data[:10]:  # Limit to first 10 texts
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        # Try common text fields
                        for field in ['caption', 'description', 'text', 'sentence', 'question']:
                            if field in item and isinstance(item[field], str):
                                texts.append(item[field])
                                break
                        else:
                            # Fallback: convert entire dict to string
                            texts.append(str(item))
                    else:
                        texts.append(str(item))
            else:
                texts = [str(text_data)]
            
            if not texts:
                return None
            
            # Clean and limit text length
            cleaned_texts = []
            for text in texts:
                # Clean text and limit length
                clean_text = str(text).strip()[:200]  # Limit to 200 chars
                if clean_text:
                    cleaned_texts.append(clean_text)
            
            if not cleaned_texts:
                return None
            
            # Tokenize texts for CLIP
            text_tokens = clip.tokenize(cleaned_texts, truncate=True).to(self.clip_device)
            
            # Extract text features
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
            
            return text_features
            
        except Exception as e:
            print(f"[Error] Text feature extraction: {e}")
            return None

    def compute_image_text_alignment(self, data1: Any, data2: Any, mod1: str, mod2: str) -> float:
        """Compute CLIP-based image-text semantic alignment"""
        try:
            # Determine which is image and which is text
            if any(img in mod1.lower() for img in ["image", "visual", "img", "picture", "photo"]):
                image_data, text_data = data1, data2
            else:
                image_data, text_data = data2, data1
            
            # Prepare image data
            image_features = self.extract_image_features(image_data)
            if image_features is None:
                return 0.0
            
            # Prepare text data
            text_features = self.extract_text_features(text_data)
            if text_features is None:
                return 0.0
            
            # Compute CLIP similarity
            with torch.no_grad():
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # Compute similarity
                similarity = torch.matmul(image_features, text_features.T)
                
                # Average similarity score
                similarity_score = similarity.mean().item()
                
                # Convert from [-1, 1] to [0, 1]
                normalized_score = (similarity_score + 1) / 2
                
                return max(0.0, min(1.0, normalized_score))
                
        except Exception as e:
            print(f"[Error] Image-text alignment: {e}")
            return 0.0

    def compute_text_text_alignment(self, data1: Any, data2: Any, mod1: str, mod2: str) -> float:
        """Compute text-text semantic alignment using CLIP text encoder"""
        try:
            # Extract text features for both modalities
            text_features1 = self.extract_text_features(data1)
            text_features2 = self.extract_text_features(data2)
            
            if text_features1 is None or text_features2 is None:
                return 0.0
            
            # Compute similarity
            with torch.no_grad():
                # Normalize features
                text_features1 = text_features1 / text_features1.norm(dim=1, keepdim=True)
                text_features2 = text_features2 / text_features2.norm(dim=1, keepdim=True)
                
                # Compute similarity
                similarity = torch.matmul(text_features1, text_features2.T)
                
                # Average similarity score
                similarity_score = similarity.mean().item()
                
                # Convert from [-1, 1] to [0, 1]
                normalized_score = (similarity_score + 1) / 2
                
                return max(0.0, min(1.0, normalized_score))
                
        except Exception as e:
            print(f"[Error] Text-text alignment: {e}")
            return 0.0

    def compute_simple_text_similarity(self, text1: Any, text2: Any) -> float:
        """Simple text similarity using basic string methods"""
        try:
            # Convert to strings
            str1 = str(text1).lower() if not isinstance(text1, list) else ' '.join(str(x) for x in text1[:5]).lower()
            str2 = str(text2).lower() if not isinstance(text2, list) else ' '.join(str(x) for x in text2[:5]).lower()
            
            # Simple word overlap similarity
            words1 = set(str1.split())
            words2 = set(str2.split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            print(f"[Error] Simple text similarity: {e}")
            return 0.0

    def compute_fallback_semantic_alignment(self, data1: Any, data2: Any, mod1: str, mod2: str) -> float:
        """Fallback semantic alignment for modality combinations not supported by CLIP"""
        try:
            # Simple text-based similarity for unsupported combinations
            if isinstance(data1, (list, str)) and isinstance(data2, (list, str)):
                return self.compute_simple_text_similarity(data1, data2)
            else:
                # Return moderate score for unknown combinations
                return 0.5
                
        except Exception as e:
            print(f"[Error] Fallback semantic alignment: {e}")
            return 0.0

    def compute_semantic_alignment(self, data1: Any, data2: Any, mod1: str, mod2: str) -> float:
        """Real semantic alignment using CLIP for image-text pairs and other combinations"""
        try:
            # Load CLIP model if needed
            self.load_clip_model()
            
            if self.clip_model is None:
                print(f"[Warning] CLIP model not available, using fallback for {mod1}-{mod2}")
                return self.compute_fallback_semantic_alignment(data1, data2, mod1, mod2)
            
            # Determine semantic alignment type
            if self.is_image_text_pair(mod1, mod2):
                return self.compute_image_text_alignment(data1, data2, mod1, mod2)
            elif self.is_text_text_pair(mod1, mod2):
                return self.compute_text_text_alignment(data1, data2, mod1, mod2)
            else:
                # Fallback for other combinations
                return self.compute_fallback_semantic_alignment(data1, data2, mod1, mod2)
                
        except Exception as e:
            print(f"[Error] Semantic alignment {mod1}-{mod2}: {e}")
            return 0.0

    # ============================================================================
    # SENSOR-SPECIFIC TEMPORAL ALIGNMENT
    # ============================================================================

    def extract_sensor_specific_patterns(self, data: np.ndarray, sensor_type: str) -> np.ndarray:
        """Extract temporal patterns specific to each sensor type"""

        if sensor_type == "eyes":
            if len(data.shape) >= 2 and data.shape[1] >= 2:
                gaze_x_vel = np.abs(np.diff(data[:, 0])) if data.shape[0] > 1 else np.array([0])
                gaze_y_vel = np.abs(np.diff(data[:, 1])) if data.shape[0] > 1 else np.array([0])
                gaze_velocity = np.sqrt(gaze_x_vel**2 + gaze_y_vel**2)
                if data.shape[1] > 2:
                    pupil_changes = np.abs(np.diff(data[:, 2]))
                    return gaze_velocity + pupil_changes
                return gaze_velocity
            else:
                return np.abs(np.diff(data.flatten())) if len(data.flatten()) > 1 else np.array([0])

        elif sensor_type == "temperature":
            temp_data = data.flatten()
            if len(temp_data) > 1:
                temp_changes = np.abs(np.diff(temp_data))
                if len(temp_changes) >= 5:
                    return signal.savgol_filter(temp_changes, window_length=5, polyorder=1)
                return temp_changes
            return np.array([0])

        elif sensor_type == "heart":
            hr_data = data.flatten()
            if len(hr_data) > 2:
                hrv = np.abs(np.diff(hr_data))
                rate_change = np.abs(np.diff(hr_data, n=2))
                return hrv[:-1] + rate_change
            elif len(hr_data) > 1:
                return np.abs(np.diff(hr_data))
            return np.array([0])

        elif sensor_type == "emotions":
            if len(data.shape) >= 2:
                emotion_intensity = np.linalg.norm(data, axis=1)
                return np.abs(np.diff(emotion_intensity)) if len(emotion_intensity) > 1 else np.array([0])
            else:
                emotion_data = data.flatten()
                return np.abs(np.diff(emotion_data)) if len(emotion_data) > 1 else np.array([0])

        elif sensor_type == "brain":
            if len(data.shape) >= 2:
                brain_power = np.mean(np.abs(data), axis=1)
                return np.abs(np.diff(brain_power)) if len(brain_power) > 1 else np.array([0])
            else:
                brain_data = data.flatten()
                return np.abs(np.diff(brain_data)) if len(brain_data) > 1 else np.array([0])

        else:
            # Default: Use magnitude changes
            if len(data.shape) >= 2:
                diff_data = np.diff(data, axis=0)
                return np.linalg.norm(diff_data, axis=1) if len(diff_data) > 0 else np.array([0])
            else:
                flat_data = data.flatten()
                return np.abs(np.diff(flat_data)) if len(flat_data) > 1 else np.array([0])

    def cross_correlation_alignment(self, sig1: np.ndarray, sig2: np.ndarray) -> Tuple[float, int]:
        """Cross-correlation temporal alignment algorithm"""
        try:
            if len(sig1) < 2 or len(sig2) < 2:
                return 0.0, 0

            # Normalize signals
            sig1_norm = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-8)
            sig2_norm = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-8)

            # Cross-correlation
            correlation = correlate(sig1_norm, sig2_norm, mode='full')
            lags = np.arange(-len(sig2_norm) + 1, len(sig1_norm))

            # Find peak correlation within reasonable lag range
            max_lag = min(len(sig1_norm), len(sig2_norm)) // 3
            valid_indices = np.abs(lags) <= max_lag
            correlation = correlation[valid_indices]
            lags = lags[valid_indices]

            # Get best alignment
            best_idx = np.argmax(np.abs(correlation))
            best_lag = lags[best_idx]
            alignment_score = np.abs(correlation[best_idx])

            # Normalize to [0,1]
            alignment_score = min(alignment_score / len(sig1_norm), 1.0)

            return alignment_score, best_lag

        except Exception:
            return 0.0, 0

    def validate_alignment(self, sig1: np.ndarray, sig2: np.ndarray, lag: int) -> float:
        """Validate alignment using Pearson correlation"""
        try:
            # Apply lag alignment
            if lag >= 0:
                aligned_sig1 = sig1
                aligned_sig2 = sig2[lag:lag+len(sig1)] if lag < len(sig2) else sig2[-len(sig1):]
            else:
                aligned_sig1 = sig1[-lag:-lag+len(sig2)] if -lag < len(sig1) else sig1[-len(sig2):]
                aligned_sig2 = sig2

            # Ensure same length
            min_len = min(len(aligned_sig1), len(aligned_sig2))
            if min_len < 2:
                return 0.0

            aligned_sig1 = aligned_sig1[:min_len]
            aligned_sig2 = aligned_sig2[:min_len]

            # Pearson correlation
            corr, _ = pearsonr(aligned_sig1, aligned_sig2)
            return abs(corr) if not np.isnan(corr) else 0.0

        except Exception:
            return 0.0

    # ============================================================================
    # NOISE DETECTION
    # ============================================================================

    def detect_statistical_outliers(self, data: np.ndarray) -> float:
        """Detect outliers using Z-score analysis"""
        try:
            if data.size < 3:
                return 1.0

            flat_data = data.flatten()
            valid_data = flat_data[~np.isnan(flat_data)]

            if len(valid_data) == 0:
                return 0.0

            # Calculate Z-scores
            z_scores = np.abs(zscore(valid_data))
            outliers = np.sum(z_scores > self.z_threshold)
            outlier_ratio = outliers / len(valid_data)

            # Score: 1.0 = no outliers, 0.0 = all outliers
            return max(0.0, 1.0 - outlier_ratio)

        except Exception:
            return 0.5

    def detect_missing_data(self, data: Any) -> float:
        """Detect missing, null, or corrupted data"""
        try:
            if data is None:
                return 0.0

            # Handle lists
            if isinstance(data, list):
                if len(data) == 0:
                    return 0.0

                null_count = sum(1 for item in data if item is None or item == '')
                
                # Check for corrupted dictionary entries
                if len(data) > 0 and isinstance(data[0], dict):
                    corrupted = sum(1 for item in data if not isinstance(item, dict) or 
                                  len(item) == 0 or all(v in [None, '', 0] for v in item.values()))
                    total_missing = null_count + corrupted
                else:
                    total_missing = null_count
                
                missing_ratio = total_missing / len(data)
                return max(0.0, 1.0 - missing_ratio)

            # Handle numpy arrays
            elif isinstance(data, np.ndarray):
                if data.size == 0:
                    return 0.0

                nan_count = np.sum(np.isnan(data))
                inf_count = np.sum(np.isinf(data))
                total_missing = nan_count + inf_count
                missing_ratio = total_missing / data.size

                return max(0.0, 1.0 - missing_ratio)

            # Handle other types
            else:
                if hasattr(data, '__len__') and len(data) == 0:
                    return 0.0
                return 1.0

        except Exception:
            return 0.5

    def evaluate_noise(self, modalities: Dict[str, Any]) -> float:
        """Comprehensive noise evaluation"""
        overall_scores = []

        for modality_name, data in modalities.items():
            # Convert to numpy array for analysis
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    # Extract numeric values from dictionaries
                    numeric_values = []
                    for item in data:
                        for v in item.values():
                            if isinstance(v, (int, float)):
                                numeric_values.append(v)
                    analysis_data = np.array(numeric_values) if numeric_values else np.array([0])
                else:
                    analysis_data = np.array(data)
            elif isinstance(data, np.ndarray):
                analysis_data = data
            else:
                analysis_data = np.array([0])

            # Statistical outlier detection
            outlier_score = self.detect_statistical_outliers(analysis_data)

            # Missing data detection
            missing_score = self.detect_missing_data(data)

            # Combined noise score
            combined_score = 0.6 * outlier_score + 0.4 * missing_score
            overall_scores.append(combined_score)

        # Overall noise score
        return np.mean(overall_scores) if overall_scores else 0.0

    # ============================================================================
    # ENHANCED IMBALANCE DETECTION WITH DEBUG
    # ============================================================================

    def count_data_samples(self, data: Any, modality_name: str) -> int:
        """Count actual data samples with detailed debugging"""
        try:
            print(f"[DEBUG] Counting {modality_name}:")
            print(f"[DEBUG]   Data type: {type(data)}")
            
            if isinstance(data, np.ndarray):
                print(f"[DEBUG]   Array shape: {data.shape}")
                if len(data.shape) == 4:  # Batch of images (N, H, W, C)
                    count = data.shape[0]
                    print(f"[DEBUG]   Detected as image batch: {count} images")
                    return count
                elif len(data.shape) == 3:  # Single image (H, W, C)
                    count = 1
                    print(f"[DEBUG]   Detected as single image: {count} image")
                    return count
                elif len(data.shape) == 2:  # 2D array (time series, features)
                    count = data.shape[0]
                    print(f"[DEBUG]   Detected as 2D array: {count} samples")
                    return count
                elif len(data.shape) == 1:  # 1D array
                    count = len(data)
                    print(f"[DEBUG]   Detected as 1D array: {count} elements")
                    return count
                else:
                    count = data.size
                    print(f"[DEBUG]   Detected as multidimensional: {count} total elements")
                    return count
                    
            elif isinstance(data, list):
                count = len(data)
                print(f"[DEBUG]   List length: {count}")
                if count > 0:
                    print(f"[DEBUG]   First item type: {type(data[0])}")
                    if isinstance(data[0], dict):
                        print(f"[DEBUG]   First item keys: {list(data[0].keys())}")
                return count
                
            elif isinstance(data, dict):
                print(f"[DEBUG]   Dict keys: {list(data.keys())}")
                # Try to find meaningful count in dict structure
                if 'images' in data:
                    count = len(data['images']) if hasattr(data['images'], '__len__') else 1
                    print(f"[DEBUG]   Found 'images' key with {count} items")
                    return count
                elif 'annotations' in data:
                    count = len(data['annotations']) if hasattr(data['annotations'], '__len__') else 1
                    print(f"[DEBUG]   Found 'annotations' key with {count} items")
                    return count
                elif 'captions' in data:
                    count = len(data['captions']) if hasattr(data['captions'], '__len__') else 1
                    print(f"[DEBUG]   Found 'captions' key with {count} items")
                    return count
                else:
                    count = len(data)
                    print(f"[DEBUG]   Using dict length: {count}")
                    return count
            else:
                count = 1
                print(f"[DEBUG]   Unknown type, defaulting to: {count}")
                return count
                
        except Exception as e:
            print(f"[DEBUG] Error counting {modality_name}: {e}")
            return 1

    def analyze_count_ratios(self, modalities: Dict[str, Any]) -> float:
        """Enhanced count ratio analysis with comprehensive debugging"""
        try:
            print(f"\n[IMBALANCE DEBUG] Starting imbalance analysis...")
            print(f"[IMBALANCE DEBUG] Number of modalities: {len(modalities)}")
            
            modality_counts = {}

            # Count samples for each modality
            for name, data in modalities.items():
                count = self.count_data_samples(data, name)
                modality_counts[name] = count
            
            print(f"\n[IMBALANCE DEBUG] Final modality counts: {modality_counts}")

            # Filter out zero counts
            non_zero_counts = [count for count in modality_counts.values() if count > 0]
            print(f"[IMBALANCE DEBUG] Non-zero counts: {non_zero_counts}")

            if len(non_zero_counts) < 2:
                print(f"[IMBALANCE DEBUG] Less than 2 modalities with data, returning perfect balance: 1.0")
                return 1.0

            max_count = max(non_zero_counts)
            min_count = min(non_zero_counts)
            count_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"[IMBALANCE DEBUG] Max count: {max_count}")
            print(f"[IMBALANCE DEBUG] Min count: {min_count}")
            print(f"[IMBALANCE DEBUG] Ratio: {max_count}/{min_count} = {count_ratio:.2f}")

            # Enhanced scoring logic with debug
            if count_ratio <= 2.0:
                score = 1.0
                grade = "EXCELLENT"
            elif count_ratio <= 5.0:
                score = 0.8
                grade = "GOOD"
            elif count_ratio <= 10.0:
                score = 0.6
                grade = "FAIR"
            elif count_ratio <= 50.0:
                score = 0.3
                grade = "POOR"
            else:
                score = 0.1
                grade = "CRITICAL"
            
            print(f"[IMBALANCE DEBUG] Score: {score} ({grade})")
            print(f"[IMBALANCE DEBUG] Imbalance analysis complete.\n")

            return score

        except Exception as e:
            print(f"[IMBALANCE DEBUG] Error in imbalance analysis: {e}")
            return 0.5

    def evaluate_imbalance(self, modalities: Dict[str, Any]) -> float:
        """Comprehensive imbalance evaluation with debug"""
        return self.analyze_count_ratios(modalities)

    # ============================================================================
    # GRADING SYSTEM
    # ============================================================================

    def get_quality_grade(self, score: float) -> str:
        """Get quality grade for a score"""
        if score >= 0.8:
            return "EXCELLENT"
        elif score >= 0.6:
            return "GOOD"
        elif score >= 0.4:
            return "FAIR"
        elif score >= 0.2:
            return "POOR"
        else:
            return "CRITICAL"

    # ============================================================================
    # MAIN EVALUATION
    # ============================================================================

    def prepare_data_for_alignment(self, data: Any, modality_name: str) -> np.ndarray:
        """Enhanced data preparation with better sensor support"""
        try:
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    # TVQA-style data with timestamps
                    if 'ts' in data[0]:
                        timestamps = []
                        for item in data:
                            ts = item.get('ts', '0.0-1.0')
                            try:
                                start_time = float(ts.split('-')[0])
                                timestamps.append(start_time)
                            except:
                                timestamps.append(0.0)
                        return np.array(timestamps)
                    else:
                        # Extract numeric values or create sequence
                        numeric_values = []
                        for item in data:
                            for v in item.values():
                                if isinstance(v, (int, float)):
                                    numeric_values.append(v)
                        return np.array(numeric_values) if numeric_values else np.arange(len(data), dtype=float)
                else:
                    return np.array(data, dtype=float)

            elif isinstance(data, np.ndarray):
                return data

            else:
                return np.array([0.0])

        except Exception:
            return np.array([0.0])

    def compute_temporal_alignment(self, data1: Any, data2: Any, mod1: str, mod2: str) -> float:
        """Enhanced temporal alignment with sensor-specific processing"""
        try:
            # Prepare data
            prep_data1 = self.prepare_data_for_alignment(data1, mod1)
            prep_data2 = self.prepare_data_for_alignment(data2, mod2)

            # Extract temporal signatures (sensor-specific if applicable)
            sensor_types = ["eyes", "temperature", "heart", "emotions", "brain"]

            if any(sensor in mod1.lower() for sensor in sensor_types):
                temp_sig1 = self.extract_sensor_specific_patterns(prep_data1, mod1.lower())
            else:
                temp_sig1 = self.extract_temporal_signature(prep_data1)

            if any(sensor in mod2.lower() for sensor in sensor_types):
                temp_sig2 = self.extract_sensor_specific_patterns(prep_data2, mod2.lower())
            else:
                temp_sig2 = self.extract_temporal_signature(prep_data2)

            if len(temp_sig1) < 2 or len(temp_sig2) < 2:
                return 0.0

            # Cross-correlation alignment
            cross_corr_score, best_lag = self.cross_correlation_alignment(temp_sig1, temp_sig2)

            # Validation
            pearson_score = self.validate_alignment(temp_sig1, temp_sig2, best_lag)

            # Sensor-specific weighting
            sensor_weights = {
                "eyes": 0.8, "temperature": 0.4, "heart": 0.6,
                "emotions": 0.7, "brain": 0.9
            }

            weight = 0.7  # Default
            for sensor in sensor_weights:
                if sensor in mod1.lower() or sensor in mod2.lower():
                    weight = sensor_weights[sensor]
                    break

            final_score = weight * cross_corr_score + (1 - weight) * pearson_score

            return round(final_score, 3)

        except Exception:
            return 0.0

    def extract_temporal_signature(self, data: np.ndarray, method: str = "norm") -> np.ndarray:
        """Extract temporal signature from data"""
        try:
            if len(data.shape) == 1:
                return data
            elif len(data.shape) == 2:
                if method == "norm":
                    return np.linalg.norm(data, axis=1)
                elif method == "mean":
                    return np.mean(data, axis=1)
                else:
                    return np.linalg.norm(data, axis=1)
            else:
                return np.linalg.norm(data.reshape(data.shape[0], -1), axis=1)
        except Exception:
            return np.array([0.0])

    def evaluate(self, modalities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete MM-Score evaluation with CLIP-based semantic alignment and debug imbalance
        """
        
        # ========================================================================
        # 1. ALIGNMENT EVALUATION
        # ========================================================================

        alignment_scores = []
        modality_names = list(modalities.keys())

        for i, mod1 in enumerate(modality_names):
            for mod2 in modality_names[i+1:]:
                data1 = modalities[mod1]
                data2 = modalities[mod2]

                # Determine alignment type
                temporal_modalities = {
                    "audio", "video", "sensor", "eyes", "temperature",
                    "heart", "emotions", "brain", "sensor_eye", "sensor_brain",
                    "sensor_body", "sensor_heart", "sensor_temperature"
                }

                if any(mod in temporal_modalities for mod in [mod1.lower(), mod2.lower()]):
                    # Temporal alignment using cross-correlation
                    score = self.compute_temporal_alignment(data1, data2, mod1, mod2)
                    alignment_scores.append(score)
                else:
                    # Semantic alignment using CLIP
                    score = self.compute_semantic_alignment(data1, data2, mod1, mod2)
                    alignment_scores.append(score)

        overall_alignment_score = np.mean(alignment_scores) if alignment_scores else 0.0

        # ========================================================================
        # 2. NOISE EVALUATION
        # ========================================================================

        overall_noise_score = self.evaluate_noise(modalities)

        # ========================================================================
        # 3. IMBALANCE EVALUATION (WITH DEBUG)
        # ========================================================================

        overall_imbalance_score = self.evaluate_imbalance(modalities)

        # ========================================================================
        # 4. OVERALL MM-SCORE CALCULATION
        # ========================================================================

        overall_mm_score = (
            self.weights['alignment'] * overall_alignment_score +
            self.weights['noise'] * overall_noise_score +
            self.weights['imbalance'] * overall_imbalance_score
        )

        # Get grades
        overall_grade = self.get_quality_grade(overall_mm_score)
        alignment_grade = self.get_quality_grade(overall_alignment_score)
        noise_grade = self.get_quality_grade(overall_noise_score)
        imbalance_grade = self.get_quality_grade(overall_imbalance_score)

        # ========================================================================
        # 5. SIMPLE RESULTS
        # ========================================================================

        results = {
            "mm_score_summary": {
                "overall_score": round(overall_mm_score, 2),
                "overall_grade": overall_grade,
                "modalities_analyzed": len(modalities)
            },
            "component_scores": {
                "alignment": {
                    "score": round(overall_alignment_score, 2),
                    "grade": alignment_grade
                },
                "noise": {
                    "score": round(overall_noise_score, 2),
                    "grade": noise_grade
                },
                "imbalance": {
                    "score": round(overall_imbalance_score, 2),
                    "grade": imbalance_grade
                }
            }
        }

        # ========================================================================
        # 6. CLEAN DISPLAY
        # ========================================================================
        print(f"\n🎯 MM-Score Evaluation Results:")
        print(f"Modalities: {', '.join(modalities.keys())}")
        print(f"Overall Score: {results['mm_score_summary']['overall_score']} ({overall_grade})")
        print(f"\nComponent Scores:")
        print(f"  Alignment: {results['component_scores']['alignment']['score']} ({alignment_grade})")
        print(f"  Noise: {results['component_scores']['noise']['score']} ({noise_grade})")
        print(f"  Imbalance: {results['component_scores']['imbalance']['score']} ({imbalance_grade})")
        print(f"\n✅ Evaluation complete.")

        return results
