#!/usr/bin/env python3
"""
Enhanced MM-Score Research Framework
Combines research-grade rigor with smart features for academic use.

Features:
- Dynamic per-command flexibility (while maintaining reproducibility)
- Enhanced summary + detailed results  
- Research-friendly interface with consumer-grade UX
- Full YAML configuration support + command-line overrides
- Scientific methodology with modern convenience
"""

import yaml
import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from Utils.loaders import load_embeddings         
import subprocess


# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics import get_metric
from loaders import load_modalities, smart_load_dataset

class EnhancedMMSCOREvaluator:
    """
    Enhanced MM-Score evaluator that combines research rigor with smart features.
    
    Supports both traditional YAML configuration and dynamic command-line usage.
    """

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize evaluator with config file and/or dynamic parameters.
        
        Args:
            config_path: Path to YAML configuration file (optional)
            **kwargs: Dynamic parameters that override config file
        """
        self.version = "2.0-Research"
        self.framework_name = "MM-Score Enhanced Research Framework"
        
        # Load configuration
        self.config = self._load_config(config_path, **kwargs)
        
        # Initialize components
        self.modalities = None
        self.metrics = []
        
    def _load_config(self, config_path: Optional[str], **kwargs) -> Dict[str, Any]:
        """Load configuration from file and apply dynamic overrides"""
        
        # Default research configuration
        default_config = {
            "modalities": {},
            "evaluation": {
                "metrics": ["alignment"],
                "weights": {
                    "alignment": 0.5,
                    "noise": 0.3,
                    "imbalance": 0.2
                },
                "max_items": None
            },
            "output": {
                "detailed_results": True,
                "save_config": True,
                "format": "json"
            },
            "research": {
                "reproducible": True,
                "random_seed": 42,
                "methodology_logging": True
            }
        }
        
        # Load from YAML file if provided
        if config_path and os.path.exists(config_path):
            print(f"[Config] Loading configuration from: {config_path}")
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
            
            # Merge with defaults
            config = self._deep_merge(default_config, file_config)
        else:
            config = default_config.copy()
        
        # Apply dynamic overrides from command line
        if kwargs:
            print(f"[Config] Applying dynamic overrides: {list(kwargs.keys())}")
            config = self._apply_overrides(config, kwargs)
        
        return config

    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_overrides(self, config: Dict, overrides: Dict) -> Dict:
        """Apply command-line overrides to configuration"""
        
        # Handle common override patterns
        if 'dataset_path' in overrides:
            # Dynamic dataset specification
            config['dynamic_dataset'] = overrides['dataset_path']
        
        if 'modalities' in overrides:
            # Dynamic modality specification
            config['dynamic_modalities'] = overrides['modalities']
        
        if 'max_items' in overrides:
            config['evaluation']['max_items'] = overrides['max_items']
        
        if 'output_file' in overrides:
            config['output']['file'] = overrides['output_file']
        
        # Add override tracking for reproducibility
        config['dynamic_overrides'] = overrides
        config['override_timestamp'] = datetime.now().isoformat()
        
        return config

    def evaluate(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive MM-Score evaluation with enhanced features.
        
        Args:
            verbose: Whether to show detailed progress
            
        Returns:
            Enhanced evaluation results with research metadata
        """
        
        if verbose:
            self._print_research_header()
        
        try:
            # Step 1: Data Loading (Smart or Traditional)
            if verbose:
                print("=" * 80)
                print("PHASE 1: DATA LOADING & PREPARATION")
                print("=" * 80)
            
            loaded_data = self._load_data_intelligently(verbose)
            
            if not loaded_data:
                return self._create_error_result("No data could be loaded")
            
            # Step 2: Metric Initialization
            if verbose:
                print("\n" + "=" * 80)
                print("PHASE 2: METRIC INITIALIZATION") 
                print("=" * 80)
            
            self._initialize_metrics(verbose)
            
            # Step 3: Comprehensive Evaluation
            if verbose:
                print("\n" + "=" * 80)
                print("PHASE 3: COMPREHENSIVE EVALUATION")
                print("=" * 80)
            
            evaluation_results = self._run_evaluation(loaded_data, verbose)
            
            # Step 4: Enhanced Results Compilation
            if verbose:
                print("\n" + "=" * 80)
                print("PHASE 4: RESULTS COMPILATION")
                print("=" * 80)
            
            final_results = self._compile_enhanced_results(
                loaded_data, evaluation_results, verbose
            )
            
            # Step 5: Output Generation
            self._handle_output(final_results, verbose)
            
            if verbose:
                self._print_enhanced_summary(final_results)
            
            return final_results
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            if verbose:
                print(f" {error_msg}")
                import traceback
                traceback.print_exc()
            return self._create_error_result(error_msg)

    def _load_data_intelligently(self, verbose: bool) -> Dict[str, Any]:
        """
        Load **pre‑computed embeddings** from the `embeddings/` folder
        (if present), otherwise fall back to the smart / YAML loaders.
        """
        emb_dir = Path("embeddings")
        if emb_dir.exists() and any(emb_dir.glob("*.npy")):
            if verbose:
                print(f" Loading cached embeddings from: {emb_dir}/")
            from loaders import load_embeddings                     # helper we added
            return {
                p.stem: load_embeddings(emb_dir, p.stem)
                for p in emb_dir.glob("*.npy")
            }

        # ── legacy paths (raw files) ─────────────────────────────
        if 'dynamic_dataset' in self.config:                       # CLI override
            if verbose:
                print(f" Smart loading raw data from: {self.config['dynamic_dataset']}")
            return smart_load_dataset(
                dataset_path=self.config['dynamic_dataset'],
                modalities=self.config.get('dynamic_modalities'),
                max_items=self.config['evaluation'].get('max_items')
            )
        else:                                                      # YAML
            if verbose:
                print(" Traditional raw‑data loading from YAML configuration")
            return load_modalities(self.config["modalities"])

    
    def run_preprocess(dataset: str, mods: list[str] | None, verbose=True):
        """Spawn preprocess_dataset.py with the same CLI you’d use by hand."""
        cmd = [sys.executable, "preprocess_dataset.py"]
        if dataset:
            cmd += ["--dataset", dataset]
        if mods:
            cmd += ["--modalities", *mods]
        if verbose:
            print(f"[Preprocess] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)           # raises if preprocessing fails

    def _initialize_metrics(self, verbose: bool):
        """Initialize evaluation metrics"""
        
        metric_names = self.config["evaluation"]["metrics"]
        self.metrics = [get_metric(name)(self.config) for name in metric_names]
        
        if verbose:
            print(f"Initialized {len(self.metrics)} metrics: {metric_names}")
            
            # Show research methodology
            weights = self.config["evaluation"]["weights"]
            print(f" Evaluation weights: Alignment={weights['alignment']}, "
                  f"Noise={weights['noise']}, Imbalance={weights['imbalance']}")

    def _run_evaluation(self, data: Dict[str, Any], verbose: bool) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        
        results = {}
        
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            if verbose:
                print(f" Running {metric_name}...")
            
            try:
                metric_result = metric.evaluate(data)
                results[metric_name] = metric_result
                
                if verbose:
                    score = metric_result.get("mm_score_summary", {}).get("overall_score", "N/A")
                    print(f"{metric_name}: {score}")
                    
            except Exception as e:
                if verbose:
                    print(f"{metric_name} failed: {e}")
                results[metric_name] = {"error": str(e)}
        
        return results

    def _compile_enhanced_results(self, data: Dict[str, Any], 
                                 evaluation_results: Dict[str, Any], 
                                 verbose: bool) -> Dict[str, Any]:
        """Compile enhanced results with research metadata"""
        
        if verbose:
            print("Compiling comprehensive results...")
        
        # Extract primary evaluation (assuming first metric is main one)
        primary_result = list(evaluation_results.values())[0] if evaluation_results else {}
        
        enhanced_results = {
            "success": True,
            "framework_info": {
                "name": self.framework_name,
                "version": self.version,
                "evaluation_timestamp": datetime.now().isoformat(),
                "configuration_source": "YAML + overrides" if 'dynamic_overrides' in self.config else "YAML only"
            },
            "research_metadata": {
                "methodology": {
                    "evaluation_weights": self.config["evaluation"]["weights"],
                    "metrics_used": self.config["evaluation"]["metrics"],
                    "reproducible": self.config["research"]["reproducible"],
                    "random_seed": self.config["research"]["random_seed"]
                },
                "data_info": {
                    "modalities_evaluated": list(data.keys()),
                    "sample_sizes": {name: len(content) if hasattr(content, '__len__') else 1 
                                   for name, content in data.items()},
                    "total_samples": sum(len(content) if hasattr(content, '__len__') else 1 
                                       for content in data.values())
                },
                "configuration": self.config,
                "reproducibility_info": {
                    "exact_config": self.config,
                    "override_applied": self.config.get('dynamic_overrides', {}),
                    "timestamp": self.config.get('override_timestamp')
                }
            },
            "evaluation_results": evaluation_results,
            "mm_score_summary": primary_result.get("mm_score_summary", {}),
            "quality_assessment": primary_result.get("quality_assessment", {}),
            "enhanced_summary": self._create_enhanced_summary(primary_result, data)
        }
        
        return enhanced_results

    def _create_enhanced_summary(self, primary_result: Dict, data: Dict) -> Dict[str, Any]:
        """Create enhanced summary for quick reference"""
        
        mm_summary = primary_result.get("mm_score_summary", {})
        quality = primary_result.get("quality_assessment", {})
        
        return {
            "overall_score": mm_summary.get("overall_score", 0.0),
            "overall_grade": mm_summary.get("overall_grade", "UNKNOWN"),
            "component_scores": mm_summary.get("component_scores", {}),
            "publication_readiness": quality.get("publication_readiness", "Unknown"),
            "dataset_ranking": quality.get("dataset_ranking", "Unknown"),
            "modalities_evaluated": list(data.keys()),
            "research_ready": mm_summary.get("overall_score", 0.0) >= 0.6,
            "recommended_actions": quality.get("recommendations", [])[:3]
        }

    def _handle_output(self, results: Dict[str, Any], verbose: bool):
        """Handle output generation and saving"""
        
        output_config = self.config.get("output", {})
        
        # Save to file if specified
        if output_config.get("file") or self.config.get("output", {}).get("file"):
            output_file = output_config.get("file") or self.config["output"]["file"]
            self._save_results(results, output_file, verbose)
        
        # Save configuration for reproducibility
        if output_config.get("save_config", True):
            config_file = f"mm_score_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            self._save_config(config_file, verbose)

    def _save_results(self, results: Dict[str, Any], output_file: str, verbose: bool):
        """Save results to file"""
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            if verbose:
                print(f"Results saved to: {output_file}")
                
        except Exception as e:
            if verbose:
                print(f"Failed to save results: {e}")

    def _save_config(self, config_file: str, verbose: bool):
        """Save current configuration for reproducibility"""
        try:
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            if verbose:
                print(f"Configuration saved to: {config_file}")
                
        except Exception as e:
            if verbose:
                print(f"Failed to save config: {e}")

    def _print_research_header(self):
        """Print research-grade header"""
        print("=" * 80)
        print(f" {self.framework_name} v{self.version}")
        print("=" * 80)
        print(" Research-Grade Multimodal Dataset Evaluation Framework")
        print(" Novel Cross-Correlation Temporal Alignment + Comprehensive Assessment")
        print(" Academic Publication Ready • Reproducible Research Methodology")
        print("=" * 80)

    def _print_enhanced_summary(self, results: Dict[str, Any]):
        """Print enhanced summary with research details"""
        
        if not results.get("success"):
            return
        
        summary = results.get("enhanced_summary", {})
        metadata = results.get("research_metadata", {})
        
        print("\n" + "=" * 80)
        print("ENHANCED MM-SCORE RESEARCH SUMMARY")
        print("=" * 80)
        
        # Core Results
        print(f"Overall Score: {summary.get('overall_score', 0.0):.3f} ({summary.get('overall_grade', 'UNKNOWN')})")
        print(f"Research Ready: {'YES' if summary.get('research_ready', False) else '❌ NO'}")
        print(f"Publication Status: {summary.get('publication_readiness', 'Unknown')}")
        
        # Component Breakdown
        components = summary.get('component_scores', {})
        if components:
            print(f"\nComponent Analysis:")
            for component, score in components.items():
                print(f"   {component.title()}: {score:.3f}")
        
        # Research Metadata
        methodology = metadata.get('methodology', {})
        print(f"\n Research Methodology:")
        print(f"   Evaluation Weights: {methodology.get('evaluation_weights', {})}")
        print(f"   Metrics Used: {methodology.get('metrics_used', [])}")
        print(f"   Reproducible: {methodology.get('reproducible', False)}")
        
        # Data Information
        data_info = metadata.get('data_info', {})
        print(f"\n Dataset Information:")
        print(f"   Modalities: {', '.join(data_info.get('modalities_evaluated', []))}")
        print(f"   Total Samples: {data_info.get('total_samples', 0)}")
        
        # Recommendations
        actions = summary.get('recommended_actions', [])
        if actions:
            print(f"\n Research Recommendations:")
            for i, action in enumerate(actions, 1):
                print(f"   {i}. {action}")
        
        print("\n" + "=" * 80)
        print(" Enhanced MM-Score evaluation complete! Research-grade assessment ready.")
        print("=" * 80)

    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "success": False,
            "error": error_msg,
            "framework_info": {
                "name": self.framework_name,
                "version": self.version,
                "evaluation_timestamp": datetime.now().isoformat()
            }
        }

def main():
    """Enhanced command-line interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced MM-Score Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Research Examples:
  # Traditional YAML configuration
  python mm_score_main.py --config config_mm_score.yaml
  
  # Dynamic dataset evaluation with YAML base
  python mm_score_main.py --config config_base.yaml --dataset ./new_data/ --modalities video audio
  
  # Quick research evaluation
  python mm_score_main.py --dataset ./research_data/ --modalities text image --max-items 1000
  
  # Full research pipeline with output
  python mm_score_main.py --config research_config.yaml --output research_results.json
        """
    )
    
    parser.add_argument("--config", "-c", 
                       help="YAML configuration file (optional)")
    
    parser.add_argument("--dataset", "-d",
                       help="Dataset path (overrides config)")
    
    parser.add_argument("--modalities", "-m", nargs="+",
                       help="Modalities to evaluate (overrides config)")
    
    parser.add_argument("--max-items", type=int,
                       help="Maximum items per modality (overrides config)")
    
    parser.add_argument("--output", "-o",
                       help="Output file for results (overrides config)")
    
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode - minimal output")
    
    parser.add_argument(
    "--preprocess", action="store_true",
    help="Run preprocess_dataset.py first, then evaluate")

    
    args = parser.parse_args()
    
    # Prepare dynamic overrides
    overrides = {}
    if args.dataset:
        overrides['dataset_path'] = args.dataset
    if args.modalities:
        overrides['modalities'] = args.modalities
    if args.max_items:
        overrides['max_items'] = args.max_items
    if args.output:
        overrides['output_file'] = args.output
    
    # Initialize and run evaluator
    evaluator = EnhancedMMSCOREvaluator(
        config_path=args.config,
        **overrides
    )
    
    results = evaluator.evaluate(verbose=not args.quiet)
    
    # Exit with appropriate code
    if results.get("success", False):
        print(f"\n Research evaluation completed successfully!")
        sys.exit(0)
    else:
        print(f"\n Evaluation failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
