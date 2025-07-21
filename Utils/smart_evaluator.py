#!/usr/bin/env python3
"""
Smart Multimodal Dataset Evaluator - One-Command Interface
The ultimate tool for evaluating any multimodal dataset with a single command.

Usage:
    python smart_evaluator.py --dataset ./my_dataset/ --modalities video audio
    python smart_evaluator.py --dataset https://dataset.com/data.zip --auto
    python smart_evaluator.py --dataset ./dataset/ --auto --output results.json
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loaders import smart_load_dataset
from metrics.alignment import ComprehensiveAlignmentMetric

class SmartMultimodalEvaluator:
    """
    The ultimate smart multimodal dataset evaluator that works with any dataset.
    """
    
    def __init__(self):
        self.version = "2.0"
        self.framework_name = "MM-Score Smart Evaluator"
        
    def evaluate_dataset(self, 
                        dataset_path: str, 
                        modalities: Optional[List[str]] = None,
                        max_items: Optional[int] = None,
                        output_file: Optional[str] = None,
                        verbose: bool = True) -> Dict[str, Any]:
        """
        The one-command smart evaluator that researchers will use.
        
        Args:
            dataset_path: Local path, URL, or HuggingFace dataset identifier
            modalities: Optional list of specific modalities to evaluate
            max_items: Optional limit on number of items per modality for testing
            output_file: Optional file to save results
            verbose: Whether to print detailed progress
            
        Returns:
            Comprehensive evaluation results dictionary
        """
        
        if verbose:
            self._print_header()
            print(f"🎯 Dataset: {dataset_path}")
            print(f"🔍 Modalities: {modalities or 'AUTO-DETECT'}")
            print(f"📊 Max items: {max_items or 'ALL'}")
            print()
        
        try:
            # Step 1: Smart dataset loading
            if verbose:
                print("=" * 60)
                print("PHASE 1: SMART DATASET LOADING")
                print("=" * 60)
            
            loaded_data = smart_load_dataset(
                dataset_path=dataset_path,
                modalities=modalities,
                max_items=max_items
            )
            
            if not loaded_data:
                error_msg = "❌ No data could be loaded from the dataset"
                if verbose:
                    print(error_msg)
                return {"error": error_msg, "success": False}
            
            if verbose:
                print(f"\n✅ Successfully loaded {len(loaded_data)} modalities")
                for name, data in loaded_data.items():
                    size_info = getattr(data, 'shape', len(data) if hasattr(data, '__len__') else 'N/A')
                    print(f"   📂 {name}: {type(data).__name__} - {size_info}")
            
            # Step 2: Comprehensive evaluation
            if verbose:
                print("\n" + "=" * 60)
                print("PHASE 2: COMPREHENSIVE MM-SCORE EVALUATION")
                print("=" * 60)
            
            evaluator = ComprehensiveAlignmentMetric({})
            evaluation_results = evaluator.evaluate(loaded_data)
            
            # Step 3: Enhanced results compilation
            final_results = self._compile_final_results(
                dataset_path=dataset_path,
                modalities_requested=modalities,
                loaded_data=loaded_data,
                evaluation_results=evaluation_results
            )
            
            # Step 4: Save results if requested
            if output_file:
                self._save_results(final_results, output_file, verbose)
            
            if verbose:
                self._print_summary(final_results)
            
            return final_results
            
        except Exception as e:
            error_msg = f"❌ Evaluation failed: {str(e)}"
            if verbose:
                print(error_msg)
                import traceback
                traceback.print_exc()
            return {"error": error_msg, "success": False}
    
    def _compile_final_results(self, 
                              dataset_path: str,
                              modalities_requested: Optional[List[str]],
                              loaded_data: Dict[str, Any],
                              evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile enhanced final results with metadata"""
        
        return {
            "success": True,
            "framework_info": {
                "name": self.framework_name,
                "version": self.version,
                "evaluation_timestamp": self._get_timestamp()
            },
            "dataset_info": {
                "path": dataset_path,
                "modalities_requested": modalities_requested or "AUTO-DETECT",
                "modalities_loaded": list(loaded_data.keys()),
                "modalities_count": len(loaded_data),
                "total_data_size": sum(len(data) if hasattr(data, '__len__') else 1 for data in loaded_data.values())
            },
            "evaluation_results": evaluation_results,
            "quick_summary": {
                "overall_score": evaluation_results.get("mm_score_summary", {}).get("overall_score", 0.0),
                "overall_grade": evaluation_results.get("mm_score_summary", {}).get("overall_grade", "UNKNOWN"),
                "publication_readiness": evaluation_results.get("quality_assessment", {}).get("publication_readiness", "Unknown"),
                "primary_recommendations": evaluation_results.get("quality_assessment", {}).get("recommendations", [])[:3]
            }
        }
    
    def _save_results(self, results: Dict[str, Any], output_file: str, verbose: bool = True):
        """Save results to file"""
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            if verbose:
                print(f"\n💾 Results saved to: {output_file}")
                
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to save results: {e}")
    
    def _print_header(self):
        """Print framework header"""
        print("=" * 80)
        print(f"🎯 {self.framework_name} v{self.version}")
        print("=" * 80)
        print("🚀 The Universal Smart Multimodal Dataset Evaluator")
        print("📊 Evaluates ANY multimodal dataset with a single command")
        print("🔬 Novel cross-correlation temporal alignment + comprehensive quality assessment")
        print("=" * 80)
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print concise summary"""
        if not results.get("success"):
            return
            
        summary = results.get("quick_summary", {})
        dataset_info = results.get("dataset_info", {})
        
        print("\n" + "=" * 80)
        print("🎯 SMART EVALUATION SUMMARY")
        print("=" * 80)
        print(f"📂 Dataset: {dataset_info.get('path', 'Unknown')}")
        print(f"🔍 Modalities Evaluated: {', '.join(dataset_info.get('modalities_loaded', []))}")
        print(f"📊 Overall Score: {summary.get('overall_score', 0.0)} ({summary.get('overall_grade', 'UNKNOWN')})")
        print(f"📖 Publication Status: {summary.get('publication_readiness', 'Unknown')}")
        
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(summary.get('primary_recommendations', [])[:3], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "=" * 80)
        print("✨ Smart evaluation complete! Your dataset quality assessment is ready.")
        print("=" * 80)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def evaluate_dataset(dataset_path: str, 
                    modalities: Optional[List[str]] = None,
                    max_items: Optional[int] = None,
                    output_file: Optional[str] = None,
                    verbose: bool = True) -> Dict[str, Any]:
    """
    The ultimate one-command smart multimodal dataset evaluator.
    
    Args:
        dataset_path: Local path, URL, or HuggingFace dataset identifier
        modalities: Optional list of specific modalities to evaluate 
        max_items: Optional limit on number of items per modality
        output_file: Optional file to save results
        verbose: Whether to print progress
        
    Returns:
        Comprehensive evaluation results
        
    Examples:
        # Auto-detect and evaluate all modalities
        results = evaluate_dataset("./my_dataset/")
        
        # Evaluate specific modalities
        results = evaluate_dataset("./dataset/", ["video", "audio"])
        
        # Evaluate online dataset
        results = evaluate_dataset("https://dataset.com/data.zip", ["text", "image"])
        
        # Quick test with limited data
        results = evaluate_dataset("./dataset/", max_items=100)
    """
    evaluator = SmartMultimodalEvaluator()
    return evaluator.evaluate_dataset(
        dataset_path=dataset_path,
        modalities=modalities,
        max_items=max_items,
        output_file=output_file,
        verbose=verbose
    )

def main():
    """Command-line interface for the smart evaluator"""
    parser = argparse.ArgumentParser(
        description="Smart Multimodal Dataset Evaluator - Evaluate any dataset with one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect and evaluate all modalities
  python smart_evaluator.py --dataset ./my_dataset/ --auto
  
  # Evaluate specific modalities
  python smart_evaluator.py --dataset ./dataset/ --modalities video audio
  
  # Evaluate with output file
  python smart_evaluator.py --dataset ./dataset/ --auto --output results.json
  
  # Quick test with limited data
  python smart_evaluator.py --dataset ./dataset/ --auto --max-items 100
        """
    )
    
    parser.add_argument("--dataset", "-d", required=True,
                       help="Dataset path (local directory, URL, or HuggingFace identifier)")
    
    parser.add_argument("--modalities", "-m", nargs="+",
                       help="Specific modalities to evaluate (e.g., video audio text)")
    
    parser.add_argument("--auto", "-a", action="store_true",
                       help="Auto-detect and evaluate all available modalities")
    
    parser.add_argument("--max-items", type=int,
                       help="Maximum number of items per modality (for testing)")
    
    parser.add_argument("--output", "-o",
                       help="Output file to save results (JSON format)")
    
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode - minimal output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.auto and not args.modalities:
        print("❌ Error: Must specify either --auto or --modalities")
        parser.print_help()
        sys.exit(1)
    
    # Determine modalities
    modalities = None if args.auto else args.modalities
    
    # Run evaluation
    results = evaluate_dataset(
        dataset_path=args.dataset,
        modalities=modalities,
        max_items=args.max_items,
        output_file=args.output,
        verbose=not args.quiet
    )
    
    # Exit with appropriate code
    if results.get("success", False):
        print(f"\n🎉 Evaluation completed successfully!")
        if args.output:
            print(f"📄 Detailed results saved to: {args.output}")
        sys.exit(0)
    else:
        print(f"\n❌ Evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
