import json
import pandas as pd
from collections import defaultdict, Counter

class SimpleMultiFacetEvaluator:
    """
    Your multi-facet framework implementation
    Evaluates datasets across 5 facets: Data, Source, System, Task, Human
    For 3 dimensions: Alignment, Noise, Imbalance
    """

    def __init__(self, dataset_name="test"):
        self.dataset_name = dataset_name
        self.data = []
        self.results = {}

    def load_data(self, filename):
        """Load any JSON dataset"""
        try:
            with open(filename, 'r') as f:
                self.data = json.load(f)
            print(f"✅ Loaded {len(self.data)} items from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return False

    def evaluate_alignment(self):
        """Evaluate ALIGNMENT across all 5 facets"""
        print("\n🎯 Evaluating ALIGNMENT...")

        alignment_scores = {}

        # FACET 1: DATA - Is content well aligned across modalities?
        alignment_scores['data'] = self._alignment_data_facet()

        # FACET 2: SOURCE - Are aligned modalities from trusted sources?
        alignment_scores['source'] = self._alignment_source_facet()

        # FACET 3: SYSTEM - Was alignment preserved in processing pipeline?
        alignment_scores['system'] = self._alignment_system_facet()

        # FACET 4: TASK - Is alignment useful for downstream tasks?
        alignment_scores['task'] = self._alignment_task_facet()

        # FACET 5: HUMAN - Do humans confirm alignment is correct?
        alignment_scores['human'] = self._alignment_human_facet()

        return alignment_scores

    def _alignment_data_facet(self):
        """DATA: Check if different modalities are temporally/semantically aligned"""
        if not self.data:
            return {"score": 0.0, "reason": "No data to evaluate"}

        # For demonstration, check if items have multiple modalities
        multimodal_count = 0
        for item in self.data[:100]:  # Sample first 100
            modalities = 0
            if isinstance(item, dict):
                # Count different types of data in each item
                if any(key in item for key in ['image', 'img', 'video', 'vid']):
                    modalities += 1
                if any(key in item for key in ['text', 'caption', 'question', 'q']):
                    modalities += 1
                if any(key in item for key in ['audio', 'sound']):
                    modalities += 1
                if any(key in item for key in ['timestamp', 'ts', 'time']):
                    modalities += 1

                if modalities >= 2:
                    multimodal_count += 1

        score = multimodal_count / min(100, len(self.data))
        return {
            "score": score,
            "reason": f"{multimodal_count} items have multiple modalities",
            "details": f"Multimodal alignment rate: {score:.2%}"
        }

    def _alignment_source_facet(self):
        """SOURCE: Check consistency across different data sources"""
        score = 0.85  # Placeholder - would analyze source metadata
        return {
            "score": score,
            "reason": "Source consistency analysis needed",
            "details": "Would check if aligned data comes from same/trusted sources"
        }

    def _alignment_system_facet(self):
        """SYSTEM: Check if processing pipeline preserved alignment"""
        if not self.data:
            return {"score": 0.0, "reason": "No data"}

        # Check for systematic processing errors
        missing_fields = 0
        total_expected_fields = 0

        for item in self.data[:50]:  # Sample check
            if isinstance(item, dict):
                total_expected_fields += 3  # Expect at least 3 key fields
                if len(item.keys()) < 3:
                    missing_fields += 1

        if total_expected_fields == 0:
            score = 1.0
        else:
            score = max(0, 1 - (missing_fields / total_expected_fields))

        return {
            "score": score,
            "reason": f"System integrity check: {missing_fields} issues found",
            "details": f"Processing pipeline preserved {score:.2%} of expected structure"
        }

    def _alignment_task_facet(self):
        """TASK: Check if alignment helps downstream tasks"""
        score = 0.75  # Placeholder - would check task-specific relevance
        return {
            "score": score,
            "reason": "Task relevance analysis needed",
            "details": "Would measure if alignment improves task performance"
        }

    def _alignment_human_facet(self):
        """HUMAN: Human validation of alignment quality"""
        score = 0.80  # Placeholder - would require human annotation
        return {
            "score": score,
            "reason": "Human validation study needed",
            "details": "Would require manual annotation to verify alignment quality"
        }

    def evaluate_noise(self):
        """Evaluate NOISE across all 5 facets"""
        print("\n🔍 Evaluating NOISE...")

        noise_scores = {}

        # FACET 1: DATA - Detect outliers, missing data, quality issues
        noise_scores['data'] = self._noise_data_facet()

        # FACET 2: SOURCE - Check for source-specific noise patterns
        noise_scores['source'] = self._noise_source_facet()

        # FACET 3: SYSTEM - Check for pipeline-introduced noise
        noise_scores['system'] = self._noise_system_facet()

        # FACET 4: TASK - Check if noise affects task performance
        noise_scores['task'] = self._noise_task_facet()

        # FACET 5: HUMAN - Human perception of noise
        noise_scores['human'] = self._noise_human_facet()

        return noise_scores

    def _noise_data_facet(self):
        """DATA: Basic noise detection in raw data"""
        if not self.data:
            return {"score": 0.0, "reason": "No data"}

        noise_indicators = {
            'empty_items': 0,
            'malformed_items': 0,
            'duplicate_items': 0
        }

        seen_items = set()

        for item in self.data[:100]:
            # Check for empty items
            if not item or (isinstance(item, dict) and len(item) == 0):
                noise_indicators['empty_items'] += 1
                continue

            # Check for malformed items
            if isinstance(item, dict):
                if not any(isinstance(v, (str, int, float, list)) for v in item.values()):
                    noise_indicators['malformed_items'] += 1

            # Check for duplicates (simple string comparison)
            item_str = str(item)
            if item_str in seen_items:
                noise_indicators['duplicate_items'] += 1
            seen_items.add(item_str)

        total_noise = sum(noise_indicators.values())
        total_checked = min(100, len(self.data))
        noise_score = max(0, 1 - (total_noise / total_checked))

        return {
            "score": noise_score,
            "reason": f"Found {total_noise} noise indicators in {total_checked} items",
            "details": noise_indicators
        }

    def _noise_source_facet(self):
        """SOURCE: Source-specific noise analysis"""
        return {"score": 0.85, "reason": "Source noise analysis placeholder"}

    def _noise_system_facet(self):
        """SYSTEM: Pipeline noise detection"""
        return {"score": 0.90, "reason": "System noise analysis placeholder"}

    def _noise_task_facet(self):
        """TASK: Task-relevant noise detection"""
        return {"score": 0.75, "reason": "Task noise analysis placeholder"}

    def _noise_human_facet(self):
        """HUMAN: Human-perceived noise"""
        return {"score": 0.80, "reason": "Human noise study needed"}

    def evaluate_imbalance(self):
        """Evaluate IMBALANCE across all 5 facets"""
        print("\n⚖️ Evaluating IMBALANCE...")

        imbalance_scores = {}

        # FACET 1: DATA - Check for class/modality imbalances
        imbalance_scores['data'] = self._imbalance_data_facet()

        # FACET 2: SOURCE - Check for source representation imbalances
        imbalance_scores['source'] = self._imbalance_source_facet()

        # FACET 3: SYSTEM - Check for processing-induced imbalances
        imbalance_scores['system'] = self._imbalance_system_facet()

        # FACET 4: TASK - Check if imbalance affects task fairness
        imbalance_scores['task'] = self._imbalance_task_facet()

        # FACET 5: HUMAN - Human perception of fairness/balance
        imbalance_scores['human'] = self._imbalance_human_facet()

        return imbalance_scores

    def _imbalance_data_facet(self):
        """DATA: Detect class or modality imbalances"""
        if not self.data:
            return {"score": 0.0, "reason": "No data"}

        # Simple balance check - look at key distributions
        key_counts = Counter()
        for item in self.data[:100]:
            if isinstance(item, dict):
                for key in item.keys():
                    key_counts[key] += 1

        if not key_counts:
            return {"score": 0.5, "reason": "No analyzable structure"}

        # Check if any key is overly dominant
        total_keys = sum(key_counts.values())
        max_proportion = max(key_counts.values()) / total_keys

        # Good balance = no single key dominates more than 50%
        balance_score = max(0, 1 - max(0, max_proportion - 0.5) * 2)

        return {
            "score": balance_score,
            "reason": f"Key distribution analysis: max proportion {max_proportion:.2%}",
            "details": dict(key_counts.most_common(5))
        }

    def _imbalance_source_facet(self):
        """SOURCE: Source representation balance"""
        return {"score": 0.75, "reason": "Source balance analysis placeholder"}

    def _imbalance_system_facet(self):
        """SYSTEM: Processing-induced imbalances"""
        return {"score": 0.85, "reason": "System balance analysis placeholder"}

    def _imbalance_task_facet(self):
        """TASK: Task fairness implications"""
        return {"score": 0.80, "reason": "Task balance analysis placeholder"}

    def _imbalance_human_facet(self):
        """HUMAN: Human fairness perception"""
        return {"score": 0.75, "reason": "Human balance study needed"}

    def run_full_evaluation(self):
        """Run complete multi-facet quality evaluation"""
        print("=" * 60)
        print(f"🚀 MULTI-FACET QUALITY EVALUATION: {self.dataset_name}")
        print("=" * 60)

        # Run all evaluations
        alignment_results = self.evaluate_alignment()
        noise_results = self.evaluate_noise()
        imbalance_results = self.evaluate_imbalance()

        # Store results
        self.results = {
            'alignment': alignment_results,
            'noise': noise_results,
            'imbalance': imbalance_results
        }

        # Generate report
        self._generate_report()

        return self.results

    def _generate_report(self):
        """Generate comprehensive quality report"""
        print("\n" + "=" * 60)
        print("📊 QUALITY ASSESSMENT REPORT")
        print("=" * 60)

        dimensions = ['alignment', 'noise', 'imbalance']
        facets = ['data', 'source', 'system', 'task', 'human']

        # Print detailed scores
        for dim in dimensions:
            print(f"\n🎯 {dim.upper()} DIMENSION:")
            print("-" * 40)

            total_score = 0
            for facet in facets:
                result = self.results[dim][facet]
                score = result['score']
                reason = result['reason']
                print(f"  {facet.capitalize():8}: {score:.3f} - {reason}")
                total_score += score

            avg_score = total_score / len(facets)
            print(f"  {'Average':8}: {avg_score:.3f}")

        # Overall summary
        overall_scores = []
        for dim in dimensions:
            dim_avg = sum(self.results[dim][facet]['score'] for facet in facets) / len(facets)
            overall_scores.append(dim_avg)

        overall_quality = sum(overall_scores) / len(overall_scores)

        print(f"\n🏆 OVERALL QUALITY SCORE: {overall_quality:.3f}")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)

        if overall_quality >= 0.8:
            print("✅ High quality dataset - suitable for production use")
        elif overall_quality >= 0.6:
            print("⚠️ Medium quality - consider improvements before use")
        else:
            print("❌ Low quality - significant improvements needed")

        print("1. Focus on lowest-scoring facets for maximum impact")
        print("2. Implement human validation studies where needed")
        print("3. Consider domain-specific quality metrics")
        print("4. Monitor quality over time as dataset evolves")

# Demo usage
def run_demo():
    """Run a demonstration of the multi-facet evaluator"""
    print("🎉 Welcome to the Multi-Facet Quality Evaluator!")
    print("This implements your 5-facet framework for dataset quality assessment.\n")

    # Create evaluator instance
    evaluator = SimpleMultiFacetEvaluator("demo_dataset")

    # Create some demo data
    demo_data = [
        {"id": 1, "text": "A cat sits on a mat", "image": "cat.jpg", "timestamp": "0:05"},
        {"id": 2, "text": "Dog running in park", "video": "dog.mp4", "timestamp": "0:10"},
        {"id": 3, "text": "Bird flying high", "image": "bird.jpg"},
        {"id": 4, "question": "What color is the sky?", "answer": "blue", "image": "sky.jpg"},
        {"id": 5, "text": "Car driving fast", "video": "car.mp4", "audio": "engine.wav"}
    ]

    # Save demo data
    with open('demo_data.json', 'w') as f:
        json.dump(demo_data, f, indent=2)

    # Load and evaluate
    if evaluator.load_data('demo_data.json'):
        results = evaluator.run_full_evaluation()

if __name__ == "__main__":
    run_demo()
