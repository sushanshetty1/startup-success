# Research Paper Analysis: Startup Success Prediction Model Evaluation
# Detailed Statistical Analysis and Research Findings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ResearchAnalysis:
    """
    Research-grade analysis for startup success prediction
    Includes statistical tests, detailed performance metrics, and research insights
    """
    
    def __init__(self, predictor_results):
        """Initialize with results from main predictor"""
        self.results = predictor_results
        self.models = list(predictor_results.keys())
        
    def statistical_significance_test(self):
        """Perform statistical significance tests between models"""
        print("Statistical Significance Analysis")
        print("="*50)
        
        # Extract CV scores for all models
        cv_scores = {}
        for model in self.models:
            cv_scores[model] = self.results[model]['cv_scores']
        
        # Perform pairwise t-tests
        print("Pairwise t-test results (p-values):")
        print("-" * 40)
        
        significance_matrix = np.zeros((len(self.models), len(self.models)))
        
        for i, model1 in enumerate(self.models):
            for j, model2 in enumerate(self.models):
                if i != j:
                    statistic, p_value = stats.ttest_rel(cv_scores[model1], cv_scores[model2])
                    significance_matrix[i, j] = p_value
                    print(f"{model1} vs {model2}: p = {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
        
        return significance_matrix
    
    def performance_stability_analysis(self):
        """Analyze the stability and consistency of model performance"""
        print("\nPerformance Stability Analysis")
        print("="*50)
        
        stability_metrics = {}
        
        for model in self.models:
            cv_scores = self.results[model]['cv_scores']
            stability_metrics[model] = {
                'mean': np.mean(cv_scores),
                'std': np.std(cv_scores),
                'min': np.min(cv_scores),
                'max': np.max(cv_scores),
                'range': np.max(cv_scores) - np.min(cv_scores),
                'cv_coefficient': np.std(cv_scores) / np.mean(cv_scores),  # Lower is more stable
                'consistency_score': 1 - (np.std(cv_scores) / np.mean(cv_scores))  # Higher is more consistent
            }
        
        # Create stability comparison
        stability_df = pd.DataFrame(stability_metrics).T
        print("\nModel Stability Metrics:")
        print(stability_df.round(4))
        
        # Identify most stable model
        most_stable = max(stability_metrics.keys(), 
                         key=lambda x: stability_metrics[x]['consistency_score'])
        print(f"\nMost Stable Model: {most_stable}")
        print(f"Consistency Score: {stability_metrics[most_stable]['consistency_score']:.4f}")
        
        return stability_metrics
    
    def effect_size_analysis(self):
        """Calculate effect sizes between models using Cohen's d"""
        print("\nEffect Size Analysis (Cohen's d)")
        print("="*50)
        
        def cohens_d(x, y):
            """Calculate Cohen's d effect size"""
            nx, ny = len(x), len(y)
            dof = nx + ny - 2
            pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
            return (np.mean(x) - np.mean(y)) / pooled_std
        
        # Get best model
        best_model = max(self.models, key=lambda x: self.results[x]['cv_mean'])
        best_scores = self.results[best_model]['cv_scores']
        
        print(f"Effect sizes compared to best model ({best_model}):")
        print("-" * 40)
        
        for model in self.models:
            if model != best_model:
                effect_size = cohens_d(best_scores, self.results[model]['cv_scores'])
                magnitude = "negligible" if abs(effect_size) < 0.2 else \
                           "small" if abs(effect_size) < 0.5 else \
                           "medium" if abs(effect_size) < 0.8 else "large"
                print(f"{model:25s}: d = {effect_size:6.3f} ({magnitude})")
    
    def create_research_visualization(self):
        """Create publication-ready visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Research Analysis: Startup Success Prediction Models', fontsize=16, fontweight='bold')
        
        # 1. Box plot of CV scores
        ax1 = axes[0, 0]
        cv_data = []
        labels = []
        for model in self.models:
            cv_data.append(self.results[model]['cv_scores'])
            labels.append(model)
        
        box_plot = ax1.boxplot(cv_data, labels=labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.models)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_title('Cross-Validation Score Distribution', fontweight='bold')
        ax1.set_ylabel('ROC-AUC Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance comparison with error bars
        ax2 = axes[0, 1]
        means = [self.results[model]['cv_mean'] for model in self.models]
        stds = [self.results[model]['cv_std'] for model in self.models]
        
        bars = ax2.bar(self.models, means, yerr=stds, capsize=5, 
                      color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Mean Performance with Standard Deviation', fontweight='bold')
        ax2.set_ylabel('ROC-AUC Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add significance annotations
        max_height = max([bar.get_height() + std for bar, std in zip(bars, stds)])
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Learning curves simulation
        ax3 = axes[1, 0]
        for i, model in enumerate(self.models):
            # Simulate learning curve based on CV scores
            cv_scores = self.results[model]['cv_scores']
            simulated_curve = np.cumsum(cv_scores) / np.arange(1, len(cv_scores) + 1)
            ax3.plot(range(1, 11), simulated_curve, marker='o', 
                    label=model, color=colors[i], linewidth=2)
        
        ax3.set_title('Learning Convergence (Cumulative CV Performance)', fontweight='bold')
        ax3.set_xlabel('CV Fold Number')
        ax3.set_ylabel('Cumulative ROC-AUC')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance metrics heatmap
        ax4 = axes[1, 1]
        metrics_data = []
        metric_names = ['CV Mean', 'CV Std', 'Test Acc', 'Test AUC']
        
        for model in self.models:
            metrics_data.append([
                self.results[model]['cv_mean'],
                self.results[model]['cv_std'],
                self.results[model]['test_accuracy'],
                self.results[model]['test_auc']
            ])
        
        metrics_df = pd.DataFrame(metrics_data, index=self.models, columns=metric_names)
        
        # Normalize for heatmap
        metrics_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
        
        sns.heatmap(metrics_normalized, annot=metrics_df.values, fmt='.3f', 
                   cmap='RdYlBu_r', ax=ax4, cbar_kws={'label': 'Normalized Score'})
        ax4.set_title('Performance Metrics Heatmap', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('research_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_research_summary(self):
        """Generate a research summary suitable for publication"""
        print("\n" + "="*80)
        print("RESEARCH SUMMARY: AI-POWERED STARTUP SUCCESS PREDICTION")
        print("="*80)
        
        # Model performance ranking
        performance_ranking = sorted(self.models, 
                                   key=lambda x: self.results[x]['cv_mean'], 
                                   reverse=True)
        
        print("\n1. MODEL PERFORMANCE RANKING:")
        print("-" * 40)
        for i, model in enumerate(performance_ranking, 1):
            result = self.results[model]
            print(f"{i}. {model}")
            print(f"   CV ROC-AUC: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
            print(f"   Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"   Test ROC-AUC: {result['test_auc']:.4f}")
            print()
        
        # Statistical insights
        best_model = performance_ranking[0]
        best_performance = self.results[best_model]['cv_mean']
        
        print("2. KEY FINDINGS:")
        print("-" * 40)
        print(f"• Best performing model: {best_model}")
        print(f"• Achieved ROC-AUC of {best_performance:.4f} in 10-fold cross-validation")
        print(f"• Performance improvement over baseline: {(best_performance - 0.5) / 0.5 * 100:.1f}%")
        
        # Performance gaps
        performance_gaps = []
        for i in range(len(performance_ranking) - 1):
            gap = self.results[performance_ranking[i]]['cv_mean'] - \
                  self.results[performance_ranking[i+1]]['cv_mean']
            performance_gaps.append(gap)
        
        print(f"• Average performance gap between models: {np.mean(performance_gaps):.4f}")
        print(f"• Maximum performance gap: {max(performance_gaps):.4f}")
        
        print("\n3. RESEARCH IMPLICATIONS:")
        print("-" * 40)
        if best_performance > 0.8:
            print("• High predictive accuracy achieved - suitable for practical deployment")
        elif best_performance > 0.7:
            print("• Good predictive accuracy - requires additional feature engineering")
        else:
            print("• Moderate predictive accuracy - consider alternative approaches")
        
        print("• 10-fold cross-validation ensures robust and generalizable results")
        print("• Multiple model comparison provides comprehensive evaluation")
        
        return {
            'best_model': best_model,
            'best_performance': best_performance,
            'performance_ranking': performance_ranking,
            'performance_gaps': performance_gaps
        }

def create_research_report(predictor_results):
    """Create a comprehensive research analysis report"""
    research = ResearchAnalysis(predictor_results)
    
    # Perform all analyses
    significance_matrix = research.statistical_significance_test()
    stability_metrics = research.performance_stability_analysis()
    research.effect_size_analysis()
    research.create_research_visualization()
    summary = research.generate_research_summary()
    
    return {
        'significance_matrix': significance_matrix,
        'stability_metrics': stability_metrics,
        'summary': summary
    }
