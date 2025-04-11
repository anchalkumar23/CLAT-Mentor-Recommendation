import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import ndcg_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class RecommendationEvaluator:
    """
    A class to evaluate the performance of mentor recommendations.
    Computes various metrics to assess recommendation quality.
    """
    
    def __init__(self, aspirants_df: pd.DataFrame, mentors_df: pd.DataFrame):
        """
        Initialize the evaluator with aspirant and mentor data.
        
        Args:
            aspirants_df: Processed aspirant data
            mentors_df: Processed mentor data
        """
        self.aspirants = aspirants_df
        self.mentors = mentors_df
        self.metrics_history = []
        
    def evaluate_recommendations(
        self, 
        recommendations: pd.DataFrame, 
        sample_size: Optional[int] = None,
        k: int = 3
    ) -> Dict[str, float]:
        """
        Evaluate recommendation quality using multiple metrics.
        
        Args:
            recommendations: DataFrame containing recommendations
            sample_size: Number of aspirants to evaluate (None for all)
            k: Number of top recommendations to consider
            
        Returns:
            Dictionary of metric names and values
        """
        if sample_size is not None:
            aspirant_sample = self.aspirants.sample(min(sample_size, len(self.aspirants)))
        else:
            aspirant_sample = self.aspirants
            
        metrics = {
            'precision_at_k': [],
            'recall_at_k': [],
            'ndcg_at_k': [],
            'learning_style_match_rate': [],
            'subject_match_rate': [],
            'college_match_rate': [],
            'avg_similarity': []
        }
        
        # Group recommendations by aspirant
        grouped = recommendations.groupby('aspirant_id')
        
        for aspirant_id, aspirant_recs in grouped:
            if aspirant_id not in aspirant_sample['aspirant_id'].values:
                continue
                
            aspirant = self.aspirants[self.aspirants['aspirant_id'] == aspirant_id].iloc[0]
            top_k_recs = aspirant_recs.head(k)
            
            # Calculate metrics
            metrics['precision_at_k'].append(self._precision_at_k(aspirant, top_k_recs))
            metrics['recall_at_k'].append(self._recall_at_k(aspirant, top_k_recs))
            metrics['ndcg_at_k'].append(self._ndcg_at_k(aspirant, top_k_recs))
            metrics['learning_style_match_rate'].append(
                self._learning_style_match_rate(aspirant, top_k_recs))
            metrics['subject_match_rate'].append(
                self._subject_match_rate(aspirant, top_k_recs))
            metrics['college_match_rate'].append(
                self._college_match_rate(aspirant, top_k_recs))
            metrics['avg_similarity'].append(
                top_k_recs['similarity_score'].mean())
        
        # Aggregate metrics
        results = {
            'num_aspirants': len(aspirant_sample),
            'num_recommendations': len(recommendations),
            'precision@k': np.mean(metrics['precision_at_k']),
            'recall@k': np.mean(metrics['recall_at_k']),
            'ndcg@k': np.mean(metrics['ndcg_at_k']),
            'learning_style_match_rate': np.mean(metrics['learning_style_match_rate']),
            'subject_match_rate': np.mean(metrics['subject_match_rate']),
            'college_match_rate': np.mean(metrics['college_match_rate']),
            'avg_similarity': np.mean(metrics['avg_similarity'])
        }
        
        # Store metrics for tracking over time
        self.metrics_history.append(results)
        
        return results
    
    def _precision_at_k(self, aspirant: pd.Series, recommendations: pd.DataFrame) -> float:
        """
        Calculate precision@k - fraction of relevant recommendations in top k.
        
        Args:
            aspirant: Aspirant data
            recommendations: Top k recommendations
            
        Returns:
            Precision@k score
        """
        relevant = 0
        for _, mentor in recommendations.iterrows():
            if self._is_relevant_match(aspirant, mentor):
                relevant += 1
        return relevant / len(recommendations)
    
    def _recall_at_k(self, aspirant: pd.Series, recommendations: pd.DataFrame) -> float:
        """
        Calculate recall@k - fraction of all relevant mentors found in top k.
        
        Args:
            aspirant: Aspirant data
            recommendations: Top k recommendations
            
        Returns:
            Recall@k score
        """
        # Count all potentially relevant mentors (simplified approximation)
        total_relevant = sum(
            1 for _, mentor in self.mentors.iterrows() 
            if self._is_relevant_match(aspirant, mentor)
        )
        
        if total_relevant == 0:
            return 0.0
            
        relevant_in_top_k = sum(
            1 for _, mentor in recommendations.iterrows()
            if self._is_relevant_match(aspirant, mentor)
        )
        
        return relevant_in_top_k / total_relevant
    
    def _ndcg_at_k(self, aspirant: pd.Series, recommendations: pd.DataFrame) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            aspirant: Aspirant data
            recommendations: Top k recommendations
            
        Returns:
            NDCG@k score
        """
        # Create relevance scores (binary in this case)
        true_relevance = np.zeros(len(self.mentors))
        ideal_relevance = np.zeros(len(self.mentors))
        
        for i, (_, mentor) in enumerate(self.mentors.iterrows()):
            true_relevance[i] = 1 if self._is_relevant_match(aspirant, mentor) else 0
            ideal_relevance[i] = true_relevance[i]  # For binary relevance, ideal is same
            
        # Get predicted relevance (using similarity scores)
        pred_relevance = np.zeros(len(self.mentors))
        mentor_id_to_idx = {mentor_id: i for i, mentor_id in enumerate(self.mentors['mentor_id'])}
        
        for _, rec in recommendations.iterrows():
            if rec['mentor_id'] in mentor_id_to_idx:
                pred_relevance[mentor_id_to_idx[rec['mentor_id']]] = rec['similarity_score']
        
        # Calculate NDCG
        try:
            return ndcg_score([true_relevance], [pred_relevance], k=len(recommendations))
        except:
            return 0.0
    
    def _is_relevant_match(self, aspirant: pd.Series, mentor: pd.Series) -> bool:
        """
        Determine if a mentor is relevant for an aspirant.
        
        Args:
            aspirant: Aspirant data
            mentor: Mentor data
            
        Returns:
            True if mentor is relevant for aspirant
        """
        # Check subject match
        subject_match = (
            aspirant['preferred_subject_1'] in mentor['specialization'] or
            'All-rounder' in mentor['specialization']
        )
        
        # Check learning style match
        learning_style = aspirant['learning_style'].lower().replace('/', '_')
        style_match = (
            mentor['teaching_style'].lower().replace('/', '_') == learning_style or
            mentor['teaching_style'] == 'Mixed'
        )
        
        # Check college match
        college_match = (
            mentor['college'] == aspirant['target_college_1'] or
            mentor['college'] == aspirant['target_college_2']
        )
        
        return subject_match and style_match and college_match
    
    def _learning_style_match_rate(self, aspirant: pd.Series, recommendations: pd.DataFrame) -> float:
        """
        Calculate learning style match rate in recommendations.
        
        Args:
            aspirant: Aspirant data
            recommendations: Recommended mentors
            
        Returns:
            Match rate (0-1)
        """
        if len(recommendations) == 0:
            return 0.0
            
        learning_style = aspirant['learning_style'].lower().replace('/', '_')
        matches = sum(
            1 for _, mentor in recommendations.iterrows()
            if (mentor['teaching_style'].lower().replace('/', '_') == learning_style or
                mentor['teaching_style'] == 'Mixed')
        )
        return matches / len(recommendations)
    
    def _subject_match_rate(self, aspirant: pd.Series, recommendations: pd.DataFrame) -> float:
        """
        Calculate subject specialization match rate in recommendations.
        
        Args:
            aspirant: Aspirant data
            recommendations: Recommended mentors
            
        Returns:
            Match rate (0-1)
        """
        if len(recommendations) == 0:
            return 0.0
            
        matches = sum(
            1 for _, mentor in recommendations.iterrows()
            if (aspirant['preferred_subject_1'] in mentor['specialization'] or
                'All-rounder' in mentor['specialization'])
        )
        return matches / len(recommendations)
    
    def _college_match_rate(self, aspirant: pd.Series, recommendations: pd.DataFrame) -> float:
        """
        Calculate college match rate in recommendations.
        
        Args:
            aspirant: Aspirant data
            recommendations: Recommended mentors
            
        Returns:
            Match rate (0-1)
        """
        if len(recommendations) == 0:
            return 0.0
            
        target_colleges = {aspirant['target_college_1'], aspirant['target_college_2']}
        matches = sum(
            1 for _, mentor in recommendations.iterrows()
            if mentor['college'] in target_colleges
        )
        return matches / len(recommendations)
    
    def track_metrics_over_time(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Track metrics over multiple evaluation runs.
        
        Args:
            output_path: Optional path to save results
            
        Returns:
            DataFrame with metrics history
        """
        metrics_df = pd.DataFrame(self.metrics_history)
        
        if output_path:
            metrics_df.to_csv(output_path, index=False)
        
        return metrics_df
    
    def plot_metrics_trends(self, output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot trends of metrics over time.
        
        Args:
            output_path: Optional path to save plot
            
        Returns:
            Matplotlib figure object
        """
        if not self.metrics_history:
            raise ValueError("No metrics history available")
            
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df['evaluation_run'] = range(1, len(metrics_df) + 1)
        
        # Select metrics to plot
        plot_metrics = [
            'precision@k', 'recall@k', 'ndcg@k',
            'learning_style_match_rate', 
            'subject_match_rate',
            'college_match_rate',
            'avg_similarity'
        ]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        
        # Plot each metric
        for metric in plot_metrics:
            sns.lineplot(
                data=metrics_df,
                x='evaluation_run',
                y=metric,
                label=metric.replace('@k', '').replace('_', ' ').title(),
                marker='o'
            )
        
        plt.title("Recommendation Metrics Over Time", fontsize=16)
        plt.xlabel("Evaluation Run", fontsize=12)
        plt.ylabel("Metric Value", fontsize=12)
        plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        
        return plt.gcf()


def analyze_recommendation_distributions(
    recommendations: pd.DataFrame,
    output_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Analyze distributions of recommendation features.
    
    Args:
        recommendations: DataFrame of recommendations
        output_dir: Optional directory to save plots
        
    Returns:
        Dictionary of feature names and matplotlib figures
    """
    figures = {}
    
    # 1. Similarity score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(recommendations['similarity_score'], bins=20, kde=True)
    plt.title("Distribution of Recommendation Similarity Scores")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    figures['similarity_dist'] = plt.gcf()
    if output_dir:
        plt.savefig(f"{output_dir}/similarity_dist.png")
    plt.close()
    
    # 2. Mentor specialization distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(
        y='specialization',
        data=recommendations,
        order=recommendations['specialization'].value_counts().index
    )
    plt.title("Mentor Specializations in Recommendations")
    plt.xlabel("Count")
    plt.ylabel("Specialization")
    figures['specialization_dist'] = plt.gcf()
    if output_dir:
        plt.savefig(f"{output_dir}/specialization_dist.png")
    plt.close()
    
    # 3. Teaching style distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(
        y='teaching_style',
        data=recommendations,
        order=recommendations['teaching_style'].value_counts().index
    )
    plt.title("Teaching Styles in Recommendations")
    plt.xlabel("Count")
    plt.ylabel("Teaching Style")
    figures['teaching_style_dist'] = plt.gcf()
    if output_dir:
        plt.savefig(f"{output_dir}/teaching_style_dist.png")
    plt.close()
    
    return figures


if __name__ == "__main__":
    # Test the evaluation functions
    from data.generate_data import generate_aspirant_data, generate_mentor_data
    from models.preprocessing import preprocess_data
    from models.recommendation import batch_recommend
    
    print("Generating sample data...")
    aspirants = generate_aspirant_data(50)
    mentors = generate_mentor_data(15)
    
    print("\nPreprocessing data...")
    aspirants_processed, mentors_processed = preprocess_data(aspirants, mentors)
    
    print("\nGenerating recommendations...")
    recommendations = batch_recommend(aspirants_processed, mentors_processed)
    
    print("\nInitializing evaluator...")
    evaluator = RecommendationEvaluator(aspirants_processed, mentors_processed)
    
    print("\nEvaluating recommendations...")
    metrics = evaluator.evaluate_recommendations(recommendations)
    
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    print("\nAnalyzing recommendation distributions...")
    figures = analyze_recommendation_distributions(recommendations)
    
    print("\nSaving sample plots...")
    for name, fig in figures.items():
        fig.savefig(f"{name}_sample.png")
        plt.close(fig)
    
    print("\nSample evaluation complete. Check generated plots for visualizations.")