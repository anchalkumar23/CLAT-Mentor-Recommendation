#!/usr/bin/env python3
"""
CLAT Mentor Recommendation System - Main Entry Point

This script coordinates the entire recommendation pipeline:
1. Data generation
2. Data preprocessing
3. Recommendation generation
4. System evaluation
5. Result visualization
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clat_recommendations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import from local modules
from data.generate_data import generate_aspirant_data, generate_mentor_data
from models.preprocessing import preprocess_data
from models.recommendation import MentorRecommender, batch_recommend
from evaluation.metrics import RecommendationEvaluator, analyze_recommendation_distributions
from utils.visualization import plot_improvement_metrics

# Configuration
CONFIG = {
    'data': {
        'num_aspirants': 100,
        'num_mentors': 20,
        'random_seed': 42
    },
    'recommendation': {
        'top_n': 3,
        'min_similarity': 0.5,
        'diversity_factor': 0.2
    },
    'evaluation': {
        'sample_size': 20,
        'track_metrics': True
    },
    'output': {
        'directory': 'results',
        'save_recommendations': True,
        'save_metrics': True,
        'save_plots': True
    }
}

def setup_output_directory(config: Dict[str, Any]) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to output directory
    """
    output_dir = Path(config['output']['directory'])
    output_dir.mkdir(exist_ok=True)
    return output_dir

def generate_and_save_data(config: Dict[str, Any], output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate and save sample data.
    
    Args:
        config: Configuration dictionary
        output_dir: Path to output directory
        
    Returns:
        Tuple of (aspirants_df, mentors_df)
    """
    logger.info("Generating sample data...")
    
    # Set random seed for reproducibility
    np.random.seed(config['data']['random_seed'])
    
    # Generate data
    aspirants = generate_aspirant_data(config['data']['num_aspirants'])
    mentors = generate_mentor_data(config['data']['num_mentors'])
    
    # Save raw data
    if config['output']['save_recommendations']:
        aspirants.to_csv(output_dir / 'aspirants_raw.csv', index=False)
        mentors.to_csv(output_dir / 'mentors_raw.csv', index=False)
        logger.info(f"Saved raw data to {output_dir}")
    
    return aspirants, mentors

def run_recommendation_pipeline(
    aspirants: pd.DataFrame,
    mentors: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run the complete recommendation pipeline.
    
    Args:
        aspirants: Aspirant data
        mentors: Mentor data
        config: Configuration dictionary
        output_dir: Path to output directory
        
    Returns:
        Dictionary containing pipeline results
    """
    results = {}
    
    # 1. Preprocess data
    logger.info("Preprocessing data...")
    aspirants_processed, mentors_processed = preprocess_data(aspirants, mentors)
    
    # 2. Initialize recommender
    logger.info("Initializing recommender...")
    recommender = MentorRecommender(mentors_processed)
    
    # 3. Generate recommendations
    logger.info("Generating recommendations...")
    recommendations = batch_recommend(
        aspirants_processed,
        mentors_processed,
        top_n=config['recommendation']['top_n'],
        min_similarity=config['recommendation']['min_similarity'],
        diversity_factor=config['recommendation']['diversity_factor']
    )
    
    # Save recommendations
    if config['output']['save_recommendations']:
        rec_path = output_dir / 'recommendations.csv'
        recommendations.to_csv(rec_path, index=False)
        logger.info(f"Saved recommendations to {rec_path}")
        results['recommendations_path'] = rec_path
    
    # 4. Evaluate recommendations
    logger.info("Evaluating recommendations...")
    evaluator = RecommendationEvaluator(aspirants_processed, mentors_processed)
    metrics = evaluator.evaluate_recommendations(
        recommendations,
        sample_size=config['evaluation']['sample_size'],
        k=config['recommendation']['top_n']
    )
    
    # Save metrics
    if config['output']['save_metrics']:
        metrics_path = output_dir / 'evaluation_metrics.json'
        pd.Series(metrics).to_json(metrics_path)
        logger.info(f"Saved evaluation metrics to {metrics_path}")
        results['metrics_path'] = metrics_path
    
    # 5. Generate visualizations
    if config['output']['save_plots']:
        logger.info("Generating visualizations...")
        
        # Plot metrics trends
        if config['evaluation']['track_metrics']:
            metrics_fig_path = output_dir / 'metrics_trends.png'
            evaluator.plot_metrics_trends(metrics_fig_path)
            results['metrics_plot_path'] = metrics_fig_path
        
        # Plot recommendation distributions
        dist_figures = analyze_recommendation_distributions(
            recommendations,
            output_dir=output_dir
        )
        results['distribution_plots'] = list(dist_figures.keys())
    
    return results

def main():
    """
    Main execution function for the recommendation system.
    """
    try:
        logger.info("Starting CLAT Mentor Recommendation System")
        
        # 1. Setup output directory
        output_dir = setup_output_directory(CONFIG)
        logger.info(f"Output will be saved to: {output_dir.resolve()}")
        
        # 2. Generate and save sample data
        aspirants, mentors = generate_and_save_data(CONFIG, output_dir)
        
        # 3. Run recommendation pipeline
        results = run_recommendation_pipeline(aspirants, mentors, CONFIG, output_dir)
        
        # 4. Display summary
        logger.info("\n=== Pipeline Execution Summary ===")
        logger.info(f"Generated recommendations for {len(aspirants)} aspirants")
        logger.info(f"Using {len(mentors)} mentors in the pool")
        
        if 'metrics_path' in results:
            metrics = pd.read_json(results['metrics_path'], typ='series')
            logger.info("\nEvaluation Metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric:25}: {value:.3f}")
        
        logger.info("\nRecommendation system completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in recommendation pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()