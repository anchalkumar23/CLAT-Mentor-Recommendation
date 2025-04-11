"""
Visualization utilities for CLAT Mentor Recommendation System
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from evaluation.feedback import FeedbackSystem 

def plot_mentor_distributions(mentors_df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distributions of mentor characteristics.
    
    Args:
        mentors_df: Processed mentor data
        output_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    # 1. Specialization distribution
    plt.subplot(2, 2, 1)
    sns.countplot(
        y='specialization',
        data=mentors_df,
        order=mentors_df['specialization'].value_counts().index
    )
    plt.title("Mentor Specializations")
    plt.xlabel("Count")
    
    # 2. College distribution
    plt.subplot(2, 2, 2)
    sns.countplot(
        y='college',
        data=mentors_df,
        order=mentors_df['college'].value_counts().index
    )
    plt.title("Mentor Colleges")
    plt.xlabel("Count")
    
    # 3. Expertise distribution
    plt.subplot(2, 2, 3)
    expertise_cols = [col for col in mentors_df.columns if col.endswith('_expertise')]
    expertise_data = mentors_df[expertise_cols].melt(var_name='Subject', value_name='Expertise')
    expertise_data['Subject'] = expertise_data['Subject'].str.replace('_expertise', '')
    sns.boxplot(x='Expertise', y='Subject', data=expertise_data)
    plt.title("Expertise Distribution by Subject")
    
    # 4. CLAT score vs rank
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='clat_rank', y='clat_score', data=mentors_df)
    plt.title("CLAT Score vs Rank")
    plt.gca().invert_xaxis()  # Lower rank is better
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    return plt.gcf()

def plot_aspirant_distributions(aspirants_df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distributions of aspirant characteristics.
    
    Args:
        aspirants_df: Processed aspirant data
        output_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(15, 10))
    sns.set_style("whitegrid")
    
    # 1. Preparation level
    plt.subplot(2, 2, 1)
    sns.countplot(x='preparation_level', data=aspirants_df)
    plt.title("Preparation Level Distribution")
    
    # 2. Learning styles
    plt.subplot(2, 2, 2)
    sns.countplot(y='learning_style', data=aspirants_df)
    plt.title("Learning Style Distribution")
    
    # 3. Target colleges
    plt.subplot(2, 2, 3)
    college_cols = ['target_college_1', 'target_college_2']
    college_data = aspirants_df[college_cols].melt(var_name='College_Type', value_name='College')
    sns.countplot(
        y='College',
        data=college_data,
        order=college_data['College'].value_counts().index
    )
    plt.title("Target College Preferences")
    
    # 4. Study hours
    plt.subplot(2, 2, 4)
    sns.histplot(x='hours_per_week', data=aspirants_df, bins=10)
    plt.title("Weekly Study Hours Distribution")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    return plt.gcf()

def plot_recommendation_metrics(metrics_df: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot recommendation evaluation metrics over time.
    
    Args:
        metrics_df: DataFrame containing metrics over time
        output_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Select metrics to plot
    plot_metrics = [
        'precision@k', 'recall@k', 'ndcg@k',
        'learning_style_match_rate',
        'subject_match_rate',
        'college_match_rate'
    ]
    
    # Plot each metric
    for metric in plot_metrics:
        sns.lineplot(
            data=metrics_df,
            x='evaluation_run',
            y=metric,
            label=metric.replace('@k', '').replace('_', ' ').title(),
            marker='o'
        )
    
    plt.title("Recommendation Metrics Over Time", fontsize=14)
    plt.xlabel("Evaluation Run", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    return plt.gcf()

def plot_feedback_analysis(feedback_system: 'FeedbackSystem', output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot analysis of user feedback data.
    
    Args:
        feedback_system: Initialized FeedbackSystem
        output_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Get top mentors
    top_mentors = feedback_system.get_popular_mentors(10)
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Create subplots
    plt.subplot(2, 1, 1)
    sns.barplot(
        x='avg_rating',
        y='name',
        data=top_mentors,
        hue='specialization',
        dodge=False
    )
    plt.title("Top Rated Mentors by Specialization")
    plt.xlabel("Average Rating")
    plt.ylabel("Mentor")
    
    plt.subplot(2, 1, 2)
    sns.scatterplot(
        x='positive_feedback',
        y='avg_rating',
        data=top_mentors,
        hue='college',
        size='total_impressions',
        sizes=(50, 200)
    )
    plt.title("Feedback Analysis: Ratings vs Selections")
    plt.xlabel("Number of Times Selected")
    plt.ylabel("Average Rating")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    return plt.gcf()

def plot_improvement_metrics(improvement_data: List[Dict], output_path: Optional[str] = None) -> plt.Figure:
    """
    Plot system improvement metrics over iterations.
    
    Args:
        improvement_data: List of metric dictionaries
        output_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if not improvement_data:
        raise ValueError("No improvement data provided")
    
    metrics_df = pd.DataFrame(improvement_data)
    metrics_df['iteration'] = range(1, len(metrics_df) + 1)
    
    return plot_recommendation_metrics(metrics_df, output_path)