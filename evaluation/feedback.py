"""
Feedback system for CLAT Mentor Recommendations

Handles user feedback collection and system improvement.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from data.config import SUBJECT_MAPPING

class FeedbackSystem:
    """
    System to collect and incorporate user feedback into recommendations.
    """
    
    def __init__(self, mentors_df: pd.DataFrame):
        """
        Initialize with mentor data.
        
        Args:
            mentors_df: DataFrame of mentor information
        """
        self.mentors = mentors_df.copy()
        self.feedback_log = []
        self.mentor_stats = self._initialize_mentor_stats()
    
    def _initialize_mentor_stats(self) -> Dict[str, Dict]:
        """Initialize statistics tracking for each mentor"""
        return {
            mentor_id: {
                'positive_feedback': 0,
                'negative_feedback': 0,
                'total_impressions': 0,
                'subject_feedback': {subject: 0 for subject in SUBJECT_MAPPING},
                'avg_rating': 0,
                'rating_count': 0
            }
            for mentor_id in self.mentors['mentor_id']
        }
    
    def record_feedback(
        self,
        aspirant_id: str,
        mentor_id: str,
        rating: Optional[int] = None,
        selected: bool = False,
        feedback_comments: Optional[str] = None
    ) -> None:
        """
        Record user feedback on a mentor recommendation.
        
        Args:
            aspirant_id: ID of the aspirant providing feedback
            mentor_id: ID of the mentor being rated
            rating: Numeric rating (1-5)
            selected: Whether the mentor was selected
            feedback_comments: Optional text feedback
        """
        feedback_entry = {
            'aspirant_id': aspirant_id,
            'mentor_id': mentor_id,
            'timestamp': pd.Timestamp.now(),
            'rating': rating,
            'selected': selected,
            'feedback_comments': feedback_comments
        }
        
        self.feedback_log.append(feedback_entry)
        
        # Update mentor statistics
        if mentor_id in self.mentor_stats:
            stats = self.mentor_stats[mentor_id]
            stats['total_impressions'] += 1
            
            if selected:
                stats['positive_feedback'] += 1
            
            if rating is not None:
                # Update average rating
                stats['rating_count'] += 1
                stats['avg_rating'] = (
                    (stats['avg_rating'] * (stats['rating_count'] - 1) + rating) / 
                    stats['rating_count']
                )
    
    def get_mentor_feedback_stats(self, mentor_id: str) -> Dict:
        """
        Get feedback statistics for a specific mentor.
        
        Args:
            mentor_id: ID of the mentor
            
        Returns:
            Dictionary of feedback statistics
        """
        return self.mentor_stats.get(mentor_id, {})
    
    def get_popular_mentors(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get the most popular mentors based on feedback.
        
        Args:
            top_n: Number of top mentors to return
            
        Returns:
            DataFrame of top mentors with feedback stats
        """
        mentor_ids = sorted(
            self.mentor_stats.keys(),
            key=lambda x: (
                self.mentor_stats[x]['positive_feedback'],
                self.mentor_stats[x]['avg_rating']
            ),
            reverse=True
        )[:top_n]
        
        stats = [self.mentor_stats[mentor_id] for mentor_id in mentor_ids]
        result = pd.DataFrame(stats)
        result['mentor_id'] = mentor_ids
        result = result.merge(self.mentors, on='mentor_id')
        
        return result[
            ['mentor_id', 'name', 'specialization', 'college', 
             'positive_feedback', 'avg_rating', 'total_impressions']
        ]
    
    def adjust_mentor_expertise(self, adjustment_factor: float = 0.1) -> pd.DataFrame:
        """
        Adjust mentor expertise based on feedback.
        
        Args:
            adjustment_factor: How much to adjust scores
            
        Returns:
            Updated mentor DataFrame
        """
        updated_mentors = self.mentors.copy()
        
        for mentor_id, stats in self.mentor_stats.items():
            if stats['rating_count'] == 0:
                continue
                
            # Calculate adjustment based on average rating
            rating_adjustment = (stats['avg_rating'] - 3) * adjustment_factor
            
            # Apply to all expertise fields
            for subject in SUBJECT_MAPPING.values():
                col = f"{subject}_expertise"
                if col in updated_mentors.columns:
                    # Cap expertise between 0 and 10
                    updated_mentors.loc[
                        updated_mentors['mentor_id'] == mentor_id, col
                    ] = np.clip(
                        updated_mentors[col] + rating_adjustment,
                        0,
                        10
                    )
        
        return updated_mentors

def simulate_feedback(recommendations: pd.DataFrame, feedback_system: FeedbackSystem) -> None:
    """
    Simulate user feedback for testing purposes.
    
    Args:
        recommendations: DataFrame of recommendations
        feedback_system: FeedbackSystem instance
    """
    for _, row in recommendations.iterrows():
        # Simulate some users providing feedback
        if np.random.random() > 0.7:  # 30% chance of feedback
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.3, 0.4])
            selected = rating >= 4
            feedback_system.record_feedback(
                row['aspirant_id'],
                row['mentor_id'],
                rating=rating,
                selected=selected
            )