import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Dict, Optional
from data.config import SUBJECT_MAPPING, PREPARATION_LEVELS, TARGET_COLLEGES

class MentorRecommender:
    """
    A content-based recommendation system for matching CLAT aspirants with mentors.
    Uses cosine similarity between aspirant preference vectors and mentor profiles.
    """
    
    def __init__(self, mentors_processed: pd.DataFrame):
        """
        Initialize the recommender with processed mentor data.
        
        Args:
            mentors_processed: Preprocessed mentor DataFrame
        """
        self.mentors = mentors_processed
        self.mentor_profiles = self._create_mentor_profiles()
        self.feature_weights = self._get_default_feature_weights()
        
    def _create_mentor_profiles(self) -> pd.DataFrame:
        """
        Create mentor profiles from processed data.
        
        Returns:
            DataFrame of mentor feature vectors
        """
        # Select important features for matching
        features = [
            'legal_reasoning_expertise', 'logical_reasoning_expertise',
            'english_expertise', 'current_affairs_expertise', 'quantitative_expertise',
            'teaches_visual', 'teaches_auditory', 'teaches_reading_writing', 
            'teaches_kinesthetic', 'teaches_mixed',
            'mentorship_experience_months', 'clat_score'
        ]
        
        # Add college features
        college_features = [col for col in self.mentors.columns if col.startswith('college_')]
        features.extend(college_features)
        
        return self.mentors[features]
    
    def _get_default_feature_weights(self) -> Dict[str, float]:
        """
        Get default weights for different features in similarity calculation.
        
        Returns:
            Dictionary of feature weights
        """
        return {
            'subject_expertise': 0.4,        # Combined weight for all subjects
            'learning_style': 0.3,          # Matching teaching/learning styles
            'college_match': 0.2,           # Target college alignment
            'mentorship_experience': 0.05,   # Mentor experience
            'clat_score': 0.05              # Mentor's CLAT performance
        }
    
    def create_aspirant_vector(self, aspirant: pd.Series) -> pd.Series:
        """
        Create a feature vector for an aspirant that matches mentor profile dimensions.
        
        Args:
            aspirant: Aspirant data as a pandas Series
            
        Returns:
            Aspirant feature vector
        """
        # Initialize vector with zeros
        aspirant_vector = pd.Series(0, index=self.mentor_profiles.columns)
        
        # 1. Set subject preferences (normalized to 0-1 scale)
        for subject, field in SUBJECT_MAPPING.items():
            aspirant_vector[f'{field}_expertise'] = aspirant[f'{field}_preference'] / 10
        
        # Apply subject expertise weight
        subject_cols = [f'{field}_expertise' for field in SUBJECT_MAPPING.values()]
        aspirant_vector[subject_cols] *= self.feature_weights['subject_expertise']
        
        # 2. Set learning style preference
        learning_style = aspirant['learning_style'].lower()
        for style in ['visual', 'auditory', 'reading_writing', 'kinesthetic', 'mixed']:
            if style in learning_style:
                aspirant_vector[f'teaches_{style}'] = 1
        
        # Apply learning style weight
        style_cols = [col for col in aspirant_vector.index if col.startswith('teaches_')]
        aspirant_vector[style_cols] *= self.feature_weights['learning_style']
        
        # 3. Set college preference
        target_colleges = [
            aspirant['target_college_1'],
            aspirant['target_college_2']
        ]
        for college in target_colleges:
            if college and f'college_{college.lower().replace(" ", "_")}' in aspirant_vector.index:
                aspirant_vector[f'college_{college.lower().replace(" ", "_")}'] = 1
        
        # Apply college match weight
        college_cols = [col for col in aspirant_vector.index if col.startswith('college_')]
        aspirant_vector[college_cols] *= self.feature_weights['college_match']
        
        # 4. Set preparation level importance
        prep_level_importance = {
            'Beginner': 0.8,
            'Intermediate': 0.5,
            'Advanced': 0.3
        }
        aspirant_vector['mentorship_experience_months'] = (
            prep_level_importance[aspirant['preparation_level']] * 
            self.feature_weights['mentorship_experience']
        )
        
        # 5. CLAT score importance
        aspirant_vector['clat_score'] = self.feature_weights['clat_score']
        
        return aspirant_vector
    
    def recommend_mentors(
        self,
        aspirant: pd.Series,
        top_n: int = 3,
        diversity_factor: float = 0.2,
        min_similarity: float = 0.5
    ) -> pd.DataFrame:
        """
        Recommend top N mentors for an aspirant based on cosine similarity.
        
        Args:
            aspirant: Aspirant data
            top_n: Number of recommendations to return
            diversity_factor: Controls diversity of recommendations (0-1)
            min_similarity: Minimum similarity score threshold
            
        Returns:
            DataFrame of recommended mentors with similarity scores
        """
        # Create aspirant vector
        aspirant_vector = self.create_aspirant_vector(aspirant)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(
            [aspirant_vector], 
            self.mentor_profiles
        )[0]
        
        # Apply diversity factor
        if diversity_factor > 0:
            similarities = self._apply_diversity(similarities, diversity_factor)
        
        # Get top N recommendations
        recommendations = self._get_top_recommendations(similarities, top_n, min_similarity)
        
        return recommendations
    
    def _apply_diversity(self, similarities: np.ndarray, factor: float) -> np.ndarray:
        """
        Apply diversity to recommendations by slightly penalizing similarity with mentors
        who are too similar to each other.
        
        Args:
            similarities: Array of similarity scores
            factor: Diversity factor (0-1)
            
        Returns:
            Adjusted similarity scores
        """
        # Calculate mentor-mentor similarity
        mentor_similarity = cosine_similarity(self.mentor_profiles)
        
        # For each mentor, calculate average similarity to other mentors
        avg_similarity = mentor_similarity.mean(axis=1)
        
        # Adjust original similarities
        return similarities * (1 - factor * avg_similarity)
    
    def _get_top_recommendations(
        self,
        similarities: np.ndarray,
        top_n: int,
        min_similarity: float
    ) -> pd.DataFrame:
        """
        Get top N recommendations meeting minimum similarity threshold.
        
        Args:
            similarities: Array of similarity scores
            top_n: Number of recommendations
            min_similarity: Minimum similarity threshold
            
        Returns:
            DataFrame of recommendations
        """
        # Filter by minimum similarity
        valid_indices = np.where(similarities >= min_similarity)[0]
        
        if len(valid_indices) == 0:
            return pd.DataFrame()  # Return empty if no matches meet threshold
        
        # Get top N indices
        top_indices = similarities[valid_indices].argsort()[::-1][:top_n]
        top_indices = valid_indices[top_indices]
        
        # Create recommendations DataFrame
        recommendations = self.mentors.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        # Select relevant columns
        output_cols = [
            'mentor_id', 'name', 'clat_rank', 'clat_score', 'college',
            'specialization', 'teaching_style', 'mentorship_experience_months',
            'similarity_score'
        ]
        
        return recommendations[output_cols].sort_values('similarity_score', ascending=False)
    
    def explain_recommendation(
        self,
        aspirant: pd.Series,
        mentor_id: str
    ) -> Dict[str, float]:
        """
        Generate explanation for why a particular mentor was recommended.
        
        Args:
            aspirant: Aspirant data
            mentor_id: ID of mentor to explain
            
        Returns:
            Dictionary of explanation components and scores
        """
        # Get mentor data
        mentor = self.mentors[self.mentors['mentor_id'] == mentor_id].iloc[0]
        
        # Create aspirant vector
        aspirant_vector = self.create_aspirant_vector(aspirant)
        
        # Get mentor profile
        mentor_profile = self.mentor_profiles.loc[mentor.name]
        
        # Calculate contribution of each feature group
        explanation = {
            'subject_match': self._calculate_subject_match(aspirant, mentor),
            'learning_style_match': self._calculate_learning_style_match(aspirant, mentor),
            'college_match': self._calculate_college_match(aspirant, mentor),
            'experience_match': self._calculate_experience_match(aspirant, mentor),
            'clat_score_match': mentor_profile['clat_score']
        }
        
        return explanation
    
    def _calculate_subject_match(self, aspirant: pd.Series, mentor: pd.Series) -> float:
        """Calculate subject expertise match score."""
        score = 0
        for subject, field in SUBJECT_MAPPING.items():
            pref = aspirant[f'{field}_preference'] / 10
            expertise = mentor[f'{field}_expertise']
            score += pref * expertise
        
        return score * self.feature_weights['subject_expertise'] / len(SUBJECT_MAPPING)
    
    def _calculate_learning_style_match(self, aspirant: pd.Series, mentor: pd.Series) -> float:
        """Calculate learning style match score."""
        learning_style = aspirant['learning_style'].lower()
        teaches_style = f'teaches_{learning_style}'
        
        if teaches_style in mentor:
            return mentor[teaches_style] * self.feature_weights['learning_style']
        return 0
    
    def _calculate_college_match(self, aspirant: pd.Series, mentor: pd.Series) -> float:
        """Calculate college match score."""
        target_colleges = [
            aspirant['target_college_1'].lower().replace(" ", "_"),
            aspirant['target_college_2'].lower().replace(" ", "_")
        ]
        
        for college in target_colleges:
            college_feature = f'college_{college}'
            if college_feature in mentor and mentor[college_feature] > 0:
                return self.feature_weights['college_match']
        
        return 0
    
    def _calculate_experience_match(self, aspirant: pd.Series, mentor: pd.Series) -> float:
        """Calculate mentorship experience match score."""
        prep_level = aspirant['preparation_level']
        importance = {
            'Beginner': 0.8,
            'Intermediate': 0.5,
            'Advanced': 0.3
        }.get(prep_level, 0.5)
        
        return mentor['mentorship_experience_months'] * importance * self.feature_weights['mentorship_experience']


def batch_recommend(
    aspirants_df: pd.DataFrame,
    mentors_processed: pd.DataFrame,
    top_n: int = 3,
    min_similarity: float = 0.5,
    diversity_factor: float = 0.2,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate recommendations for multiple aspirants at once.
    
    Args:
        aspirants_df: DataFrame of aspirants
        mentors_processed: Processed mentor data
        top_n: Number of recommendations per aspirant
        min_similarity: Minimum similarity threshold
        diversity_factor: Diversity adjustment factor
        output_path: Optional path to save results
        
    Returns:
        DataFrame with recommendations for all aspirants
    """
    recommender = MentorRecommender(mentors_processed)
    all_recommendations = []
    
    for _, aspirant in aspirants_df.iterrows():
        recommendations = recommender.recommend_mentors(
            aspirant,
            top_n=top_n,
            min_similarity=min_similarity,
            diversity_factor=diversity_factor
        )
        recommendations['aspirant_id'] = aspirant['aspirant_id']
        all_recommendations.append(recommendations)
    
    combined = pd.concat(all_recommendations, ignore_index=True)
    
    if output_path:
        combined.to_csv(output_path, index=False)
    
    return combined


if __name__ == "__main__":
    # Test the recommendation system
    from data.generate_data import generate_aspirant_data, generate_mentor_data
    from preprocessing import preprocess_data
    
    print("Generating sample data...")
    aspirants = generate_aspirant_data(10)
    mentors = generate_mentor_data(5)
    
    print("\nPreprocessing data...")
    aspirants_processed, mentors_processed = preprocess_data(aspirants, mentors)
    
    print("\nInitializing recommender...")
    recommender = MentorRecommender(mentors_processed)
    
    # Test with first aspirant
    sample_aspirant = aspirants_processed.iloc[0]
    print(f"\nSample aspirant: {sample_aspirant['aspirant_id']}")
    print(f"Preferred subjects: {sample_aspirant['preferred_subject_1']}, {sample_aspirant['preferred_subject_2']}")
    print(f"Learning style: {sample_aspirant['learning_style']}")
    print(f"Target colleges: {sample_aspirant['target_college_1']}, {sample_aspirant['target_college_2']}")
    
    print("\nGenerating recommendations...")
    recommendations = recommender.recommend_mentors(sample_aspirant)
    
    print("\nTop Recommendations:")
    print(recommendations[['name', 'specialization', 'teaching_style', 'college', 'similarity_score']])
    
    if not recommendations.empty:
        print("\nExplanation for top recommendation:")
        explanation = recommender.explain_recommendation(sample_aspirant, recommendations.iloc[0]['mentor_id'])
        for factor, score in explanation.items():
            print(f"{factor.replace('_', ' ').title()}: {score:.3f}")
    
    print("\nRunning batch recommendations for all aspirants...")
    batch_results = batch_recommend(aspirants_processed, mentors_processed)
    print(f"\nGenerated {len(batch_results)} total recommendations")
    print("\nSample batch results:")
    print(batch_results.head())
    