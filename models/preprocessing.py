import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from data.config import SUBJECT_MAPPING, LEARNING_STYLES, TARGET_COLLEGES
from typing import Tuple

def preprocess_data(aspirants_df: pd.DataFrame, mentors_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess both aspirant and mentor data for the recommendation system.
    
    Args:
        aspirants_df: Raw aspirant data
        mentors_df: Raw mentor data
        
    Returns:
        Tuple of (processed_aspirants, processed_mentors) DataFrames
    """
    aspirants_processed = preprocess_aspirants(aspirants_df)
    mentors_processed = preprocess_mentors(mentors_df)
    return aspirants_processed, mentors_processed

def preprocess_aspirants(aspirants_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess aspirant data with minimal transformations.
    Main processing happens during recommendation vector creation.
    
    Args:
        aspirants_df: Raw aspirant data
        
    Returns:
        Processed aspirant DataFrame
    """
    # Create copy to avoid modifying original
    processed = aspirants_df.copy()
    
    # Ensure consistent naming in learning styles
    processed['learning_style'] = processed['learning_style'].str.replace('/', '_').str.lower()
    
    # Convert preparation level to ordered categorical
    prep_level_order = ['Beginner', 'Intermediate', 'Advanced']
    processed['preparation_level'] = pd.Categorical(
        processed['preparation_level'],
        categories=prep_level_order,
        ordered=True
    )
    
    return processed

def preprocess_mentors(mentors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess mentor data with feature engineering and transformations.
    
    Args:
        mentors_df: Raw mentor data
        
    Returns:
        Processed mentor DataFrame
    """
    processed = mentors_df.copy()
    
    # 1. Handle learning style matching
    processed = _add_teaching_style_features(processed)
    
    # 2. One-hot encode colleges
    processed = _encode_colleges(processed)
    
    # 3. Normalize numerical features
    processed = _normalize_features(processed)
    
    return processed

def _add_teaching_style_features(mentors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary features for each teaching style.
    
    Args:
        mentors_df: Mentor DataFrame
        
    Returns:
        DataFrame with added teaching style features
    """
    processed = mentors_df.copy()
    
    # Clean teaching style values
    processed['teaching_style'] = (
        processed['teaching_style']
        .str.replace('/', '_')
        .str.lower()
    )
    
    # Create binary columns for each teaching style
    for style in ['visual', 'auditory', 'reading_writing', 'kinesthetic', 'mixed']:
        processed[f'teaches_{style}'] = (
            (processed['teaching_style'] == style) | 
            (processed['teaching_style'] == 'mixed')
        ).astype(int)
    
    return processed

def _encode_colleges(mentors_df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode college information.
    
    Args:
        mentors_df: Mentor DataFrame
        
    Returns:
        DataFrame with one-hot encoded colleges
    """
    processed = mentors_df.copy()
    
    # Create encoder for all possible colleges (even if not present in current data)
    college_encoder = OneHotEncoder(
        categories=[TARGET_COLLEGES],
        sparse_output=False,
        handle_unknown='ignore'
    )
    
    # Fit and transform college data
    college_encoded = college_encoder.fit_transform(processed[['college']])
    college_columns = [f'college_{col.lower().replace(" ", "_")}' for col in TARGET_COLLEGES]
    
    # Create DataFrame with encoded colleges
    college_df = pd.DataFrame(
        college_encoded,
        columns=college_columns,
        index=processed.index
    )
    
    # Combine with original data
    processed = pd.concat([processed, college_df], axis=1)
    
    return processed

def _normalize_features(mentors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numerical features to [0,1] range.
    
    Args:
        mentors_df: Mentor DataFrame
        
    Returns:
        DataFrame with normalized numerical features
    """
    processed = mentors_df.copy()
    
    # Define numerical features to normalize
    numerical_features = [
        'clat_score',
        'mentorship_experience_months',
        'legal_reasoning_expertise',
        'logical_reasoning_expertise',
        'english_expertise',
        'current_affairs_expertise',
        'quantitative_expertise'
    ]
    
    # Create and fit scaler
    scaler = MinMaxScaler()
    processed[numerical_features] = scaler.fit_transform(processed[numerical_features])
    
    return processed

def get_preprocessing_pipeline() -> ColumnTransformer:
    """
    Create a scikit-learn preprocessing pipeline for mentor data.
    Useful if integrating with scikit-learn models.
    
    Returns:
        ColumnTransformer with preprocessing steps
    """
    # Numerical features pipeline
    numerical_features = [
        'clat_score',
        'mentorship_experience_months',
        'legal_reasoning_expertise',
        'logical_reasoning_expertise',
        'english_expertise',
        'current_affairs_expertise',
        'quantitative_expertise'
    ]
    numerical_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    
    # Categorical features pipeline
    categorical_features = ['college']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Teaching style features (will be handled separately)
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Leave other columns unchanged
    )
    
    return preprocessor

if __name__ == "__main__":
    # Test the preprocessing functions
    from data.generate_data import generate_aspirant_data, generate_mentor_data
    
    print("Generating sample data...")
    aspirants = generate_aspirant_data(10)
    mentors = generate_mentor_data(5)
    
    print("\nOriginal Aspirant Data:")
    print(aspirants.head())
    
    print("\nOriginal Mentor Data:")
    print(mentors.head())
    
    print("\nPreprocessing data...")
    aspirants_processed, mentors_processed = preprocess_data(aspirants, mentors)
    
    print("\nProcessed Aspirant Data:")
    print(aspirants_processed.head())
    
    print("\nProcessed Mentor Data:")
    print(mentors_processed.head())
    
    print("\nMentor Numerical Features Summary:")
    print(mentors_processed.describe())
    
    print("\nTeaching Style Features:")
    print(mentors_processed.filter(regex='teaches_').sum())
    
    print("\nCollege Encoding Features:")
    print(mentors_processed.filter(regex='college_').sum())