import random
import pandas as pd
import numpy as np
from config import *

def generate_aspirant_data(n=100):
    """Generate mock data for CLAT aspirants."""
    data = []
    
    for i in range(n):
        # First generate all the independent fields
        preferred_subjects = random.sample(SUBJECTS, k=random.randint(2, 4))
        preferences = {subject: (5 if subject in preferred_subjects else random.randint(1, 3)) 
                     for subject in SUBJECTS}
        target_college_1 = random.choice(TARGET_COLLEGES)
        
        # Now create the aspirant dict with all fields
        aspirant = {
            'aspirant_id': f'ASP{i+1:03d}',
            'preferred_subject_1': preferred_subjects[0],
            'preferred_subject_2': preferred_subjects[1] if len(preferred_subjects) > 1 else None,
            'target_college_1': target_college_1,
            'target_college_2': random.choice([c for c in TARGET_COLLEGES if c != target_college_1]),
            'preparation_level': random.choice(PREPARATION_LEVELS),
            'learning_style': random.choice(LEARNING_STYLES),
            'hours_per_week': random.randint(10, 50),
            'legal_reasoning_preference': preferences['Legal Reasoning'],
            'logical_reasoning_preference': preferences['Logical Reasoning'],
            'english_preference': preferences['English'],
            'current_affairs_preference': preferences['Current Affairs'],
            'quantitative_preference': preferences['Quantitative Techniques'],
        }
        data.append(aspirant)
    
    return pd.DataFrame(data)

def generate_mentor_data(n=20):
    """
    Generate mock data for CLAT mentors.
    
    Args:
        n (int): Number of mentors to generate
        
    Returns:
        pd.DataFrame: DataFrame containing mentor data
    """
    data = []
    
    for i in range(n):
        specialization = random.choice(MENTOR_SPECIALIZATIONS)
        expertise = generate_expertise(specialization)
        
        clat_rank = random.randint(1, 200)
        average_expertise = sum(expertise.values()) / len(expertise)
        clat_score = int(150 * (0.8 + 0.2 * (average_expertise / 10)) * (1 - clat_rank/400))
        
        teaching_style = generate_teaching_style(specialization)
        
        mentor = {
            'mentor_id': f'MNT{i+1:03d}',
            'name': f'Mentor {i+1}',
            'clat_year': random.choice(EXAM_YEARS),
            'clat_rank': clat_rank,
            'clat_score': clat_score,
            'college': random.choice(TARGET_COLLEGES[:5]),  # Top mentors from top colleges
            'specialization': specialization,
            'teaching_style': teaching_style,
            'mentorship_experience_months': random.randint(6, 48),
            'legal_reasoning_expertise': expertise['Legal Reasoning'],
            'logical_reasoning_expertise': expertise['Logical Reasoning'],
            'english_expertise': expertise['English'],
            'current_affairs_expertise': expertise['Current Affairs'],
            'quantitative_expertise': expertise['Quantitative Techniques'],
        }
        data.append(mentor)
    
    return pd.DataFrame(data)

def generate_expertise(specialization):
    """
    Generate expertise levels based on mentor specialization.
    
    Args:
        specialization (str): Mentor's specialization area
        
    Returns:
        dict: Dictionary of expertise levels for each subject
    """
    expertise = {}
    
    if specialization == 'Legal Reasoning Expert':
        expertise = {
            'Legal Reasoning': random.randint(8, 10),
            'Logical Reasoning': random.randint(5, 8),
            'English': random.randint(5, 8),
            'Current Affairs': random.randint(5, 8),
            'Quantitative Techniques': random.randint(5, 8)
        }
    elif specialization == 'Logical Reasoning Expert':
        expertise = {
            'Legal Reasoning': random.randint(5, 8),
            'Logical Reasoning': random.randint(8, 10),
            'English': random.randint(5, 8),
            'Current Affairs': random.randint(5, 8),
            'Quantitative Techniques': random.randint(5, 8)
        }
    elif specialization == 'English Language Expert':
        expertise = {
            'Legal Reasoning': random.randint(5, 8),
            'Logical Reasoning': random.randint(5, 8),
            'English': random.randint(8, 10),
            'Current Affairs': random.randint(5, 8),
            'Quantitative Techniques': random.randint(5, 8)
        }
    elif specialization == 'Current Affairs Expert':
        expertise = {
            'Legal Reasoning': random.randint(5, 8),
            'Logical Reasoning': random.randint(5, 8),
            'English': random.randint(5, 8),
            'Current Affairs': random.randint(8, 10),
            'Quantitative Techniques': random.randint(5, 8)
        }
    elif specialization == 'Quantitative Expert':
        expertise = {
            'Legal Reasoning': random.randint(5, 8),
            'Logical Reasoning': random.randint(5, 8),
            'English': random.randint(5, 8),
            'Current Affairs': random.randint(5, 8),
            'Quantitative Techniques': random.randint(8, 10)
        }
    else:  # All-rounder
        expertise = {
            'Legal Reasoning': random.randint(7, 9),
            'Logical Reasoning': random.randint(7, 9),
            'English': random.randint(7, 9),
            'Current Affairs': random.randint(7, 9),
            'Quantitative Techniques': random.randint(7, 9)
        }
    
    return expertise

def generate_teaching_style(specialization):
    """
    Generate teaching style based on mentor specialization.
    
    Args:
        specialization (str): Mentor's specialization area
        
    Returns:
        str: Teaching style that matches the specialization
    """
    if specialization == 'Legal Reasoning Expert':
        return random.choice(['Reading/Writing', 'Visual', 'Mixed'])
    elif specialization == 'Logical Reasoning Expert':
        return random.choice(['Visual', 'Kinesthetic', 'Mixed'])
    elif specialization == 'English Language Expert':
        return random.choice(['Reading/Writing', 'Auditory', 'Mixed'])
    elif specialization == 'Current Affairs Expert':
        return random.choice(['Visual', 'Reading/Writing', 'Mixed'])
    elif specialization == 'Quantitative Expert':
        return random.choice(['Visual', 'Kinesthetic', 'Mixed'])
    else:  # All-rounder
        return random.choice(LEARNING_STYLES)

def validate_data(aspirants_df, mentors_df):
    """
    Validate the generated data for consistency and quality.
    
    Args:
        aspirants_df (pd.DataFrame): Aspirant data
        mentors_df (pd.DataFrame): Mentor data
        
    Returns:
        bool: True if data passes validation checks
    """
    # Check for missing values
    if aspirants_df.isnull().sum().sum() > 0:
        raise ValueError("Aspirant data contains missing values")
    if mentors_df.isnull().sum().sum() > 0:
        raise ValueError("Mentor data contains missing values")
    
    # Check value ranges
    for col in ['legal_reasoning_preference', 'logical_reasoning_preference',
               'english_preference', 'current_affairs_preference',
               'quantitative_preference']:
        if not aspirants_df[col].between(1, 5).all():
            raise ValueError(f"Aspirant {col} values out of range (1-5)")
    
    for col in ['legal_reasoning_expertise', 'logical_reasoning_expertise',
               'english_expertise', 'current_affairs_expertise',
               'quantitative_expertise']:
        if not mentors_df[col].between(1, 10).all():
            raise ValueError(f"Mentor {col} values out of range (1-10)")
    
    return True

if __name__ == "__main__":
    # Generate sample data when run directly
    aspirants = generate_aspirant_data(10)
    mentors = generate_mentor_data(5)
    
    print("Sample Aspirants:")
    print(aspirants.head())
    
    print("\nSample Mentors:")
    print(mentors.head())
    
    # Validate the generated data
    try:
        validate_data(aspirants, mentors)
        print("\nData validation passed successfully")
    except ValueError as e:
        print(f"\nData validation failed: {str(e)}")