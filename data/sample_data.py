"""
Sample data for CLAT Mentor Recommendation System

This module provides sample datasets for testing and demonstration purposes.
"""

import pandas as pd
from data.generate_data import generate_aspirant_data, generate_mentor_data

def load_sample_aspirants(n=50) -> pd.DataFrame:
    """Load sample aspirant data"""
    return generate_aspirant_data(n)

def load_sample_mentors(n=15) -> pd.DataFrame:
    """Load sample mentor data"""
    return generate_mentor_data(n)

def save_sample_data(output_dir="data/sample"):
    """Generate and save sample datasets"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    aspirants = load_sample_aspirants(100)
    mentors = load_sample_mentors(20)
    
    aspirants.to_csv(f"{output_dir}/sample_aspirants.csv", index=False)
    mentors.to_csv(f"{output_dir}/sample_mentors.csv", index=False)
    
    print(f"Sample data saved to {output_dir}")

def load_saved_sample_data(data_dir="data/sample") -> tuple:
    """Load previously saved sample data"""
    aspirants = pd.read_csv(f"{data_dir}/sample_aspirants.csv")
    mentors = pd.read_csv(f"{data_dir}/sample_mentors.csv")
    return aspirants, mentors

if __name__ == "__main__":
    # Generate and save sample data when run directly
    save_sample_data()