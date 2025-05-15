import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def create_visualizations(train_df, output_dir="models"):
    """
    Create visualizations for the Titanic dataset
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data
    output_dir : str
        Directory to save the visualizations
    """
    logger.info("Creating visualizations")
    
    # Create output directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # 1. Survival by Sex
    logger.info("Creating survival by sex visualization")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Sex', hue='Survived', data=train_df)
    plt.title('Survival by Sex')
    plt.savefig(os.path.join(vis_dir, 'survival_by_sex.png'))
    plt.close()
    
    # 2. Survival by Pclass
    logger.info("Creating survival by class visualization")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Pclass', hue='Survived', data=train_df)
    plt.title('Survival by Passenger Class')
    plt.savefig(os.path.join(vis_dir, 'survival_by_class.png'))
    plt.close()
    
    # 3. Age distribution
    logger.info("Creating age distribution visualization")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_df, x='Age', hue='Survived', multiple='stack', bins=20)
    plt.title('Age Distribution by Survival')
    plt.savefig(os.path.join(vis_dir, 'age_distribution.png'))
    plt.close()
    
    # 4. Fare distribution
    logger.info("Creating fare distribution visualization")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_df, x='Fare', hue='Survived', multiple='stack', bins=20)
    plt.title('Fare Distribution by Survival')
    plt.savefig(os.path.join(vis_dir, 'fare_distribution.png'))
    plt.close()
    
    # 5. Correlation heatmap
    logger.info("Creating correlation heatmap")
    plt.figure(figsize=(12, 8))
    numeric_df = train_df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(vis_dir, 'correlation_heatmap.png'))
    plt.close()
    
    logger.info(f"Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    # This allows the module to be run as a script for testing
    logging.basicConfig(level=logging.INFO)
    
    # For testing, use the training dataset
    import sys
    import os
    
    # Add parent directory to path
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    
    from data.make_dataset import load_data
    
    # Update paths as needed
    train_path = "../../data/raw/train.csv"
    test_path = "../../data/raw/test.csv"
    
    train_df, _ = load_data(train_path, test_path)
    
    # Create visualizations
    create_visualizations(train_df)
    
    logger.info("Script executed successfully")