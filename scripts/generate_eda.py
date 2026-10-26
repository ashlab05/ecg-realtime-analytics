#!/usr/bin/env python3
"""Exploratory Data Analysis script for the MIT-BIH dataset.

Loads the dataset using the core preprocessing library and generates
figures for class distribution, feature distributions, and correlations.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg_core.preprocessing import load_dataset
from ecg_core.labels import CLASS_NAMES

def main():
    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Class Distribution
    print("Generating class distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='type', order=CLASS_NAMES, hue='type', palette='viridis', legend=False)
    plt.title('Class Distribution in MIT-BIH Dataset')
    plt.xlabel('Beat Class')
    plt.ylabel('Count')
    plt.yscale('log') # Log scale is useful because Normal class is 96%
    plt.text(-0.2, 1e5, 'Note: Log Scale', fontsize=10, style='italic', color='grey')
    plt.tight_layout()
    plt.savefig(out_dir / 'class_distribution.png')
    plt.close()
    
    # 2. RR Intervals across classes
    print("Generating RR interval distribution plot...")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='type', y='0_pre-RR', order=CLASS_NAMES, hue='type', palette='Set2', legend=False)
    plt.title('Pre-RR Interval by Class')
    plt.ylim(0, 1000) # Trim outliers for better view
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='type', y='0_post-RR', order=CLASS_NAMES, hue='type', palette='Set2', legend=False)
    plt.title('Post-RR Interval by Class')
    plt.ylim(0, 1000)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'rr_intervals.png')
    plt.close()
    
    # 3. Correlation Matrix (Lead II Features)
    print("Generating correlation matrix...")
    lead2_cols = [c for c in df.columns if c.startswith('0_')]
    if lead2_cols:
        corr = df[lead2_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Feature Correlation Matrix (Lead II)')
        plt.tight_layout()
        plt.savefig(out_dir / 'correlation_matrix.png')
        plt.close()
    
    print(f"EDA figures saved to {out_dir}/")

if __name__ == "__main__":
    main()
