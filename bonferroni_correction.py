import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# calculate if differences in model performance are statistically significant
# with bonferroni correction 

def compare_code_text_models(models, alpha=0.05):
    """
    compare Code-based and Text-based models for each time frame (1, 5, 10 years) using multiple performance metrics.
    
    :param models: Dict of model data, each containing 'micro-f1' and 'micro-auprc' data
    :param alpha: Overall significance level (default 0.05)
    """
    metrics = ['micro-f1', 'micro-auprc']
    comparisons = [
        ("1yr Code", "1yr Text"),
        ("5yr Code", "5yr Text"),
        ("10yr Code", "10yr Text")
    ]
    
    # bonferroni corrected significance level
    num_comparisons = len(comparisons) * len(metrics)
    alpha_bonferroni = alpha / num_comparisons
    
    print(f"Performing {num_comparisons} comparisons")
    print(f"Bonferroni corrected significance level: {alpha_bonferroni:.6f}")
    
    fig, axes = plt.subplots(len(comparisons), 2, figsize=(15, 5 * len(comparisons)))
    
    for i, (model1_name, model2_name) in enumerate(comparisons):
        print(f"\nComparison: {model1_name} vs {model2_name}")
        for j, metric in enumerate(metrics):
            model1_score, model1_ci = models[model1_name][metric]
            model2_score, model2_ci = models[model2_name][metric]
            
            # Calculate the difference in scores
            diff = model2_score - model1_score  # Text - Code
            
            # Calculate the standard error of the difference
            se1 = (model1_ci[1] - model1_ci[0]) / (2 * 1.96)
            se2 = (model2_ci[1] - model2_ci[0]) / (2 * 1.96)
            se_diff = np.sqrt(se1**2 + se2**2)
            
            # Calculate z-score and p-value
            z_score = diff / se_diff
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            print(f"\n{metric.upper()}:")
            print(f"  {model1_name}: {model1_score:.4f} ({model1_ci[0]:.4f} - {model1_ci[1]:.4f})")
            print(f"  {model2_name}: {model2_score:.4f} ({model2_ci[0]:.4f} - {model2_ci[1]:.4f})")
            print(f"  Difference (Text - Code): {diff:.4f}")
            print(f"  Z-score: {z_score:.4f}")
            print(f"  P-value: {p_value:.6f}")
            print(f"  {'Statistically significant' if p_value < alpha_bonferroni else 'Not statistically significant'} at α = {alpha_bonferroni:.6f} (Bonferroni corrected)")
            
            # Visualization
            ax = axes[i, j]
            ax.errorbar([model1_name, model2_name], [model1_score, model2_score], 
                        yerr=[[model1_score - model1_ci[0], model2_score - model2_ci[0]],
                              [model1_ci[1] - model1_score, model2_ci[1] - model2_score]],
                        fmt='o', capsize=5, capthick=2)
            ax.set_title(f"{metric.upper()} - {model1_name[:4]}")
            ax.set_ylabel("Score")
            ax.set_ylim(min(model1_ci[0], model2_ci[0]) * 0.9, max(model1_ci[1], model2_ci[1]) * 1.1)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
    
    plt.tight_layout()
    plt.show()

# Example usage
models = {
    "1yr Code": {
        'micro-f1': (0.45, (0.44, 0.46)),
        'micro-auprc': (0.44, (0.43, 0.46))
    },
    "5yr Code": {
        'micro-f1': (0.43, (0.42, 0.44)),
        'micro-auprc': (0.43, (0.41, 0.44))
    },
    "10yr Code": {
        'micro-f1': (0.47, (0.46, 0.49)),
        'micro-auprc': (0.47, (0.45, 0.48))
    },
    "1yr Text": {
        'micro-f1': (0.45, (0.43, 0.46)), # need to replace
        'micro-auprc': (0.44, (0.42, 0.45))
    },
    "5yr Text": {
        'micro-f1': (0.50, (0.49, 0.51)),
        'micro-auprc': (0.51, (0.50, 0.52))
    },
    "10yr Text": {
        'micro-f1': (0.49, (0.48, 0.50)),
        'micro-auprc': (0.50, (0.49, 0.51))
    }
}

compare_code_text_models(models)