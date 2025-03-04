from scipy.stats import ttest_rel
import numpy as np

baseline = np.array([
    [0.5801, 0.6043, 0.7360, 0.5237, 0.7360],  
    [0.4862, 0.5772, 0.6872, 0.4389, 0.6872],  
    [0.6507, 0.6553, 0.7483, 0.6152, 0.7483],  
    [0.5288, 0.6053, 0.7226, 0.4904, 0.7226],   
    [0.6185, 0.6529, 0.7589, 0.5876, 0.7589]  
])

improved = np.array([
    [0.8477, 0.6669, 0.6321, 0.6458, 0.6321], 
    [0.8621, 0.7060, 0.7983, 0.7370, 0.7983],  
    [0.8433, 0.7549, 0.6984, 0.7196, 0.6984],  
    [0.8926, 0.8503, 0.7796, 0.8077, 0.7796],  
    [0.8942, 0.6562, 0.6404, 0.6477, 0.6404]   
])

# Perform paired t-tests for each metric
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
for i, metric in enumerate(metrics):
    stat, p_value = ttest_rel(improved[:, i], baseline[:, i])
    print(f"{metric}: t-statistic = {stat:.4f}, p-value = {p_value:.4f}")

    if p_value < 0.05:
        print(f"  -> Significant improvement in {metric} (p < 0.05)")
    else:
        print(f"  -> No significant improvement in {metric} (p ≥ 0.05)")
