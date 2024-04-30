If the levels are ordered, you could use numerical encoding ("label encoding", but assuring that the numbers are assigned in correct order.)
If not ordered, you need dummy variables.

to-do:
- check for correlations between variables (heatmap, etc)

Type_encoded x Level_encoded
correlation_coefficient, p_value = pearsonr(data['Type_encoded'], data['Level_encoded'])
Pearson correlation coefficient: 0.1125080108593004 - Sig but weak
P-value: 1.0993840127818986e-09 - Sig

while the correlation coefficient is statistically significant due to the small p-value,
the actual strength of the correlation is very weak, suggesting that the 'Type_encoded' and 'Level_encoded'
variables are not strongly related in a linear fashion.

