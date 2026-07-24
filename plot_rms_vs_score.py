import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

features_file_path = "GTZAN/bands_dev_global_features_nyquist.csv"
features_data = pd.read_csv(features_file_path)

# square the column "TotalRMS"
features_data['TotalPower'] = features_data['TotalRMS'] ** 2
# sort by TotalPower
features_data = features_data.sort_values(by='TotalPower', ascending=False)

scores_file_path = "GTZAN/original_song_scores.csv"
scores_data = pd.read_csv(scores_file_path)

# merge scores_data with features_data on the song name
merged_data = pd.merge(
    scores_data[['song_name', 'score']],
    features_data,
    left_on='song_name',
    right_on=features_data.columns[0]
)

outliers_file_path = "GTZAN/outliers.txt"
with open(outliers_file_path, 'r') as f:
    outliers = f.read().splitlines()
    
# remove outliers from merged_data
merged_data = merged_data[~merged_data['song_name'].isin(outliers)]

# normalize score to 0-1
min_score = merged_data['score'].min()
max_score = merged_data['score'].max()
merged_data['score'] = (merged_data['score'] - min_score) / (max_score - min_score)

# plot TotalPower vs score
plt.figure(figsize=(4, 3))
plt.scatter(merged_data['TotalRMS'], merged_data['score'], alpha=0.5)
plt.xlabel('Total RMS')
plt.ylabel('Score ')
# plt.title('GTZAN Total RMS vs Score')
plt.grid(True)

# add a trend line
z = np.polyfit(merged_data['TotalRMS'], merged_data['score'], 1)
p = np.poly1d(z)
# print the equation of the line
print(f"Trend line equation: y = {z[0]:.2f}x + {z[1]:.2f}")

z2 = np.polyfit(merged_data['TotalRMS'], merged_data['score'], 2)
p2 = np.poly1d(z2)
# print the equation of the line
print(f"Trend line equation (2nd degree): y = {z2[0]:.2f}x^2 + {z2[1]:.2f}x + {z2[2]:.2f}")
# Sort data by 'TotalRMS' for smooth curve plotting
sorted_data = merged_data.sort_values(by='TotalRMS')
# plt.plot(merged_data['TotalRMS'], p(merged_data['TotalRMS']), "r--")
correlation_matrix2 = np.corrcoef(merged_data['score'], p2(merged_data['TotalRMS']))
correlation_xy2 = correlation_matrix2[0,1]
r_squared2 = correlation_xy2**2
plt.plot(sorted_data['TotalRMS'], p2(sorted_data['TotalRMS']), "r", label=f'Trend Line (R²={r_squared2:.3f})')

# determine which trend line fits better by calculating the r^2 value
# correlation_matrix = np.corrcoef(merged_data['score'], p(merged_data['TotalRMS']))
# correlation_xy = correlation_matrix[0,1]
# r_squared = correlation_xy**2
# print(f"R^2 value for 1st degree trend line: {r_squared:.3f}")

print(f"R^2 value for 2nd degree trend line: {r_squared2:.3f}")

# plt.legend([
#     # 'Songs', 
#     # f'1st Degree Trend Line (R²={r_squared:.3f})', 
#     # f'2nd Degree Trend Line (R²={r_squared2:.3f})'
# ])
plt.tight_layout()
# plt.legend()
plt.show()

# analytical normalization of score by 2nd degree TotalRMS. Remove TotalRMS confounding factor
mean_score = np.mean(merged_data['score'])
a = z2[0]
b = z2[1]
c = z2[2]
mean_rms = np.mean(merged_data['TotalRMS'])
print(f"Mean score: {mean_score:.3f}, Mean TotalRMS: {mean_rms:.3f}")
# normalized_score = score - (a*TotalRMS^2 + b*TotalRMS + c) + mean_score
merged_data['normalized_score'] = merged_data['score'] - (a * merged_data['TotalRMS']**2 + b * merged_data['TotalRMS'] + c) + (a * mean_rms**2 + b * mean_rms + c)
# plot normalized_score vs TotalRMS
plt.figure(figsize=(4, 3))
plt.scatter(merged_data['TotalRMS'], merged_data['normalized_score'], alpha=0.6, color='orange')
plt.xlabel('Total RMS')
plt.ylabel('Normalized score')
# plt.title('GTZAN Total RMS vs Score (Normalized)')
plt.grid(True)

#add a trend line
z3 = np.polyfit(merged_data['TotalRMS'], merged_data['normalized_score'], 1)
p3 = np.poly1d(z3)
# print the equation of the line
print(f"Trend line equation: y = {z3[0]:.2f}x + {z3[1]:.2f}")

merged_data = merged_data.sort_values(by='TotalRMS')
plt.plot(merged_data['TotalRMS'], p3(merged_data['TotalRMS']), "r", label=f'Normalized Score Trend Line (slope={z3[0]:.1f})')
plt.tight_layout()
# plt.legend()
plt.show()