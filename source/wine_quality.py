import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols, glm
file_path = 'winequality-both.csv'
wine = pd.read_csv(file_path,header=0,sep=',')
print(wine)
# update column where space to '_'
wine.columns = wine.columns.str.replace(' ', '_') 
# display sample record 5 rows
print(wine.head())
# Display descriptive statistics for all variables
print(wine.describe())
# Identify unique values
print(sorted(wine.quality.unique())) 
# Calculate value frequencies 
print(wine.quality.value_counts())
# Display descriptive statistics for quality by wine type
print(wine.groupby('type')[['quality']].describe().unstack('type'))
# Display specific quantile values for quality by wine type 
print(wine.groupby('type')[['quality']].quantile([0.25, 0.75]).unstack('type'))
# Look at the distribution of quality by wine type
red_wine = wine.loc[wine['type']=='red', 'quality']
white_wine = wine.loc[wine['type']=='white', 'quality']
sns.set_style("dark")
print(sns.distplot(red_wine, norm_hist=True, kde=False, color="red", label="Red wine"))
print(sns.distplot(white_wine, norm_hist=True, kde=False, color="white", label="White wine"))
sns.utils.axlabel("Quality Score", "Density")
plt.title("Distribution of Quality by Wine Type")
plt.legend()
plt.show()
# Test whether mean quality is different between red and white wines 
print(wine.groupby(['type'])[['quality']].agg(['std']))
tstat, pvalue, df = sm.stats.ttest_ind(red_wine, white_wine) 
print('tstat: %.3f pvalue: %.4f' % (tstat, pvalue))
# Calculate correlation matrix for all variables
print(wine.corr())
# Take a "small" sample of red and white wines for plotting 
def take_sample(data_frame, replace=False, n=200):
    return data_frame.loc[np.random.choice(data_frame.index,replace=replace, size=n)]

reds_sample = take_sample(wine.loc[wine['type']=='red', :])
whites_sample = take_sample(wine.loc[wine['type']=='white', :])
wine_sample = pd.concat([reds_sample, whites_sample])
wine['in_sample'] = np.where(wine.index.isin(wine_sample.index), 1.,0.) 
print(pd.crosstab(wine.in_sample, wine.type, margins=True))
# Look at relationship between pairs of variables
sns.set_style("dark")
g = sns.pairplot(wine_sample, kind='reg', plot_kws={"ci": False, "x_jitter": 0.25, "y_jitter": 0.25}, hue='type', diag_kind='hist',diag_kws={"bins": 10, "alpha": 1.0}, palette=dict(red="red", white="white"),markers=["o", "s"], vars=['quality', 'alcohol', 'residual_sugar'])
print(g)
plt.suptitle('Histograms and Scatter Plots of Quality, Alcohol, and Residual Sugar', fontsize=14, horizontalalignment='center', verticalalignment='top', x=0.5, y=0.999)
plt.show()
my_formula = 'quality ~ alcohol + chlorides + citric_acid + density + fixed_acidity + free_sulfur_dioxide + pH + residual_sugar + sulphates + total_sulfur_dioxide + volatile_acidity'
lm = ols(my_formula, data=wine).fit()

## Alternatively, a linear regression using generalized linear model (glm) syntax
## lm = glm(my_formula, data=wine, family=sm.families.Gaussian()).fit()
print(lm.summary())
print("\nQuantities you can extract from the result:\n%s" % dir(lm)) 
print("\nCoefficients:\n%s" % lm.params)
print("\nCoefficient Std Errors:\n%s" % lm.bse)
print("\nAdj. R-squared:\n%.2f" % lm.rsquared_adj) 
print("\nF-statistic: %.1f P-value: %.2f" % (lm.fvalue, lm.f_pvalue)) 
print("\nNumber of obs: %d Number of fitted values: %d" % (lm.nobs,len(lm.fittedvalues)))
 # Create a Series named dependent_variable to hold the quality data
dependent_variable = wine['quality']
# Create a DataFrame named independent variables to hold all of the variables # from the original wine dataset except quality, type, and in_sample 
independent_variables = wine[wine.columns.difference(['quality', 'type','in_sample'])]
# Standardize the independent variables. For each variable,
# subtract the variable's mean from each observation and
# divide each result by the variable's standard deviation 
independent_variables_standardized = (independent_variables -independent_variables.mean()) / independent_variables.std()
# Add the dependent variable, quality, as a column in the DataFrame of
# independent variables to create a new dataset with
# standardized independent variables
wine_standardized = pd.concat([dependent_variable, independent_variables_standardized], axis=1)
lm_standardized = ols(my_formula, data=wine_standardized).fit()
print(lm_standardized.summary())
# Create 10 "new" observations from the first 10 observations in the wine dataset # The new observations should only contain the independent variables used in the # model
new_observations = wine.ix[wine.index.isin(range(10)),  independent_variables.columns]
# Predict quality scores based on the new observations' wine characteristics
y_predicted = lm.predict(new_observations)
# Round the predicted values to two decimal placess and print them to the screen
y_predicted_rounded = [round(score, 2) for score in y_predicted] 
print(y_predicted_rounded)