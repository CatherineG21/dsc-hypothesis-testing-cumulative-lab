# Hypothesis Testing - Cumulative Lab

## Introduction

In this cumulative lab, you will use pandas to clean up a dataset and perform some EDA, then perform statistical tests and interpret their results in order to answer some business questions.

## Objectives

You will be able to:

* Practice using a data dictionary
* Practice using pandas to prepare data for statistical analysis
* Practice identifying an appropriate statistical test for a given question
* Practice defining the null and alternative hypotheses
* Practice executing statistical tests and interpreting their results

## Your Task: Analyzing Health Trends

![tissues and tea mug](images/sick_day.jpg)

Photo by <a href="https://unsplash.com/@kellysikkema?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Kelly Sikkema</a> on <a href="/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

### Business Understanding

Flatiron Health Insurance (FHI) is a growing private healthcare insurance provider founded on the premise that using data and analytics can improve the health insurance industry by providing better care and offerings to its patients. Every year, the Center for Disease Control (CDC) conducts surveys to understand the latest demographic, health, and fitness trends. You have been tasked with analyzing the recently published results of the 2017-2018 survey and providing your recommendations back to the Chief Analytics Officer and Chief Marketing Officer. You have been assigned the task of taking a first look at the data and beginning to answer several key questions:
											 								
1. How does health status, represented by average number of days with bad physical health in the past month (`PHYSHLTH`), differ by state?					 								
2. Digging deeper into the data, what are some factors that impact health (demographics, behaviors, etc.)?	

### Data Understanding

To get you started, the IT department formatted the data set into a tab delimited text file for only NY, NJ, and CT (FHI’s primary markets) called `case_study.csv`.

There is also a PDF data dictionary called `data_dictionary.pdf`, which explains the meanings of the features and codes contained in this dataset.

Both files are located in the `data/` directory of this repository.

Prior to each statistical test, you will need to perform some data preparation, which could include:

* Filtering out rows with irrelevant values
* Transforming data from codes into human-readable values
* Binning data to transform it from numeric to categorical
* Creating new columns based on queries of the values in other columns

For steps 2-5, you will need to select and execute an appropriate statistical test. Recall these tests we have learned so far:

1. [Chi-squared test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html): used for comparing a categorical feature against a categorical feature, to determine whether they are independent
2. [t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html): used for comparing two categories of a numeric feature, to determine whether their means are the same across categories
3. [ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.anova_lm.html#statsmodels.stats.anova.anova_lm): used for comparing more than two categories of a numeric feature, to determine whether their means are the same across categories

### Requirements

#### 1. Prepare `PHYSHLTH` Data for Analysis

Using the data dictionary, ensure that you understand the meaning of the `PHYSHLTH` column. Then clean the data so that only valid records of `PHYSHLTH` remain.

#### 2. Describe the Distribution of Health Status by State

Does health status (`PHYSHLTH`) differ by state (`STATE_`)? If so, by how much, and is it statistically significant?

#### 3. Describe the Relationship between Health Status and Home Ownership Status

Does health status (`PHYSHLTH`) differ between home owners and renters (`RENTHOM1`)? If so, by how much, and is it statistically significant?

#### 4. Describe the Relationship between Chronic Sickness and Nicotine Use

Does chronic sickness (`PHYSHLTH >= 15`) differ based on nicotine use (various columns)? If so, by how much, and is it statistically significant?

#### 5. Choose Your Own Question

Thinking about the business case, what is another question that might be useful to answer? Perform all analysis steps to answer this question.

## 1. Prepare `PHYSHLTH` Data for Analysis

In the cells below, we include the relevant imports and load the data into a dataframe called `df`:


```python
# Run this cell without changes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```


```python
# Run this cell without changes
df = pd.read_csv("data/case_study.csv", index_col=0, low_memory=False)
df
```

Our main column of interest is called `PHYSHLTH`. We display summary statistics and plot a distribution below:


```python
# Run this cell without changes
df['PHYSHLTH'].describe()
```


```python
# Run this cell without changes
fig, ax = plt.subplots()
ax.hist(df["PHYSHLTH"], bins="auto")
ax.set_xlabel("PHYSHLTH")
ax.set_ylabel("Count")
ax.set_title("Distribution of PHYSHLTH in Raw Data");
```

This feature is supposed to represent the number of days with bad physical health out of the past 30 days. Do you see anything wrong with what is displayed above? Explain.


```python
# Replace None with appropriate text
"""
If the values represent days out of the past 30, they should range from 0 to 30. However, many fall in the 70–90 range, making the mean invalid since it exceeds the maximum possible value of 30.

"""
```

Look in the data dictionary, page 17, to understand what is happening with these values. Then edit the cell below so:

* The records where the `PHYSHLTH` value label is `None` are converted to 0
* The records where the `PHYSHLTH` value label is `Number of days` are kept as-is
* All other records are dropped (i.e. records with `Don't know/Not sure`, `Refused`, and `Not asked or Missing` value labels for `PHYSHLTH` are dropped)


```python
# Code 88 means None, replace it with 0
df.loc[df["PHYSHLTH"]==88, "PHYSHLTH"] = 0

# Now, only keep records where PHYSHLTH is <= 30
# (making a copy to avoid future SettingWithCopyWarning messages)
df = df[df["PHYSHLTH"] <= 30].copy()

df```

Run the code below to ensure you have the correct, cleaned dataframe:


```python
# Run this cell without changes

# We should have fewer rows, the same number of columns
assert df.shape == (33747, 358)

# The maximum value in this column should now be 30
assert df["PHYSHLTH"].max() == 30.0
```

Now we can look at the same descriptive information on our cleaned data:


```python
# Run this cell without changes
df['PHYSHLTH'].describe()
```


```python
# Run this cell without changes
fig, ax = plt.subplots()
ax.hist(df["PHYSHLTH"])
ax.set_xlabel("PHYSHLTH")
ax.set_ylabel("Count")
ax.set_title("Distribution of PHYSHLTH in Cleaned Data");
```

That looks a lot more reasonable. Let's move on to the next step.

## 2. Describe the Distribution of Health Status by State

As mentioned previously, this dataset only includes data from three states.


```python
# Run this cell without changes
df["_STATE"].value_counts()
```

Look in the data dictionary, pages 2-3, to determine which states map onto which codes. Then replace the numbers with strings representing the state names.


```python
df["_STATE"].replace({36: "New York", 34: "New Jersey", 9: "Connecticut"}, inplace=True)
df
```

Below, we check the values:


```python
# Run this cell without changes

# Shape should be the same
assert df.shape == (33747, 358)

# Values in state column should be changed
assert sorted(list(df["_STATE"].value_counts().index)) == ['Connecticut', 'New Jersey', 'New York']
```


```python
# Run this cell without changes

ny = df.loc[df["_STATE"] == "New York", "PHYSHLTH"]
nj = df.loc[df["_STATE"] == "New Jersey", "PHYSHLTH"]
ct = df.loc[df["_STATE"] == "Connecticut", "PHYSHLTH"]

fig, ax = plt.subplots(figsize=(15, 6))

ax.hist(
    x=[ny, nj, ct],
    label=["New York", "New Jersey", "Connecticut"],
    bins=range(32),
    align="left"
)

ax.set_xlabel("PHYSHLTH")
ax.set_ylabel("Count")
ax.set_title("Distribution of PHYSHLTH by State")

ax.legend(title="State");
```

Looking at the plot above, does the distribution seem to differ by state?

(Just answer based on a visual inspection; we will do the statistical assessment next.)


```python
# Replace None with appropriate text
"""
Not exactly—the states appear to have roughly the same distribution of unhealthy days.
"""
```

For the statistical test, we will be comparing the *means* of `PHYSHLTH` across states, as a representation of the overall distribution. In other words, when operationalizing the question **does `PHYSHLTH` differ by state?** we want to answer that in terms of the mean `PHYSHLTH`.

Let's look at those means:


```python
# Run this cell without changes
df.groupby("_STATE")["PHYSHLTH"].mean()
```

You likely noted that the overall distribution looked about the same, but these means are different. We have a range from Connecticut with about 3.7 days of bad health to New Jersey with about 4.4 days. But is that difference statistically significant?

Identify which of the statistical tests you have learned is the most appropriate for this question, and why. Make sure you mention what kinds of variables are being compared (numeric and categorical), and how many categories there are.


```python
# Replace None with appropriate text
"""
ANOVA is the best choice because we are comparing a numeric variable (PHYSHLTH) across three categories. A t-test would work for two categories, but with three, ANOVA is more appropriate
"""
```

Now, identify the null and alternative hypotheses:


```python
# Replace None with appropriate text
"""
Null hypothesis: the means of PHYSHLTH are the same across states

Alternative hypothesis: the means of PHYSHLTH are not the same across states
"""
```

In the cell below, we set up and execute the statistical test for you. If this doesn't match your previous answer about which test to perform, look at the solution branch to understand why this is the appropriate test.


```python
# Run this cell without changes
import statsmodels.api as sm
from statsmodels.formula.api import ols

formula = 'PHYSHLTH ~ C(_STATE)'
lm = ols(formula, df).fit()
sm.stats.anova_lm(lm)
```

Interpret the results of this statistical test below. What is the calculated p-value? Were we able to reject the null hypothesis at an alpha of 0.05? What does this say about how `PHYSHLTH` varies by state? What recommendations would you make to the business?


```python
# Replace None with appropriate text
"""
We rejected the null hypothesis since the p-value (PR(>F)) is well below 0.05. Although state averages are close (around 3–4 unhealthy days per month), the large sample size shows a statistically significant difference. Therefore, the business should further explore why Connecticut reports the best health and New Jersey the worst

"""
```

With that section wrapped up, let's move on to the next step.

## 3. Describe the Relationship between Health Status and Home Ownership Status

This time, we want to categorize respondents by demographic information: specifically, we'll look at whether or not they own their home.

Once again, this will require some data preparation. The variable of interest is contained in the `RENTHOM1` column. Currently the values look like this:


```python
# Run this cell without changes
df["RENTHOM1"].value_counts()
```

In the cell below, modify `df` so that we have dropped all records where the `RENTHOM1` value label is neither `Own` nor `Rent`, and we have replaced the numeric codes with `Own` and `Rent` respectively. You can find more information about codes on page 33 of the data dictionary.


```python
# Replace codes with labels
df["RENTHOM1"].replace({1: "Own", 2: "Rent"}, inplace=True)

# Only keep records with those labels
df = df[df["RENTHOM1"].isin(["Own", "Rent"])].copy()

df
```


```python
# Run this cell without changes
df["RENTHOM1"].value_counts()
```

Below, we check that this was done correctly:


```python
# Run this cell without changes

# Number of rows should be smaller again
assert df.shape == (31934, 358)

# Only two values should be present in this column
assert sorted(list(df["RENTHOM1"].value_counts().index)) == ['Own', 'Rent']
```

Now, similar to the previous step, create a plot that shows the distribution of `PHYSHLTH` for those who own vs. rent their homes, including appropriate axis labels and legend. Because there is more of an imbalance of categories this time (more than twice as many owners as renters, compared to nearly-even numbers from New York, New Jersey, and Connecticut), make sure you add the argument `density=True`, so that the y-axis shows the density (proportion) rather than the count.


```python
own = df.loc[df["RENTHOM1"] == "Own", "PHYSHLTH"]
rent = df.loc[df["RENTHOM1"] == "Rent", "PHYSHLTH"]

fig, ax = plt.subplots(figsize=(15, 6))

ax.hist(
    x=[own, rent],
    label=["Own", "Rent"],
    bins=range(32),
    align="left",
    density=True
)

ax.set_xlabel("PHYSHLTH")
ax.set_ylabel("Proportion")
ax.set_title("Distribution of PHYSHLTH by Home Ownership")

ax.legend(title="Own or Rent Home");```

Then run this code to find the averages:


```python
# Run this cell without changes
df.groupby("RENTHOM1")["PHYSHLTH"].mean()
```

Now, interpret the plot and averages. Does it seem like there a difference in the number of unhealthy days between those who rent their homes and those who own their homes? How does this compare to the distributions by state?


```python
# Replace None with appropriate text
"""
Although the distributions are similar, renters report more unhealthy days than homeowners. Owners more often report 0 unhealthy days, while renters report higher values overall, with mean unhealthy days of about 3.5 vs. 5.2.

"""
```

Now, choose and execute an appropriate statistical test. Make sure you describe why you chose the test, the null and alternative hypotheses, and what the result of the test means.


```python
import scipy.stats as stats

ttest_pvalue = stats.ttest_ind(rent, own, equal_var=False).pvalue / 2
print("t-statistic p-value:", ttest_pvalue)
```
"""
Since the p-value is far below 0.05, we reject the null hypothesis and conclude that renters report more unhealthy days than homeowners. Further investigation could explore whether this difference is due to income levels or issues related to renting conditions.
"""

## 4. Describe the Relationship between Chronic Sickness and Nicotine Use

Once again, this will require some preparation before we can run the statistical test. Create a new column `NICOTINE_USE` with 1 representing someone who uses or has used nicotine in some form, and 0 representing someone who hasn't.

We define nicotine use as:

* Answered `Yes` to the `SMOKE100` question (*Have you smoked at least 100 cigarettes in your entire life?*, page 43), OR
* Answered `Every day` or `Some days` to the `USENOW3` question (*Do you currently use chewing tobacco, snuff, or snus every day, some days, or not at all?*, page 46), OR
* Answered `Yes` to the `ECIGARET` question (*Have you ever used an e-cigarette or other electronic vaping product, even just one time, in your entire life?*, page 46)

If a record matches one or more of the above criteria, `NICOTINE_USE` should be 1. Otherwise, `NICOTINE_USE` should be 0. Go ahead and keep all of the "Don't know" or "Refused" answers as 0.


```python
# Set everything to 0 initially
df["NICOTINE_USE"] = 0

# Make a mask to select the relevant values
# (this separate variable is not necessary
# but helps with readability)
mask = (
    # Has smoked at least 100 cigarettes
    (df["SMOKE100"] == 1) |
    # Uses chewing tobacco/snuff/snus every day or some days
    (df["USENOW3"] == 1) |
    (df["USENOW3"] == 2) |
    # Has smoked an e-cigarette
    (df["ECIGARET"] == 1)
)

# Set values to 1 where the mask condition is true
df.loc[mask, "NICOTINE_USE"] = 1
# Look at the distribution of values
df["NICOTINE_USE"].value_counts(normalize=True)
```

This time, let's treat health status as a categorical variable. We'll say that a "chronically sick" person is a person who reports that their physical health was not good for 15 or more out of the past 30 days. (This is a simplification but it will work for this analysis.)

In the cell below, create a new column of `df` called `CHRONIC`, which is 0 for records where `PHYSHLTH` is less than 15, and 1 for records where `PHYSHLTH` is 15 or more.


```python
df["CHRONIC"] = (df["PHYSHLTH"] >= 15).apply(int)

# Alternatives:

# Helper function:
# def is_chronic(record):
#     if record["PHYSHLTH"] >= 15:
#         return 1
#     else:
#         return 0
# df["CHRONIC"] = df.apply(is_chronic, axis=1)

# NumPy:
# df["CHRONIC"] = np.where(df["PHYSHLTH"] >= 15, 1, 0)
# View the distribution of the newly-created column
df["CHRONIC"].value_counts()
```

Now we can view the crosstabs for these two categorical variables, as well as display their distributions:


```python
# Run this cell without changes
contingency_table = pd.crosstab(index=df["CHRONIC"], columns=df["NICOTINE_USE"])
contingency_table
```


```python
# Run this cell without changes

no_nicotine_use = df.loc[df["NICOTINE_USE"] == 0, "CHRONIC"]
nicotine_use = df.loc[df["NICOTINE_USE"] == 1, "CHRONIC"]

fig, ax = plt.subplots()

ax.hist(
    x=[no_nicotine_use, nicotine_use],
    label=["No Nicotine Use", "Nicotine Use"],
    bins=[0,1,2],
    align="left"
)

ax.set_ylabel("Count")
ax.set_xticks([0,1])
ax.set_xticklabels(["Not Chronically Sick", "Chronically Sick"])
ax.set_title("Distribution of Chronic Sickness by Nicotine Use")

ax.legend();
```

Once again, it appears that there is a difference in health outcomes between these categories. In the cell below, select an appropriate statistical test, describe the null and alternative hypotheses, execute the test, and interpret the result.


```python
# Reusing the contingency_table created above
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("chi-squared p-value:", p)

results_table = pd.concat([pd.DataFrame(expected), contingency_table])
results_table.columns.name = "NICOTINE_USE"

results_table.index = ["0 (expected)", "1 (expected)", "0 (actual)", "1 (actual)"]
results_table.index.name = "CHRONIC"
results_table
```
"""
The very small p-value allows us to reject the null hypothesis, showing a statistically significant relationship between nicotine use and chronic sickness. 
Although non-users are the larger group overall, nicotine users make up the majority of those with chronic sickness. 
This suggests a strong association, though not necessarily causation, between nicotine use and both long-term and short-term health problems.
"""
## 5. Choose Your Own Question

Now that you have investigated physical health and chronic sickness and their relationships with state, home ownership, and nicotine use, you will conduct a similar investigation with variables of your choosing.

Select an independent variable based on looking at the information in the data dictionary, and perform any additional transformations needed to use it in an analysis. Then set up, execute, and interpret a statistical test that addresses the relationship between that independent variable and `PHYSHLTH`.

(There is no solution branch for this question, and feel free to move on if you have already spent more than 1.5 hours on this lab.)


```python
Choose an independent variable (e.g., income level), clean the data, and run an ANOVA since PHYSHLTH is numeric and the independent variable is categorical. 
If the p-value is below 0.05, conclude that the variable significantly affects physical health; if not, 
there’s no significant relationship```

## Conclusion

Congratulations, another cumulative lab down! In this lab you practiced reading a data dictionary, performing various data transformations with pandas, and executing statistical tests to address business questions.
