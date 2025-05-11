# Heritage House Price Predictor - ML

*Image & live link placeholder*

## Overview

The Heritage House Price Predictor is a data-driven project developed to help a client determine accurate sale prices for a set of inherited homes in **Ames, Iowa**. The primary aim is to estimate the total expected value of these properties and to understand how specific features—such as location, size, and condition—impact their market price.

Using a combination of **data analysis**, **machine learning**, and **interactive visualisation**, this project delivers actionable insights through a **Streamlit web application** (deployed via Render or Heroku). The user-friendly interface allows the client to explore predictions, input their own property values, and understand the model’s logic in real time.

The project follows the **CRISP-DM (Cross Industry Standard Process for Data Mining)** methodology. This structured approach ensures that the workflow—from understanding business needs to evaluating predictive models—is transparent and repeatable.

---

## Dataset Content

The dataset is sourced from **[Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data)** and has been adapted to fit a fictional real-world scenario in which predictive analytics is used to support a property decision.

The dataset contains approximately **1,460 housing records** from Ames, Iowa, with information on home features (e.g. floor area, year built, garage) and the corresponding **sale price**. The homes range in build year from **1872 to 2010**.

Here are a few key variables:

| Variable         | Meaning                                              | Range/Values                         |
|------------------|------------------------------------------------------|--------------------------------------|
| `GrLivArea`      | Above ground living area (sq ft)                    | 334 - 5642                           |
| `TotalBsmtSF`    | Total basement area (sq ft)                         | 0 - 6110                             |
| `GarageArea`     | Size of garage (sq ft)                              | 0 - 1418                             |
| `OverallQual`    | Overall quality of materials and finish             | 1 (Very Poor) - 10 (Very Excellent) |
| `YearBuilt`      | Year the house was originally built                 | 1872 - 2010                          |
| `YearRemodAdd`   | Year of last renovation                             | 1950 - 2010                          |
| `SalePrice`      | Sale price of the property                          | \$34,900 - \$755,000                |

Additional features include porch size, lot area, kitchen quality, and garage details.

---

## Business Requirements

Your friend has inherited four homes in Ames, Iowa, from her late great-grandfather. While she understands property values well in her home state, she is concerned that relying on that knowledge in a different location could lead to **inaccurate appraisals**.

She found a **public dataset of Ames house sales** and has asked you, as a good friend with data skills, to help her make better-informed decisions about selling the inherited properties.

### Summary of Client Requirements

1. **Correlation Analysis:**  
   The client wants to understand how house features correlate with sale price. She expects **data visualisations** to help identify which features (e.g., size, age, quality) are most influential.

2. **Price Prediction Model:**  
   The client wants to **predict the sale price** of her four inherited homes and be able to predict prices for **other homes** in Ames using a trained machine learning model.

---

## Hypothesis and How to Validate (Simple Version)

1 - We suspect that larger houses tend to sell for higher prices.  
A correlation study between `GrLivArea` (above-ground living area) and `SalePrice` will help investigate this.

2 - We believe that houses with higher quality ratings are more valuable.  
A correlation study between `OverallQual` and `SalePrice` will help investigate this.

3 - We suspect that newer or recently renovated houses sell for more.  
A correlation study between `YearBuilt`, `YearRemodAdd`, and `SalePrice` will help investigate this.

---

## Hypotheses and How to Validate Them (Expanded Version)

To help the client understand what drives house prices in Ames, Iowa, I formulated three simple hypotheses based on initial observations and domain knowledge. These are designed to be tested through data visualisation and correlation analysis.

### Hypothesis 1: Bigger Houses Sell for More

**H0 (Null Hypothesis):** There is no significant relationship between house size and sale price.  
**H1 (Alternative Hypothesis):** Larger houses (e.g., higher above-ground living area) tend to have higher sale prices.

**How to Validate:**

- Use a scatter plot to visualise `GrLivArea` vs `SalePrice`.
- Calculate correlation coefficients (Pearson and Spearman).

---

### Hypothesis 2: Higher Quality Means Higher Price

**H0:** There is no relationship between the overall quality rating and sale price.  
**H1:** Houses with higher `OverallQual` tend to sell for more.

**How to Validate:**

- Use a boxplot to compare `OverallQual` levels against `SalePrice`.
- Calculate correlation between `OverallQual` and `SalePrice`.

---

### Hypothesis 3: Newer or Renovated Homes Are Worth More

**H0:** The construction year or year of renovation has no effect on sale price.  
**H1:** Newer houses or recently renovated homes tend to sell for higher prices.

**How to Validate:**

- Create scatter plots for `YearBuilt` and `YearRemodAdd` vs `SalePrice`.
- Compare average sale prices by decade.
- Perform correlation analysis to assess significance.

---

## Rationale to Map the Business Requirements to Data Visualisations and ML Tasks

To address the client’s goal of maximising the sale price of the inherited homes, this project translates each business requirement into actionable data analysis and machine learning tasks.

### Business Requirement 1: Understand How House Attributes Influence Sale Price

**Client Expectation:**  
Identify which house features most influence the sale price through clear, informative visualisations.

**Mapped Tasks:**

- Load, clean, and explore the Ames housing dataset.
- Conduct correlation analysis using Pearson and Spearman methods.
- Visualise key relationships (e.g., scatter plots, boxplots, heatmaps).
- Test hypotheses about size, quality, and year built.
- Present results clearly in a format the client can understand.

---

### Business Requirement 2: Predict the Sale Price of Inherited Houses and Others in Ames

**Client Expectation:**  
Use a machine learning model to predict the sale prices of the four inherited homes and any future homes.

**Mapped Tasks:**

- Select important features based on correlation and domain knowledge.
- Preprocess and engineer data for ML modeling.
- Train and evaluate regression models (e.g., Linear Regression, Random Forest).
- Assess model performance using R² and MAE.
- Deploy the model via a user-friendly **Streamlit app** to allow real-time prediction.

This structured process ensures that the client's goals are met through both **insightful visual analysis** and a **practical predictive tool**, in line with the **CRISP-DM** methodology.

## ML Business Case

* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

## Dashboard Design

* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)

## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment

### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the .python-version Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)

* In case you would like to thank the people that provided support through this project.
