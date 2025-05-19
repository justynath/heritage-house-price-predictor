# Heritage House Price Predictor - ML

*Image & live link placeholder*

## Overview

The Heritage House Price Predictor is a data-driven project developed to help a client determine accurate sale prices for a set of inherited homes in **Ames, Iowa**. The primary aim is to estimate the total expected value of these properties and to understand how specific features—such as location, size, and condition—impact their market price.

Using a combination of **data analysis**, **machine learning**, and **interactive visualisation**, this project delivers actionable insights through a **Streamlit web application** (deployed via Render or Heroku). The user-friendly interface allows the client to explore predictions, input their own property values, and understand the model’s logic in real time.

The project follows the **CRISP-DM (Cross Industry Standard Process for Data Mining)** methodology. This structured approach ensures that the workflow—from understanding business needs to evaluating predictive models—is transparent and repeatable.

---

## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

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

### What We Found

We tested these three hypotheses using correlation scores and data visualisations. The results were:

* **Larger houses really do sell for more.** Features like `GrLivArea`, `GarageArea`, and `TotalBsmtSF` had strong positive correlations with `SalePrice`.
* **Higher quality makes a big difference.** `OverallQual` was the top predictor, and `KitchenQual` also showed a strong link to price.
* **Newer homes are slightly more expensive.** `YearBuilt` and `YearRemodAdd` had moderate correlations with `SalePrice`, providing some support for this hypothesis.

These findings gave us clear direction when selecting features for the prediction model.

---

## Rationale to Map the Business Requirements to Data Visualisations and ML Tasks

To address the client’s goal of maximising the sale price of the inherited homes, this project translates each business requirement into actionable data analysis and machine learning tasks.

### Business Requirement 1: Understand How House Attributes Influence Sale Price

**Client Expectation:**  
Identify which house features most influence the sale price through clear, informative visualisations.

**Mapped Tasks:**

* Load, clean, and explore the Ames housing dataset.
* Conduct correlation analysis using Pearson and Spearman methods.
* Visualise key relationships (e.g., scatter plots, boxplots, heatmaps).
* Test hypotheses about size, quality, and year built.
* Present results clearly in a format the client can understand.

These correlation results confirmed our initial assumptions about house size, quality, and age.  
They also helped us choose the top features that will be used in the machine learning model:  
`OverallQual`, `GrLivArea`, `KitchenQual`, `GarageArea`, and `TotalBsmtSF`.

### Business Requirement 2: Predict the Sale Price of Inherited Houses and Others in Ames

**Client Expectation:**  
Use a machine learning model to predict the sale prices of the four inherited homes and any future homes.

**Mapped Tasks:**

* Select important features based on correlation and domain knowledge.
* Preprocess and engineer data for ML modeling.
* Train and evaluate regression models (e.g., Linear Regression, Random Forest).
* Assess model performance using R² and MAE.
* Deploy the model via a user-friendly **Streamlit app** to allow real-time prediction.

This structured process ensures that the client's goals are met through both **insightful visual analysis** and a **practical predictive tool**, in line with the **CRISP-DM** methodology.

---

## ML Business Case

### What are we trying to predict?

We want to build a machine learning model that can **predict the sale price of a house** in Ames, Iowa, based on its characteristics (e.g. size, condition, year built, etc.).

The target variable, `SalePrice`, is a **continuous numerical variable**, so we are using a **Regression Model**.

This is a **supervised learning** task because the target variable is already known for the training data.

### What is the goal of the ML system?

To help the client:

* Predict the sale prices of the 4 inherited houses.
* Explore and compare estimated sale prices for other houses in Ames, Iowa.
* Understand which house features have the biggest impact on price.

### What makes the model successful?

**Success Criteria (as agreed with the client):**

* R² score of **at least 0.75** on both the training and test sets.
* Visual inspection of **Actual vs Predicted Sale Price** to confirm that predictions align closely with real prices.

**Failure Criteria:**

* If after 12 months of use, **more than 30% of predictions differ from the real sale price by more than 40%**, the model will be considered expired and due for retraining.
* If R² drops below 0.60 on new data, retraining will also be required.

### What are the model's inputs and outputs?

* **Inputs:** House attribute information such as square footage, quality rating, basement area, kitchen quality, garage size, etc.
* **Output:** A **numeric value** representing the predicted sale price in USD.

### What happens with the model?

* The model is deployed in a **Streamlit web app**.
* Users can interact with sliders and dropdowns to input house features.
* The model makes a prediction instantly, and the app shows the estimated price along with key feature insights.

### Heuristics and training data

* The training data comes from the publicly available **Ames, Iowa housing dataset** on Kaggle.
* It contains ~1500 housing records from the years 1872–2010.
* No sensitive or personally identifiable data is used.

---
---

## Agile Workflow

To manage this project effectively, I used an agile approach based on the **CRISP-DM methodology**, supported by:

* **Epics** and **User Stories** to break the project into actionable goals
* A visual **[Kanban board](https://github.com/users/justynath/projects/11/views/1)** to track progress and maintain focus throughout the project
* **MoSCoW prioritisation** to focus on what matters most

### Epics

* **Business & Data Understanding**  
  Define the business problem and explore the dataset to ensure it supports the ML goal.

* **Data Preparation**  
  Clean and transform the dataset to make it ready for machine learning.

* **Modelling & Evaluation**  
  Build, train, and evaluate models to address the business objective.

* **Deployment & Dashboard**  
  Develop and deploy a Streamlit dashboard to present insights and predictions.

* **Documentation & Review**  
  Document the project comprehensively and ensure it is reproducible and understandable.

### User Stories

1. **Define business objectives and success criteria**
   * Epic: Business & Data Understanding
   * Priority: Must
   * As a data practitioner, I want to define the business goals so that I can align the ML system with the client's needs.

2. **Explore and understand the dataset**  
   * Epic: Business & Data Understanding
   * Priority: Must
   * As a data practitioner, I want to explore the dataset to check if it can answer the business question.

3. **Clean and structure the data**  
   * Epic: Data Preparation
   * Priority: Must
   * As a data practitioner, I want to clean and structure the dataset so that it can be reliably used in modelling.

4. **Engineer new features for modelling**  
   * Epic: Data Preparation
   * Priority: Should
   * As a data practitioner, I want to create new features that improve model performance.

5. **Build and evaluate ML models**  
   * Epic: Modelling & Evaluation
   * Priority: Must
   * As a data practitioner, I want to build and evaluate models so that I can predict house prices accurately.

6. **Improve model with hyperparameter tuning**  
   * Epic: Modelling & Evaluation
   * Priority: Should
   * As a data practitioner, I want to optimise the model using tuning to improve predictive performance.

7. **Build a dashboard for prediction and insights**  
   * Epic: Deployment & Dashboard
   * Priority: Must
   * As a stakeholder, I want a dashboard where I can input house details and get sale price predictions.

8. **Add visualisations and interactivity to dashboard**  
   * Epic: Deployment & Dashboard
   * Priority: Should
   * As a stakeholder, I want visual insights and interactive features to explore data and predictions.

9. **Write comprehensive project documentation**  
   * Epic: Documentation & Review
   * Priority: Must
   * As a reviewer, I want to understand the full project by reading the README file.

10. **Explain and validate project hypothesis**
    * Epic: Documentation & Review
    * Priority: Should
    * As a reviewer, I want to understand the hypothesis and see how it's supported by the model results.

---

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
