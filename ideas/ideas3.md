Section 4: Practical Implementation and Parametrization ExamplesTranslating the theoretical framework of two-part models into practice requires familiarity with statistical software packages that can accommodate their unique structure. This section provides concrete examples of how to specify and fit these models in two leading environments: R and Python.4.1. Implementation in RThe R statistical programming language offers several powerful and specialized packages for fitting two-part models, with two ecosystems standing out: gamlss and glmmTMB.4.1.1. The gamlss Framework: Modeling All ParametersThe gamlss (Generalized Additive Models for Location, Scale, and Shape) package, along with its extensions like gamlss.dist and gamlss.inf, provides a comprehensive framework for distributional regression.20 Its core philosophy is to allow every parameter of a chosen distribution to be modeled as a function of covariates, using linear, non-linear, or smooth terms.22For fitting two-part models, the gamlss.inf package provides the gamlssZadj() function, which is designed specifically for zero-adjusted continuous distributions on the interval [0, \infty).53 This function leverages the decomposable nature of the likelihood by internally fitting two separate models: a binomial model for the zero-versus-positive outcome, and a conditional continuous model for the positive values.The syntax allows for explicit and separate formulas for each parameter. For a Zero-Adjusted Gamma (ZAGA) model, where the positive component follows a Gamma distribution, the parameters to be modeled are mu (the mean), sigma (the dispersion parameter, related to the shape), and xi0 (the probability of a zero outcome, \pi).Example Code (Zero-Adjusted Gamma with gamlss):Suppose we are modeling daily precipitation as a function of temperature, elevation, atmospheric pressure, and season.R# Load necessary libraries
library(gamlss)
library(gamlss.inf)

# Fit the zero-adjusted model
# family = "GA" specifies the continuous part is Gamma
# mu.formula, sigma.formula model the parameters of the Gamma distribution
# xi0.formula models the probability of a zero outcome
fit_gamlss <- gamlssZadj(y = precipitation,
                         family = "GA",
                         mu.formula = ~ pb(temperature) + elevation,
                         sigma.formula = ~ pb(temperature),
                         xi0.formula = ~ pressure + as.factor(season),
                         data = my_data)

# View the summary of the fitted model
summary(fit_gamlss)
In this example, the mean (mu) of the Gamma distribution is modeled with a smooth function of temperature (using penalized B-splines, pb()) and a linear term for elevation. The dispersion parameter (sigma) is modeled with a smooth function of temperature, allowing for heteroscedasticity. Crucially, the probability of a dry day (xi0) is modeled separately using pressure and season as predictors. This syntax provides maximum control and transparency in model specification.4.1.2. The glmmTMB Framework: Mixed Models and Zero-InflationThe glmmTMB package is a fast, flexible, and popular tool for fitting generalized linear mixed models (GLMMs).54 It is built on Template Model Builder (TMB) and uses automatic differentiation for efficient estimation. A key feature of glmmTMB is its integrated support for zero-inflated and hurdle models via the ziformula argument.For continuous data, a hurdle model is specified by selecting a continuous distribution for the family argument and providing a separate formula to ziformula.56 It is important to note that ziformula in glmmTMB models the probability of the zero-inflation process, which for a hurdle model corresponds to the probability of being in the zero state. The conditional model formula (the main formula) models the positive outcomes.Example Code (Gamma Hurdle GLMM with glmmTMB):Using the same precipitation data, we can fit a Gamma hurdle model. A key advantage of glmmTMB is the ease with which random effects can be included to account for correlated data, such as multiple measurements from the same weather station.R# Load the library
library(glmmTMB)

# Fit the Gamma hurdle model with a random intercept for station
fit_glmmTMB <- glmmTMB(precipitation ~ temperature + elevation + (1|station),
                       ziformula = ~ pressure + season,
                       family = Gamma(link = "log"),
                       data = my_data)

# View the summary of the fitted model
summary(fit_glmmTMB)
The output of summary(fit_glmmTMB) will present two distinct sets of coefficients: one for the "Conditional model" (the Gamma part for positive rainfall) and one for the "Zero-inflation model" (the logistic part for the zero hurdle). This example demonstrates how to simultaneously model the fixed effects, account for zero-inflation, and incorporate the nested structure of the data (observations within stations) in a single, unified command.4.2. Implementation in PythonWhile Python's statsmodels library offers excellent, integrated functions for zero-inflated count models (e.g., ZeroInflatedPoisson) 14, it does not currently provide a single-function solution for two-part/hurdle models with a continuous response. However, the decomposable nature of the two-part model's likelihood function makes a manual, two-step implementation both straightforward and statistically valid. This workaround is an essential technique for practitioners using Python.The process involves fitting the two model components separately and then combining their predictions for inference.Step-by-Step Python Guide (using statsmodels):Step 1: Data PreparationFirst, prepare the data by creating a binary indicator variable that represents the hurdle.Pythonimport pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Assuming 'my_data' is a pandas DataFrame with the relevant columns
# Create a binary indicator for the hurdle (1 if positive, 0 if zero)
my_data['is_positive'] = (my_data['precipitation'] > 0).astype(int)
Step 2: Fit the Hurdle Model (Part 1: Logistic Regression)Use statsmodels to fit a logistic regression model to the binary indicator. This models the probability of crossing the zero hurdle (i.e., having a positive precipitation value).Python# Fit a logistic regression model for the probability of a non-zero outcome
# The formula uses the covariates hypothesized to drive occurrence
hurdle_model = smf.logit('is_positive ~ pressure + C(season)', data=my_data).fit()
print(hurdle_model.summary())
Step 3: Filter Data for the Continuous PartCreate a new DataFrame that contains only the observations with positive precipitation values.Python# Filter the dataset to include only positive precipitation amounts
positive_data = my_data[my_data['is_positive'] == 1].copy()
Step 4: Fit the Continuous Model (Part 2: Gamma GLM)Fit a GLM to the filtered positive data. Here, we use a Gamma distribution with a log link, which is a common choice for skewed, positive data.Python# Fit a Gamma GLM to the positive values
# The formula uses covariates hypothesized to drive intensity
continuous_model = smf.glm('precipitation ~ temperature + elevation',
                           data=positive_data,
                           family=sm.families.Gamma(link=sm.families.links.log())).fit()
print(continuous_model.summary())
Step 5: Combine and InterpretThe analysis is now complete, with two separate fitted models.Interpretation: The coefficients from hurdle_model explain the factors influencing the odds of a day being wet versus dry. The coefficients from continuous_model explain the factors influencing the average rainfall amount on days that are already wet.Prediction: To get the overall expected precipitation for a new set of covariate values, you must combine the predictions from both models:E = P(Y > 0) * EThis is calculated by multiplying the predicted probability of a positive outcome from the logistic model by the predicted mean from the Gamma model.Python# Example prediction for new data
# new_covariates = pd.DataFrame(...)
prob_positive = hurdle_model.predict(new_covariates)
mean_if_positive = continuous_model.predict(new_covariates)
expected_precipitation = prob_positive * mean_if_positive
This two-step approach provides a robust and flexible method for fitting two-part models in Python, fully capturing the logic of the underlying statistical framework.