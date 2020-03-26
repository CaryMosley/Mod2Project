# Mod2Project
Module 2 Project

# File Index

AirbnbDropColumns
Notebook to drop text based columns

AirbnbClean
Notebook where we clean our data

AirbnbEDA
Notebook where we explore our data and run hypothesis tests

AirbnbFeatureEng
Notebook where we show a few of the models that we have iterated through and append into a summary df

AirbnbFeatureEngFunction
Notebook which includes a function for rapidly testing multiple feature sets

make_models.py
Function to run various ols/lasso/ridge/elastic net and poly models. When calling the function you can specify, simple, polynomial or polynomial with interactions. Also the degree of the polynomial, the type of regression as well as alpha and cv values. 

FinalNotebook
This notebook contains a run through from EDA to feature and model building. It is supported by the AirbnbEDA notebook for additional EDA and hypothesis testing. AirbnbFeatureEng and AirbnbFeatureEngFunction also go into more depth about various models that were attempted and iterated on before settling on our final functional model.



# Business Case

This linear regression project is based on that of a social media marketing company that helps individuals who operate or want to operate an AirBNB business in San Francisco get up and running.The purpose of this case is to work with prospective AirBNB business clients to help them price their units. To do this we will retrieve data, develop an alogorithm to predict optimal pricing using a robust feature set and then explain why these business drivers matter. The goal is for our client to recieve a fair market price for their unit and to maintain a business relationship for social media promotion. We plan to focus on the millenial age group and those who have units in the most popular property and room types.

# Data Source and Cleaning

We sourced our data from https://www.kaggle.com/jeploretizo/san-francisco-airbnb-listings which is a list of about 8k data points with over 100 columns. The first thing we did was remove most columns that were clearly not going to be useful as features. Examples of these are url, image, host name. From here we were left with around 40 columns that we looked into more before deciding which ones to keep.

Next we proceeded to look at null values, outliers and other attributes of the columns that needed to be cleaned. For certain columns that had very low numbers of missing values we simply dropped those observations. We were missing reviews for about 20% of our values so we imputed the values by grouping by neighborhood and then using the average rating for that neighborhood as the value. Based on our business case we decided to trim some outliers and also certain columns that we didnt believe would be useful. For example where the daily room rate was over $1,000 or minimum number of nights was over two months. As were focused in San Francisco, we dropped the ~20 data points that were outside the city.

We also decide to focus on the major property types found within San Francisco. As we are trying to present a business case to a wide reaching client base we wanted to focus on the largest potential customer base. Below you can see the distribution of property types

<img src="https://github.com/CaryMosley/Mod2Project/blob/master/propertytypecount.png">

# EDA

After our cleaning, trimming outliers and removing null values we ended up with a little under 6000 data points. 

From looking at our cleaned price data we still have a tail to the upside but its signficantly less extreme than before.

<img src="https://github.com/CaryMosley/Mod2Project/blob/master/cleanprice.png">

We next looked at histograms of our data to check to see how the distributions within the features looked. From here we created a heatmap to check for multicollinearity. Maximum and minimum nights had a correlation of almost one so we decided to drop maximum nights. Additionally, beds and accomodates had a very high correlation so we plan to drop beds by the end of our feature engineering process.

<img src="https://github.com/CaryMosley/Mod2Project/blob/master/heatmap.png">

Next we looked at scatter plots of our data to check for a linear relationship between our predictors and our outcome variable. We decided to take a closer look at a few of these potential features to see if we could find relationships. We decided to check whether being a superhost seems to impact price, whether instant booking impacts price and whether a response rate of 100% vs any other response rate matters. Our hypotheses were and we used alpha =.05 for each of them

H_0 Average price for superhost is the same as average price not superhost
H_A Average price for being a super host is not the same as average price for not superhost

<img src="https://github.com/CaryMosley/Mod2Project/blob/master/superhost.png">

H_0 Average price for instant booking is the same as average price for no instant booking
H_A Average price for instant booking is not the same as average price for no instant booking

<img src="https://github.com/CaryMosley/Mod2Project/blob/master/instantbook.png">

H_0 Average price for 100% response rate is the same as average price anything besides 100% response rate
H_A Average price for 100% response rate is higher than the average price anything besides 100% response rate

<img src="https://github.com/CaryMosley/Mod2Project/blob/master/responserate.png">

As the difference between the means for these ended up relatively low we had low cohen's d values for each and ended not being able to reject any of the null hypothesis. As our sample sizes were quite large we approached a power of 1 for each of these tests.

After trimming and imputing our outliers and missing values we created a second heatmap and looked at the scatter plots again. Next up we began our feature engineering process!

# Feature Engineering and Model Building

The first thing we did was create dummy variables for room type, property type and neighborhood. We also took the opportunity to iterate on models and drop features that our business sense indicated would have little predictive power. These included the features we had done hypothesis testing on: superhost, instant booking and response rate. Dropping these didn't increase our RMSE and actually resulted in higher r^2 values so we can be confident they're good drops. After iterating we ended up with a RMSE in the low 80s and an r^2 value in the mid .60s using a lasso regularization on a 2nd order polynomial model. In the jupyter notebook you can see our iterative process involving lasso, ridge and elastic regularization as well as exploring 2nd and 3rd order polynomial models. Below is the residual plot so we can check for homoskedasticity.

<img src="https://github.com/CaryMosley/Mod2Project/blob/master/QQPlot1.png">

After reviewing this QQ plot we can see tailed data. From here we decided to re-evaluate our outliers and ended up deciding to remove the top and bottom 2.5% of data. This corresponded to $600 on the upper end and $55 on the lower. Next we re-ran our model! This resulted in a significantly lower RMSE of 64 with an adjusted r^2 value of .70. We took a look at the QQ plot and altho still slightly tailed it is much more symmetric and significantly less extreme. 

<img src="https://github.com/CaryMosley/Mod2Project/blob/master/QQPlot2.png">

Now that we had a solid baseline model we really wanted to do some feature engineering and reduce our feature set to the ones that make the most logical sense. 

First we removed the neighborhood dummy category as it was 30 columns and we didnt believe the linear regression model would do a good job handling it. Instead we grabbed data from: https://www.bayareamarketreports.com/trend/san-francisco-neighborhood-map and replaced the neighborhood name with the median condo pricing within that area. We believed that this should be a good proxy for a lot of intangible factors such as quality of establishments, crime, walkability. Once we had this we were able to drop latitude and longitude from our feature set as they don't make intuitive sense in a regression model. We began to look at the Geopy library to potentially map airbnb location vs landmarks but due to the time constraints we weren't able to incorporate it. Given more time it would be interesting to see if location vs landmarks or even famous bar/restuarants could have predictive power.

We also examined a higher order polynomial model and found clear overfitting and worse results on our testing set so we abandonded that route. 

We used a combination of a step wise function, f regression to study the p values of our coefficients and finally our business sense of when we could drop features without losing much predictability to end up with the following features list: neighborhood median price, # of people accomdates, # bathrooms,# bedrooms, # guests included in the price, minimum_nights, number of reviews, review scores rating, and type of room and type of property.
       
Our final model is a 2nd order polynomial using a LassoCV. We compared our feature set using lasso, ridge, elastic net and OLS to see which model produced the lowest errors. Below you can see the most impactful of the polynomial coefficients. Although once we get to a 2nd order or higher model the features start to lose logical sense we can still use this to inform our decision making process.

<img src="https://github.com/CaryMosley/Mod2Project/blob/master/pricemovers2.png">

# Business Features

When we examined our coefficients the features that stood out both solely or through interaction as the most impactful were: number of people that are accomodated, # bathrooms, reviews ratings and the property type. When we go back to our clients we can use the fundamental features of their unit: size, number of bathrooms, property type to help the client get a fair market rate. Also as the review score and number of reviews were strongly correlated with price we will work with the operator to optimize their quality and help them maximize the number of reviews they can obtain.

# Future Extensions

Given more time we would want to explore a couple of things. First we would use Geopy to see if we can build a feature based score on proximity to local landmarks or quality bars and restaurants. Additionally we would want to see if we could explore how the number of nights minimum impacts price. There may be extra opportunity for either shorter or longer term pricing if we can determine a strong relationship that matters.
