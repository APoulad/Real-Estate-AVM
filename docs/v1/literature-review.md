# Literature Review on AVMs

## Introduction

Automated Valuation Models (AVMs) have become essential in real estate appraisal due to their efficiency and ability to process large datasets. Traditional methods, like the Hedonic Price Model (HPM), typically use linear regression to estimate property values based on factors like the number of bedrooms, bathrooms, or nearby amenities. While useful, these methods need to be more complex and identify non-linear relationships between some of these variables. Recent advancements in ML, particularly with Convolutional Neural Networks (CNNs), allow us to integrate additional data types like images and location data, which enhances the accuracy of the model and gives a clearer picture of property value.

This review covers two key studies—one that uses CNNs for factoring geographic variation into AVMs and another that combines CNNs with ensemble learning for image-based appraisals. Both studies demonstrate how machine learning can improve AVM accuracy by bringing in visual and spatial data.

## Using CNNs for Geographic Variation in Residential Valuation

The first study, "Machine Learning Approach to Residential Valuation," blends CNNs with fully connected networks to capture both structural property data and geographic information. One of the big issues with older models, like HPMs, is that they can struggle with geographic factors due to spatial autocorrelation and multicollinearity, which mess up the accuracy of the valuations. With the three most important factors of real estate being location, location, and location, this lack of geographic factors poses an inherent flaw in traditional models. This study proposes a fused model that uses a CNN to process multi-layer geographical data (like the distribution of amenities, demographic details, and land control) and a Multilayer Perceptron (MLP) to handle structured data such as the number of bedrooms, bathrooms, and area size.

What makes this approach stand out is the direct use of geographical layers as input for the CNN, which preserves the spatial information without reducing it to numerical or categorical variables. The study demonstrates that the fused model achieved more accurate predictions compared to the baseline model's, showing that incorporating geographical data using CNNs reduces spatial information loss and improves prediction accuracy.

## Image-Based Real Estate Appraisal with CNNs and Ensemble Learning

The second study, "Image-Based Real Estate Appraisal Using CNNs and Ensemble Learning," takes a different angle by focusing on a property’s appearance. The authors argue that AVMs traditionally rely too heavily on structured data (like square footage or room counts) and ignore how much a property’s look impacts its value (Kumkar, 2021). Using CNNs to evaluate property images, they capture key features like architectural style, condition, and visual appeal that often play a big role in a property’s worth.

In addition to the CNN, the study uses ensemble learning models—Random Forest, Gradient Boosting, and Extreme Gradient Boosting (XGBoost)—to combine the extracted visual features with structured data (e.g., square footage, number of rooms) for price estimation. The results show that the XGBoost model, which integrates visual features, achieved the best performance with a MAPE score of 9.86%, compared to 10.33% for the XGBoost model without image data (Kumkar, 2021). This confirms the hypothesis that integrating visual features using CNNs significantly enhances the performance of AVMs.

## Comparison of Techniques and Relevance to the Proposed Model

Both studies show that combining structured, visual, and geographic data makes AVMs more accurate. However, each takes a different route to get there.

- **Geographic Data and CNNs**: The first study demonstrates that traditional models don’t handle location data well, especially when it comes to spatial relationships. Using CNNs to directly process multi-layer geographic data ensures that important details aren’t lost. This aligns with the goal of creating a more comprehensive AVM that integrates visual, spatial, and structured data.

- **Visual Data and CNNs/Ensemble Learning**: The second study emphasizes the importance of visual appeal, a factor often left out in AVMs. By scoring property images with CNNs, the model picks up on aesthetic qualities that can drive market value. Using ensemble learning models like XGBoost further enhances accuracy by combining these visual elements with structured data.

## Proposed Model and Anticipated Improvements

The AVM model we propose takes the best of both worlds by combining structured, visual, and geographic data into one framework. We plan to use CNNs for both visual and spatial data, and then combine this with the traditional property attributes using something like XGBoost or another sophisticated regression model.

We expect the following improvements from this approach:

- **Less Information Loss**: Directly integrating visual and spatial data using CNNs minimizes the information loss that occurs when reducing these data types to simpler numeric or categorical formats. This approach can capture complex relationships and interactions between features that traditional models may overlook.

- **Higher Accuracy**: Both studies show that mixing multiple types of data can significantly lower prediction errors. By applying advanced machine learning techniques, we expect our model to outperform those that only use structured data.

- **Better Handling of Complex Relationships**: CNNs and ensemble models like XGBoost will allow us to model non-linear relationships more effectively, making the predictions much more reliable.

## Conclusion

In conclusion, the integration of multiple data types using CNNs and ensemble learning represents a promising advancement in the development of AVMs. The proposed model, which combines structured data, visual features, and geographical information, should provide a significant improvement in prediction accuracy and offer a more holistic view of property value. Future work should focus on optimizing the model's architecture and testing its performance across diverse real estate markets to ensure its generalizability and robustness.

## Works Cited

Lee, Hojun, et al. "Machine Learning Approach to Residential Valuation: A Convolutional Neural Network Model for Geographic Variation." _The Annals of Regional Science_, vol. 72, 2024, pp. 579–599. Springer, https://doi.org/10.1007/s00168-023-01212-7.

Kumkar, Prathamesh Dnyanesh. "Image-Based Real Estate Appraisal Using CNNs and Ensemble Learning." 2021. _Master's Projects_. San Jose State University, https://doi.org/10.31979/etd.km4q-65hg.
