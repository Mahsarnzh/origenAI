# Oil Production Rate Prediction Ensemble Framework

## Overview

This repository explores advanced techniques for predicting Well Oil Production Rates (WOPR) using a combination of Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and traditional machine learning models. The ensemble learning approach, which incorporates CNN, LSTM, and three additional ML models, demonstrates superior predictive performance compared to standalone models.

## Custom Model Integration

All the previous models including the ensemble were trained without BHP dataset, in the next step, to enhance the predictive capabilities of the models, I created a custome architecture. This costume model architecture, integrates two CNN layers to accommodate two distinct inputs:

- `Input1`: 500 x (10, 15, 25, 24)
- `Input2`: 500 x (300, 7, 1)

The model structure can be summarized as follows:

Input1 --> CNNLayer 
                    \
                     ---> FCLayer ---> Output
                    /
Input2 --> CNNLayer


## Training

All models were initially trained without utilizing the Bottom Hole Pressure (BHP) dataset. Subsequently, to gauge the impact of incorporating BHP data, the custom model was developed and trained. The addition of BHP data significantly enhances the learning rate of the models.

## Conclusion

This repository provides a comprehensive exploration of various models and their integration through ensemble learning. The custom model's innovative architecture demonstrates the potential for improved predictions by leveraging diverse datasets. Feel free to explore the code and models in the provided notebooks and scripts.



Model: "Costume Model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_cnn1 (InputLayer)     [(None, 300, 7, 1)]          0         []                            
                                                                                                  
 input_cnn2 (InputLayer)     [(None, 10, 15, 25, 24)]     0         []                            
                                                                                                  
 conv2d (Conv2D)             (None, 298, 5, 64)           640       ['input_cnn1[0][0]']          
                                                                                                  
 conv2d_1 (Conv2D)           (None, 10, 13, 23, 64)       13888     ['input_cnn2[0][0]']          
                                                                                                  
 flatten (Flatten)           (None, 95360)                0         ['conv2d[0][0]']              
                                                                                                  
 flatten_1 (Flatten)         (None, 191360)               0         ['conv2d_1[0][0]']            
                                                                                                  
 concatenate (Concatenate)   (None, 286720)               0         ['flatten[0][0]',             
                                                                     'flatten_1[0][0]']           
                                                                                                  
 dense (Dense)               (None, 2100)                 6021141   ['concatenate[0][0]']         
                                                          00                                      
                                                                                                  
 reshape (Reshape)           (None, 300, 7)               0         ['dense[0][0]']               
                                    
