# Deep Learning for predicitng well oil production rate (WOPR) 

I decided to use CNN at first based on 3D nature of the dataset and it has a very simple structure and the error rates indicate that the model is not performing well.
In the next step i was curious if LSTM can capture the pattern of ht flatten data and even though the model is sophisticated, it was not able to perform nearly as good as CNN network.
 Since not of the networks were able to capture the patterns well enough I decided to apply stacking ensemble learning and use both CNN and LSTM and three other ML models. If I had enough time, I would try GANs were I use CNN as actror and LSTM as critic and create a competition between them for lower error rate amounts.
