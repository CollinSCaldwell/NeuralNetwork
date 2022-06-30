# NeuralNetwork
From scratch, matrix based fully connected and convolutional networks

This is an old C# project I put together when I first learned about style transfer way back in 2017-2018.
I put together most of it in a few weeks and actually messed up my column / row major ordering of matrices a few times.
I really should have added comments because I had to learned partial derivatives before I even learned what a derivative was at the time

There are quite a few things I would have done different, one being the activation and error rate functions because I would like to make it much easier to add custom functions than it is currently. 

I do enjoy the ease of adding layers to the fully connected network but the convolutional network can be a bit confusing to add layers.
Even though it is confusing, its only because the size of the previous layer has to be accounted for since they need to be shrunk by certain multiples of the length and width. 

Overall I did really enjoy this project as it went to further my exploration in a lot of machine learning techniques like genetic algorithms and even researching one-shot / siamese neural networks. 
