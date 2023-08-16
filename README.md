# Chess Game Predictor

## Overview
This project was done in collaboration with Hadi Hammoud. 

* We explored different ways to predict outcomes of chess games given the results of previous ones in this [dataset](https://www.kaggle.com/competitions/chess/data).
* By modeling these games as directed weighted graphs, we first analyze their underlying structures to identify the optimal architecture for our prediction task. 
* We then achieve the best results using an Encoder-Decoder architecture with GraphSAGE convolutional layers along with multiple improvements made to the standard approach.

## Main Architecture
We implemented an encoder-decoder architecture with GraphSAGE [2] convolutional layers. 

### Encoder
* Since white pieces usually have a slight advantage over black pieces, we designed the node embeddings such that each player receives two embeddings depending on the color of the pieces he/she plays with. 
* For the encoder, we first pass the input graph into 4 GraphSAGE layers. We use an Exponential Linear Unit [1] activation function followed by a dropout rate of 0.1.  
* Since the graphs we constructed are directed, the message passing algorithm only propagates data from a player with white pieces to a player with black pieces. 
To mitigate the lack of information passed, we designed a bidirectional model which runs two independent phases of message passing: a forward phase and a backward phase. 
In both phases, we go through the same 4 GraphSAGE layers (so we share the weights in the two directions). The only difference is that in the backward phase, edges are reversed.
We end up with two node embeddings per node. We combine those by simply adding them up and pass this vector into a fully connected layer to get our final node embedding.

### Decoder
* For the decoder, we first get an embedding for every edge by concatenating the corresponding node embeddings of the endpoints. 
* The embedding is then passed into an MLP with a sigmoid function at the end. We end up with one feature per edge (u, v), which we interpret as the probability that u wins against v.

## Training
* We train our model by comparing the edge weights we get from the network with the available edge weights in the graphs.
* We trained our model for 20 epochs, using the Adam optimizer and the MSE loss function.

## Testing
* To test the performance of our model, we masked 300 edges with known weights in the last month (the dataset is divided in months), 
trained our model on the remaining edge weights and then computed the test loss on the masked edges.  
* We compared our results with two baseline methods: matrix completion on the weighted adjacency matrix (implemented with alternating least squares) 
and a simpler Graph Neural Network approach that uses GCN layers instead of GraphSAGE, does not differentiate between white and black pieces and does not propagate information in both edge directions.

## Results
<img width="250" alt="image" src="https://github.com/Charbel-11/Chess-Game-Predictor/assets/61922252/3af238e8-f8fe-4b94-823e-fb5f25bd7bb3">

* Overall, it is clear that Graph Neural Network methods outperformed the matrix completion baseline which only was 29% accurate. 
The standard GCN model performed better with a 62.33% accuracy but still was no match compared to the SAGE Encoder-Decoder model which reached an accuracy of 91%.  
* Note that the SAGE Encoder-Decoder model generates node embeddings, which can be leveraged for various other tasks, such as ranking players.


## References
[1] D.-A. Clevert, T. Unterthiner, and S. Hochreiter, “Fast and accurate deep network learning by exponential linear units (elus),” arXiv preprint arXiv:1511.07289, 2015.  
[2] W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” CoRR, vol. abs/1706.02216, 2017. Available: http://arxiv.org/abs/1706.02216
