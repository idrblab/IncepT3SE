# IncepT3SE
`License: MIT `  `TensorFlow 2.0`  `Paper: under submission`  `DOI: waiting`

Type III secreted effector (T3SE) identification with deep inception architecture and rationally designed dataset.



**Figure 1.** The negative dataset in this studied consisted of non-T3SEs from several bacterial species, including *L. pneumophila*, *L. longbeachae*, *L. drancourtii*, *C. burnetii*, *R. grylli* & *E. coli*. (A) The proportion of each species in the dataset was no more than 20%, with *E. coli* being one of the most abundant at 19.8%. (B) The two-dimensional scatter plot distribution of non-T3SE from multi-bacterial species. All sequences were encoded through their composition, transition, and distribution (CTD) features and then visualized using t-SNE algorithm. Employing the non-T3SEs of all 6 species took up more protein space (on the right side of the plot) than only using that of *E. coli* (on the left side).

<img src="https://github.com/nongchao-er/IncepT3SE/blob/main/plt_fig/Figure1.png" width="100%">

**Figure 2.** The workflow and architecture of the IncepT3SE model adopted in this study. This model directly takes sequence of protein amino acids as input. After sequence one-hot encoding, the input tensors expand into 3 parallel streams. Each stream undergoes a convolutional layer that performs 1024 convolutions, followed by max pool dimension reduction and batch normalization. Two key inception layers are then concatenated by 3 convolutional layers of different kernel size, followed by another max pool layer. After global max pooling, the tensors are squeezed into one flatten vector of 576 dimensions. Through 3 fully connected layer combined with random dropout and batch normalization, the probability of T3SE is finally predicted.

<img src="https://github.com/nongchao-er/IncepT3SE/blob/main/plt_fig/Figure2.png" width="100%">

**Figure 3.** The effect and performance comparison of different methods on benchmark independent test. (A) Independent dataset 1 includes 83 T3SEs & 14 non-T3SEs, which are screened and identified from the a plant pathogen *P.syringae*. The IncepT3SE outperforms other methods nearly in all assessment criteria. (B) Independent dataset 2 includes 108 T3SEs & 108 non-T3SEs, which are extracted from literature by Bastion3. The Bastion3 and IncepT3SE performs best under most assessment criteria. (C) The probability values of predicted to be T3SE by each model on independent dataset 1. The broken line of IncepT3SE is largely surrounding the others indicating larger probability value in most samples. (D) The probability values of predicted to be T3SE on independent dataset 2. SE: sensitivity; SP: specificity; ACC: accuracy; PRE: precision; MCC: Matthews correlation coefficient.

<img src="https://github.com/nongchao-er/IncepT3SE/blob/main/plt_fig/Figure3.png" width="100%">

**Figure 4.** The effect and performance comparison of different methods on 69 newly identified true T3SEs. All these true T3SEs were identified after September 2022 which were not existed in the training data. (A) The violin diagram of the effects of each method in which IncepT3SE showed the highest median and the lowest trailing. (B) The probability values of predicted to be T3SE where the broken line of IncepT3SE is largely surrounding the others, indicating its best generalization ability.

<img src="https://github.com/nongchao-er/IncepT3SE/blob/main/plt_fig/Figure4.png" width="50%">
