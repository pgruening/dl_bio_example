# Evaluating different model architecture on MNIST

## Readme files and experiment folders

A good practice is categorizing single experiments into subfolders containing 'exe'-files to train several models and evaluate them. In addition, you should use a README file in each experiment folder to document which question the experiment is supposed to answer, what outcome you predict. Furthermore, you should document the experiment results and conclude them.

### Introduction

This experiment trains several models on [MNIST](http://yann.lecun.com/exdb/mnist/) and compares the test scores. We use a simple sequence of convolution layers. 

### Methods

We use the standard MNIST dataset without any particular data augmentation.
The model starts with a conv-layer with batchnorm with an ouput dimension of 'hidden_dim'. Additionally, 'num_layers' conv-batchnorm-relu layers are added. Each layer increases the number of dimensions by the factor 2.
Note that no pooling layers are used, leading to a suboptimal receptive field size.
The output of the last conv layer is average-pooled globally. The resulting feature vector is transformed to an 'out-dim'-dimensional vector representing the confidence for each class.

We run experiments for different configurations:

```
    to_search = {
        'lr': [.001, .0001],  # try different learning rates
        # try different model parameters
        'num_layers': [1, 2, 3],
        'kernel_size': [3, 5],
        'hidden_dim': [8, 16]
    }
```

Furthermore, each configuration is trained with three different seeds to evaluate the variation of each model.

It is likely that a model with a larger kernel size and a higher number of layers works best since it yields the largest receptive field. A larger receptive field can observe the most context of the input image.

## Results

We report the error of the last epoch, the minimal error, and the training error averaged over three seeds. Additionally, the standard deviation is reported. The table below shows the best results.

### Ten best results

|    |   ('lr', '') |   ('num_layers', '') |   ('kernel_size', '') |   ('hidden_dim', '') |   ('last_train_error', 'mean') |   ('last_train_error', 'std') |   ('last_val_error', 'mean') |   ('last_val_error', 'std') |   ('min_val_error', 'mean') |   ('min_val_error', 'std') |
|---:|-------------:|---------------------:|----------------------:|---------------------:|-------------------------------:|------------------------------:|-----------------------------:|----------------------------:|----------------------------:|---------------------------:|
| 22 |       0.001  |                    3 |                     5 |                   16 |                        1.15833 |                      0.043589 |                      1.17333 |                   0.0152753 |                     1.14333 |                  0.0450925 |
| 23 |       0.001  |                    3 |                     5 |                    8 |                        2.18611 |                      0.106175 |                      1.87667 |                   0.193993  |                     1.87667 |                  0.193993  |
| 10 |       0.0001 |                    3 |                     5 |                   16 |                        3.87111 |                      0.037056 |                      3.18333 |                   0.0404145 |                     3.18333 |                  0.0404145 |
| 20 |       0.001  |                    3 |                     3 |                   16 |                        3.55722 |                      0.118243 |                      3.33667 |                   0.119304  |                     3.27    |                  0.155242  |
| 21 |       0.001  |                    3 |                     3 |                    8 |                        6.48778 |                      0.329584 |                      5.61    |                   0.321403  |                     5.54    |                  0.363731  |
| 11 |       0.0001 |                    3 |                     5 |                    8 |                        8.80333 |                      0.356452 |                      7.7     |                   0.155242  |                     7.69667 |                  0.150111  |
| 18 |       0.001  |                    2 |                     5 |                   16 |                       14.9283  |                      0.777069 |                     13.8     |                   0.641561  |                    13.8     |                  0.641561  |
|  8 |       0.0001 |                    3 |                     3 |                   16 |                       16.2689  |                      1.35493  |                     14.6033  |                   1.539     |                    14.6033  |                  1.539     |
| 19 |       0.001  |                    2 |                     5 |                    8 |                       19.8439  |                      0.517763 |                     18.4667  |                   0.488399  |                    18.3333  |                  0.714516  |
|  9 |       0.0001 |                    3 |                     3 |                    8 |                       36.3606  |                      0.481913 |                     34.7133  |                   0.275015  |                    34.7133  |                  0.275015  |

### Boxplots


<p float="left">
  <img src="boxplots/bp_lr_('last_val_error', 'mean').png" width="32%" />
  <img src="boxplots/bp_num_layers_('last_val_error', 'mean').png" width="32%" />
  <img src="boxplots/bp_hidden_dim_('last_val_error', 'mean').png" width="32%" />
</p>

## Conclusion

Using a learning rate of 0.001 is better for most configurations. A higher number of layers, a larger kernel size, and a higher feature dimension yields models with a larger capacity. Thus, the training error is reduced. Still, these models generalize well. 

To improve the models, operations like pooling or subsampling should be used. 