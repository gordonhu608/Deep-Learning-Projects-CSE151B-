# CSE-151B
Group for CSE 151B at UCSD

To train a single multi-layer perceptron using configs provided in config.yamy, type:
```console
  $ python main.py --train_mlp
```
To check the network gradients computed by comparing the gradient computed using'
                             'numerical approximation with that computed as in back propagation:

 ```console
  $ python main.py --check_gradients
```
  
To experiment with weight decay added to the update rule during training:

```console
  $ python main.py --regularization
```

To experiment with different activation functions for hidden units:

```console
  $ python main.py --activation
```
  
To experiment with different network topologies:

```console
  $ python main.py --topology
```
