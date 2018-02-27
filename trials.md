1. vgg16 lr = 0.001 1 layer. Result: terrible
2. vgg16 lr = 0.0001 1 layer decay per epoch = 0.01. result: high train accuracy, low test accuracy, high loss
3. vgg16, decay lr on plateau by factor 0.01, 1 layer, result, 60%
4. vgg16, decay lr on plateau by factor 0.01, 2 layers, result, 60%
5. vgg16, prelu, result: 50% wrong implementation
6. vgg16 1 layer no dropout, 50% accuracy