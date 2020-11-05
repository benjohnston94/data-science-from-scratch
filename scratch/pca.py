"""
Steps of PCA
- 'De-mean' the data so that it's centred around zero
- Initialise a guess vector 'w' with same number of dimension as data. (Should be rescaled
    to magnitude 1 each iteration)
- Calculate the sum of the squared directional variance i.e. the sum of squared the dot product
    of the w vector and all points in the dataset (this is our loss function)
- Calculate the gradient of the loss function with respect to the w vector
- Take a step in the direction of the gradient (trying to maximise variance)
- Iterate until we reach vector of maximal variance (or until we reach max number of iterations)
- This will get us our 'first principal component' i.e. vector in the direction of maximal variance
- Next we 'remove' this projection from our data and recalculate the next 'principal component'
"""
