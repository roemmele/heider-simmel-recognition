
# coding: utf-8

# # Heider-Simmel Action Recognition
# 
# ### The code below demonstrates four supervised models that predict action labels in Heider-Simmel animation data. Run this code step by step to visualize the data processing associated with each model.

# In[ ]:

import json, numpy, pandas, os
from keras import backend as K
from keras.layers import *
from keras.models import Model

pandas.options.mode.chained_assignment = None


# In[ ]:

def parse_animations(animation_files, 
                     shape_data_columns=['big_triangle_XYR','little_triangle_XYR','circle_XYR','door_XYR']):
    """
    INPUT: 
    animation_files: list of string filenames, where each filename corresponds to a single animation
    shape_data_columns: names associated with each column of data 
                        (likely no need to change default, this is just for visualization)
    RETURNS: 
    animation_data: A pandas DataFrame where each row is the data for a single animation.
                    The first column is the sequence of action labels for that animation, and the subsequent columns
                    each contain a 2-D matrix for a particular shape's X,Y,R values for each animation frame.
                    (thus the length of each matrix is equal to the number of frames in the corresponding animation)
    """
    animation_data = []
    for animation_file in animation_files:
        with open(animation_file, 'r') as f:
            animation = json.load(f)

        labels = [frame[0] for frame in animation] # first column is sequence of action labels
        bt_data = numpy.array([frame[1:4] for frame in animation]) # columns 2,3,4 are X, Y, R values for big triangle
        lt_data = numpy.array([frame[4:7] for frame in animation]) # columns 5,6,7 are X, Y, R values for little triangle
        c_data = numpy.array([frame[7:10] for frame in animation]) # columns 8,9,10 are X, Y, R values for circle
        d_data = numpy.array([frame[10] for frame in animation])[:, None] # column 11 is R value for door
        #pad door data with 0's so data can ultimately be processed in matrix format
        d_data = numpy.concatenate([numpy.zeros((len(d_data), 2)), d_data], axis=-1)
        
        animation_data.append([labels, bt_data, lt_data, c_data, d_data])

    animation_data = pandas.DataFrame(animation_data, 
                                      columns=['labels'] + shape_data_columns)
    return animation_data

def get_x_labels(raw_data, labels_to_idxs):
    """
    Extract data from DataFrame into matrix format for input/output to models
    INPUT:
    raw_data: a pandas DataFrame of animations
    labels_to_idxs: a dictionary that maps string animation labels to unique integers
    RETURNS:
    x: numpy matrix of animation data with shape (# animations, # frames in animation, # frame features (i.e. X,Y,R), # shapes)
    labels: numpy matrix of animation labels with shape (# animations, animation length, 1)
    """
    #Assume that first column of DataFrame contains action labels, and all other columns are shape data
    x = numpy.stack([numpy.stack(animation, axis=-1) for animation in raw_data.iloc[:,1:].as_matrix()])
    labels = transform_labels_to_idxs(raw_data.iloc[:,0], labels_to_idxs)
    
    return x, labels

def get_animation_snapshots(x, labels, n_snapshot_frames=100):
    """
    Splits animations into "snapshots", i.e. segments of a fixed number of frames, where all frames in a given segment 
    correspond to the same action label. The purpose is that the model will observe the data for all frames
    in the snapshot in parallel, as if it is observing an image. Note that the segments will be overlapping, such 
    that the beginning of each segment is one frame offset to the right relative to the beginning of the previous 
    segment, e.g. segment[frame_idx:frame_idx+n_snapshot_frames], segment[frame_idx+1: frame_idx+1+n_snapshot_frames]
    are two neighboring snapshots given in the output.
    
    INPUT:
    x: list or numpy array of data for each animation with shape (# animations, animation length, # frame features, # shapes) 
    labels: list or numpy array of label index sequences for each animation, with shape (# animations, animation_length, 1)
    n_snapshot_frames: the number of frames contained in each output snapshot
    
    RETURNS:
    all_snapshots: matrix of snapshot data with shape (# animations, # snapshots, n_snapshot_frames, # frame features, # shapes)
    all_snapshot_labels: matrix of action labels for each snapshot with shape (# animations, # snapshots, 1)
    
    """
    all_snapshots = []
    all_snapshot_labels = []
    for animation, animation_labels in zip(x, labels):
        assert(len(animation.shape) <= 3)
        snapshots = []
        snapshot_labels = []
        for frame_idx in range(len(animation) - n_snapshot_frames):
            snapshot = animation[frame_idx:frame_idx + n_snapshot_frames]
            snapshots.append(snapshot)
            snapshot_labels.append(animation_labels[frame_idx])
        assert(len(snapshots) == len(snapshot_labels))
        all_snapshots.append(numpy.array(snapshots))
        all_snapshot_labels.append(numpy.array(snapshot_labels))
    all_snapshots = numpy.array(all_snapshots)
    all_snapshot_labels = numpy.array(all_snapshot_labels)
    assert(len(all_snapshots) == len(all_snapshot_labels))
    return all_snapshots, all_snapshot_labels

def split_animations(x, labels, n_segment_frames=100):
    """
    Splits animations into segments of n_segment_frames to be provided an input to an RNN model. The model will iteratate
    through all frames in a given animation segment, so that the prediction for the action label at 
    animation[frame_idx] is conditioned upon animation[frame_idx - n_segment_frames:frame_idx]. This function is different from
    get_animation_snapshots(), which converts each frame of the animation into a 2-D image of frames by taking 
    into account the frames that appear after it in some window. Here, we are just splitting the animation into
    segments of datapoints that are processed as sequentially, as an alternative to the RNN observing all datapoints in 
    a given animation, which is inefficient for longer animations. Each datapoint can be a single frame or a snapshot, 
    thus this function can be called after get_animation_snapshots().
    INPUT:
    x: matrix of data for each animation with shape (# animations, # frames, # frame features, # shapes)
        or (# animations, # snapshots, # snapshot frames, # frame features, # shapes) if data contains snapshots
    labels: matrix of label index sequences for each animation, with shape (# animations, # frames, 1)
            or (# animations, # snapshot frames, 1) if data contains snapshots
    n_segment_frames: the number of frames contained in each output segment
    RETURNS:
    segments: matrix of animation segments with shape (# segments, n_segment_frames, #frame features, # shapes)
              or (# segments, n_segment_frames, # snapshot frames, # frame features, # shapes) if data contains snapshots
    segment_labels: matrix of animation labels with shape (# segments, n_segment_frames, 1)   
    
    """
    segments = []
    segment_labels = []
    for animation, animation_labels in zip(x, labels):
        animation_segments = [animation[frame_idx:frame_idx+n_segment_frames] 
                                        for frame_idx in range(0, len(animation), n_segment_frames)]
        animation_segment_labels = [animation_labels[frame_idx:frame_idx+n_segment_frames] 
                                    for frame_idx in range(0, len(animation), n_segment_frames)]
        if len(animation_segments[-1]) < n_segment_frames: #if last segment is shorter than n_frames, pad with zeros
            animation_segments[-1] = numpy.append(animation_segments[-1], 
                                                  numpy.zeros((n_segment_frames - len(animation_segments[-1]),) +
                                                               animation_segments[-1].shape[1:]), axis=0)
            animation_segment_labels[-1] = numpy.append(animation_segment_labels[-1],
                                                        numpy.zeros((n_segment_frames - len(animation_segment_labels[-1]),) +
                                                                     animation_segment_labels[-1].shape[1:]), axis=0)
        segments.extend(numpy.array(animation_segments))
        segment_labels.extend(numpy.array(animation_segment_labels))
    segments = numpy.array(segments)
    segment_labels = numpy.array(segment_labels)
    return segments, segment_labels       
            
def get_label_idx_alignment(labels):
    """
    INPUT:
    labels: a pandas Series, where each item contains the sequence of action labels for a single animation
    RETURNS:
    labels_to_idxs: a dictionary where each label found in animation_labels is mapped to a unique integer
    idxs_to_labels: a dictionary that reverses labels_to_idxs so labels can be looked up from their indices
    """
    # Reserve index 0 for labels that are not in the training data 
    # This is necessary because animations with these labels could show up in the test set
    labels_to_idxs = {'<UNKNOWN>': 0}
    cur_label_idx = 1
    for animation in labels:
        for label in animation:
            if label not in labels_to_idxs:
                labels_to_idxs[label] = cur_label_idx
                cur_label_idx += 1
    
    idxs_to_labels = {idx:label for label, idx in labels_to_idxs.items()}
                
    return labels_to_idxs, idxs_to_labels

def transform_labels_to_idxs(labels, labels_to_idxs):
    """
    INPUT:
    labels: a pandas Series, where each item contains the sequence of action labels for a single animation
    labels_to_idxs: a dictionary that maps string animation labels to unique integers
    RETURNS:
    labels: a matrix with shape (# animations, animation length, 1), where the final dimension contains 
    the numeric (index) representation of each action label
    """
    
    #If label is not in labels_to_idxs, assign it a label index of 0
    labels = numpy.stack(labels.apply(lambda animation_labels: numpy.array([labels_to_idxs[label] 
                                      if label in labels_to_idxs else 0
                                      for label in animation_labels])[:,None]).as_matrix())
    return labels 

def create_mlp_model(n_snapshot_frames, n_frame_features, n_shapes, 
                     n_labels, n_hidden_layers=1, n_hidden_dim=500):
    """
    INPUT:
    n_snapshot_frames: # of animation frames in each snapshot
    n_frame_features: # of input dimensions for each shape. This will likely be 3 (X,Y,R)
    n_shapes: # of shapes in the input data. This will likely be 4 (big triangle, little triangle, circle, door)
    n_labels: # of unique action labels in the output
    n_hidden_layers: # of hidden layers, which can be freely tuned
    n_hidden_dim: # of nodes in the hidden layer, which can be freely tuned
    RETURNS:
    model: a Keras model that is ready to be trained
    """

    input_layer = Input(shape=(n_snapshot_frames, n_frame_features, n_shapes), name='input')
    flatten_layer = Flatten(name='flatten')(input_layer)
    hidden_layer = Dense(units=n_hidden_dim, activation='sigmoid', name='hidden1')(flatten_layer)
    for layer_idx in range(1, n_hidden_layers):
        hidden_layer = Dense(units=n_hidden_dim, activation='sigmoid', 
                             name='hidden' + str(layer_idx + 1))(hidden_layer)
    output_layer = Dense(units=n_labels, activation='softmax', name='output')(hidden_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')
    return model 

def create_rnn_model(segment_length, n_frame_features, n_shapes, 
                     n_labels, n_hidden_layers=1, n_hidden_dim=500):
    """
    INPUT:
    segment_length: # of frames in each animation segment given as input
    n_frame_features: # of input dimensions for each shape. This will likely be 3 (X,Y,R)
    n_shapes: # of shapes in the input data. This will likely be 4 (big triangle, little triangle, circle, door)
    n_labels: # of unique action labels in the output
    n_hidden_layers: # of hidden layers, which can be freely tuned
    n_hidden_dim: # of nodes in the hidden layer, which can be freely tuned
    RETURNS:
    model: a Keras model that is ready to be trained
    """
    input_layer = Input(shape=(segment_length, n_frame_features, n_shapes), name='input')
    flatten_layer = TimeDistributed(Flatten(name='flatten'))(input_layer)
    rnn_layer = GRU(units=n_hidden_dim, name='rnn1', return_sequences=True)(flatten_layer)
    for layer_idx in range(1, n_hidden_layers):
        rnn_layer = GRU(units=n_hidden_dim, name='rnn' + str(layer_idx + 1), return_sequences=True)(rnn_layer)
    output_layer = Dense(units=n_labels, activation='softmax', name='output')(rnn_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')
    return model

def create_cnn_model(n_snapshot_frames, n_frame_features, n_shapes, n_labels, filters=5, 
                     kernel_size=10, strides=5, pool_size=2, n_hidden_layers=1, n_hidden_dim=500):
    """
    INPUT:
    n_snapshot_frames: # of animation frames in each snapshot
    n_frame_features: # of input dimensions for each shape. This will likely be 3 (X,Y,R)
    n_shapes: # of shapes in the input data. This will likely be 4 (big triangle, little triangle, circle, door)
    n_labels: # of unique action labels in the output
    filters: # of filters in the convolutional layer
    kernel_size: size of kernel in the convolutional layer
    strides: stride length in convolutional layer
    pool_size: size of max pooling window
    n_hidden_layers: # of hidden layers, which can be freely tuned
    n_hidden_dim: # of nodes in the hidden layer, which can be freely tuned
    RETURNS:
    model: a Keras model that is ready to be trained
    """
    input_layer = Input(shape=(n_snapshot_frames, n_frame_features, n_shapes), name='input')
    # reshape to flatten n_frame_features and n_shapes into same dimension
    reshape_layer = Reshape((n_snapshot_frames, -1))(input_layer)
    conv_layer = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, 
                        activation='sigmoid', padding='same', name='convolution')(reshape_layer)
    pool_layer = MaxPooling1D(pool_size=pool_size, padding='same', name='pool')(conv_layer)
    flatten_layer = Flatten(name='flatten')(pool_layer)
    hidden_layer = Dense(units=n_hidden_dim, activation='sigmoid', name='hidden1')(flatten_layer)
    for layer_idx in range(1, n_hidden_layers):
        hidden_layer = Dense(units=n_hidden_dim, activation='sigmoid', 
                             name='hidden' + str(layer_idx + 1))(hidden_layer)
    output_layer = Dense(units=n_labels, activation='softmax', name='output')(hidden_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')
    return model

def create_hybrid_cnn_rnn_model(segment_length, n_snapshot_frames, n_frame_features, n_shapes,
                                n_labels, filters=5, kernel_size=10, strides=5, pool_size=2, 
                                n_hidden_layers=1, n_hidden_dim=500):
    """
    INPUT:
    n_snapshot_frames: # of animation frames in each snapshot
    n_frame_features: # of input dimensions for each shape. This will likely be 3 (X,Y,R)
    n_shapes: # of shapes in the input data. This will likely be 4 (big triangle, little triangle, circle, door)
    n_labels: # of unique action labels in the output
    filters: # of filters in the convolutional layer
    kernel_size: size of kernel in the convolutional layer
    strides: stride length in convolutional layer
    pool_size: size of max pooling window
    n_hidden_layers: # of hidden layers, which can be freely tuned
    n_hidden_dim: # of nodes in the hidden layer, which can be freely tuned
    RETURNS:
    model: a Keras model that is ready to be trained
    """
    input_layer = Input(shape=(segment_length, n_snapshot_frames, n_frame_features, n_shapes), name='input')
    reshape_layer = Reshape((segment_length, n_snapshot_frames, -1))(input_layer)
    conv_layer = TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, 
                                        activation='sigmoid', padding='same', name='convolution'))(reshape_layer)
    pool_layer = TimeDistributed(MaxPooling1D(pool_size=pool_size, padding='same', name='pool'))(conv_layer)
    flatten_layer = TimeDistributed(Flatten(name='flatten'))(pool_layer)
    rnn_layer = GRU(units=n_hidden_dim, name='rnn1', return_sequences=True)(flatten_layer)
    for layer_idx in range(1, n_hidden_layers):
        rnn_layer = GRU(units=n_hidden_dim, name='rnn' + str(layer_idx + 1), return_sequences=True)(rnn_layer)
    output_layer = Dense(units=n_labels, activation='softmax', name='output')(rnn_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy')
    return model

def evaluate_prediction(model, x, labels):
    """
    Computes the accuracy and perplexity of the predicted labels for x as evaluation metrics 
    (lower perplexity indicates more accurate probability predictions for correct labels).
    
    INPUT:
    model: trained Keras model
    x: matrix of animation instances where shape is same as input format to model
    labels: matrix of animation labels where shape is same as input format to model
    RETURNS:
    accuracy: average accuracy of predicted labels for animations,
    where predicted label for an animation is the one with the max probability score
    perplexity: average perplexity of model based on predicted probability scores for labels in animations
    """
    pred_probs = model.predict(x)
    pred_probs = pred_probs.reshape((-1, pred_probs.shape[-1]))
    labels = labels.flatten().astype(int)
    accuracy = numpy.mean(numpy.argmax(pred_probs, axis=-1) == labels)
    pred_probs = pred_probs[numpy.arange(len(labels)), labels]
    perplexity = numpy.exp(-numpy.mean(numpy.log(pred_probs)))
    return accuracy, perplexity


def predict_labels(model, animation, n_best=1, recurrent=False):
    """Use the model to predict the probability of the labels for each frame in the animation.
    Return sequences of probability scores for the n_best labels with the highest scores at each frame
    INPUT:
    model: trained Keras model
    animation: a single animation instance where shape is same as input format to model
    n_best: number of labels/probability scores to return for each frame
    recurrent: whether or not the data is in sequential form (i.e. model has an RNN layer)
    RETURNS:
    all_best_labels: matrix of label predictions with shape (# frames, n_best)
    all_best_probs: matrix of probability scores for corresponding all_best_labels with shape (# frames, n_best)
    """
    all_best_labels = []
    all_best_probs = []
    if recurrent: #if model is RNN, append new dimension
        pred_probs = model.predict(animation[None])[0]
    else:
        pred_probs = model.predict(animation)
    for probs in pred_probs:
        best_labels = numpy.argsort(probs)[::-1][:n_best]
        all_best_labels.append(best_labels)
        best_probs = probs[best_labels]
        all_best_probs.append(best_probs)
    all_best_labels = numpy.array(all_best_labels)
    all_best_probs = numpy.array(all_best_probs)
    return all_best_labels, all_best_probs


# In[ ]:

"""Load the training and testing data, put in matrix format"""


animation_filenames = ['example_animations/' + filename for filename in os.listdir('example_animations')]
n_test_animations = 2
raw_train_data = parse_animations(animation_filenames[:-n_test_animations])
raw_test_data = parse_animations(animation_filenames[-n_test_animations:])
#Translate between string action labels and numerical indices
labels_to_idxs, idxs_to_labels = get_label_idx_alignment(raw_train_data['labels'])
train_x, train_labels = get_x_labels(raw_train_data, labels_to_idxs)
test_x, test_labels = get_x_labels(raw_test_data, labels_to_idxs)


# In[ ]:

raw_train_data


# In[ ]:

train_x.shape, train_labels.shape


# In[ ]:

labels_to_idxs, idxs_to_labels


# # Multilayer Perceptron (MLP)
# 
# ### The simplest model is a multilayer perceptron trained to predict action labels from a window of frame data compressed into a single image (a "snapshot"). Here, each input instance is a single snapshot. The trajectory (X,Y,R) values for each shape within the snapshot are concatenated into a single vector, and this representation is modeled by a hidden layer. The output is a probability distribution over action labels for each snapshot. 

# In[ ]:

"""Split training animations so that each input instance to the model contains 
data for multiple frames of length n_snapshot_frames. The value for n_snapshot_frames can be freely tuned.
Each of these "snapshots" will correspond to a single label."""

n_snapshot_frames = 500
train_x, train_labels = get_animation_snapshots(train_x, train_labels, n_snapshot_frames)


# In[ ]:

train_x.shape, train_labels.shape


# In[ ]:

"""As input to model, flatten data so that the first dimension corresponds to each snapshot instead of each animation.
In the MLP and CNN models, each label prediction is based on only on a single snapshot instead of a series."""

train_x = train_x.reshape(-1, train_x.shape[-3], train_x.shape[-2], train_x.shape[-1])
train_labels = train_labels.reshape(-1, train_labels.shape[-1])


# In[ ]:

train_x.shape, train_labels.shape


# In[ ]:

"""Create MLP model"""

model = create_mlp_model(n_snapshot_frames=train_x.shape[1],
                         n_frame_features=train_x.shape[-2],
                         n_shapes=train_x.shape[-1],
                         n_labels=len(labels_to_idxs))


# In[ ]:

"""Train MLP model"""

loss = model.fit(x=train_x, y=train_labels,
                 epochs=2, batch_size=100, verbose=0)
print(loss.history)


# In[ ]:

"""Divide test data into snapshots as with training animations"""

if __name__ == '__main__':
    test_x, test_labels = get_animation_snapshots(test_x, test_labels, n_snapshot_frames)


# In[ ]:

test_x.shape, test_labels.shape


# In[ ]:

"""Evaluate trained model on test data in terms of prediction accuracy and perplexity."""

accuracy, perplexity = evaluate_prediction(model, 
                                           test_x.reshape(-1, test_x.shape[-3], test_x.shape[-2], test_x.shape[-1]),
                                           test_labels.reshape(-1, test_labels.shape[-1]))


# In[ ]:

accuracy, perplexity


# In[ ]:

"""Predict the label sequence probabilities for an animation; 
predict_labels() takes a single animation as input. 
Use the n_best parameter to return the top N predictions for each snapshot"""

test_animation = test_x[0]
test_animation_labels = test_labels[0]
pred_labels, pred_probs = predict_labels(model, test_animation, n_best=2)


# In[ ]:

# Print predictions with their probabilities, alongside gold label

list(zip([[idxs_to_labels[label_idx] for label_idx in labels] for labels in pred_labels],
         pred_probs,
         [[idxs_to_labels[label_idx] for label_idx in labels] for labels in test_animation_labels]))


# # Convolutional Neural Network (CNN)
# 
# ### An extension of the MLP model is to use a convolutional layer to compute a feature representation of the snapshots, rather than the hidden layer directly observing all of the trajectory data in a snapshot. The inputs and outputs to this model are set up the same way as in the MLP model. There are various parameters associated with the CNN (i.e. # of filters, kernel size, stride length, and the size of the max pooling for the convolutional features). I don't have much of an intuition about what these parameter settings should be, so it might require some experimental tuning.

# In[ ]:


# Reload the training and testing data in case value of n_shapshot_frames is different for CNN
train_x, train_labels = get_x_labels(raw_train_data, labels_to_idxs)
test_x, test_labels = get_x_labels(raw_test_data, labels_to_idxs)

n_snapshot_frames = 500
train_x, train_labels = get_animation_snapshots(train_x, train_labels, n_snapshot_frames)
train_x = train_x.reshape(-1, train_x.shape[-3], train_x.shape[-2], train_x.shape[-1])
train_labels = train_labels.reshape(-1, train_labels.shape[-1])


# In[ ]:

train_x.shape, train_labels.shape


# In[ ]:

"""Create CNN model. Can additionally specify # of filters, kernel_size, stride length, pool_size, 
# of hidden layers, and # of hidden nodes as parameters to be freely tuned."""

model = create_cnn_model(n_snapshot_frames=train_x.shape[1],
                         n_frame_features=train_x.shape[2],
                         n_shapes=train_x.shape[-1],
                         n_labels=len(labels_to_idxs))


# In[ ]:

"""Train CNN model"""

loss = model.fit(x=train_x, y=train_labels,
                 epochs=5, batch_size=100, verbose=0)
print(loss.history)


# In[ ]:

"""Divide test data into snapshots as with training animations"""

test_x, test_labels = get_animation_snapshots(test_x, test_labels, n_snapshot_frames)


# In[ ]:

test_x.shape, test_labels.shape


# In[ ]:

"""Evaluate trained model on test data"""

accuracy, perplexity = evaluate_prediction(model, 
                                           test_x.reshape(-1, test_x.shape[-3], test_x.shape[-2], test_x.shape[-1]), 
                                           test_labels.reshape(-1, test_labels.shape[-1]))


# In[ ]:

accuracy, perplexity


# In[ ]:

"""Predict the label sequence probabilities for a given animation"""

test_animation = test_x[0]
test_animation_labels = test_labels[0]
pred_labels, pred_probs = predict_labels(model, test_animation, n_best=1)


# In[ ]:

# Print predictions alongside gold label

list(zip([[idxs_to_labels[label_idx] for label_idx in labels] for labels in pred_labels],
         pred_probs,
         [[idxs_to_labels[label_idx] for label_idx in labels] for labels in test_animation_labels]))


# # Recurrent Neural Network (RNN)
# 
# ### For a simple RNN model, instead of using snapshots that function as images, the animations will be split into smaller segments that maintain their sequential ordering. Each frame is a timepoint in the RNN. The prediction for the label at each frame is based on the trajectory features of the shapes at that specific frame, conditioned upon the data from all previous frames in the segment given by the hidden state of the RNN.

# In[ ]:

"""Divide training animations into smaller segments of frames of length n_segment_frames. The value n_segment_frames
can be freely selected. Predictions for a label at a given frame will be based on previous frames in that 
segment. This avoids the inefficiency of loading an entire animation into the recurrent layer, making the assumption
that the sequential information up to a particular timepoint in a segment is adequate for predicting the 
animation label at that timepoint."""


# Reload the datasets because format for RNN is different
train_x, train_labels = get_x_labels(raw_train_data, labels_to_idxs)
test_x, test_labels = get_x_labels(raw_test_data, labels_to_idxs)

n_segment_frames = 1000
train_x, train_labels = split_animations(train_x, train_labels, n_segment_frames)


# In[ ]:

train_x.shape, train_labels.shape


# In[ ]:

"""Create RNN model. Can additionally specify # of hidden layers and # of hidden dimensions 
as additional parameters to be tuned."""

model = create_rnn_model(segment_length=train_x.shape[1],
                         n_frame_features=train_x.shape[2],
                         n_shapes=train_x.shape[-1],
                         n_labels=len(labels_to_idxs))


# In[ ]:

"""Train RNN model"""

loss = model.fit(x=train_x, y=train_labels,
                 epochs=5, batch_size=50, verbose=0)
print(loss.history)


# In[ ]:

"""Divide test data into snapshots as with training animations"""

test_x, test_labels = split_animations(test_x, test_labels, n_segment_frames)


# In[ ]:

test_x.shape, test_labels.shape


# In[ ]:

"""Evaluate trained model on test data"""

accuracy, perplexity = evaluate_prediction(model, test_x, test_labels)


# In[ ]:

accuracy, perplexity


# In[ ]:

"""Predict the label sequence probabilities for a given animation"""

test_animation = test_x[0]
test_animation_labels = test_labels[0]
pred_labels, pred_probs = predict_labels(model, test_animation, n_best=1, recurrent=True)


# In[ ]:

# Print predictions alongside gold label

list(zip([[idxs_to_labels[label_idx] for label_idx in labels] for labels in pred_labels],
         pred_probs,
         [[idxs_to_labels[label_idx] for label_idx in labels] for labels in test_animation_labels]))


# # Hybrid CNN-RNN
# 
# ### The CNN and RNN models can be combined in a straightforward way. We extract snapshots from the animations using the procedure for the CNN, and then segment these snapshots using the procedure for the RNN. Thus, the difference between the input to the basic RNN model and this hybrid model is that here the segments consist of snapshots instead of individual frames. In the same manner as the basic CNN model above, the CNN is used to compute the features of the snapshots. Then, instead of predicting a label just from a single snapshot alone as was done for the basic CNN, the snapshots correspond to timepoints in the RNN layer. The RNN iteratively observes the CNN representation of each snapshot in the segment, encoding each into its hidden state just as it did when the timepoints were individual frames. Then the model's prediction for the label of a given snapshot is conditioned upon all previous snapshots in the segment. The motivation is that both the snapshot (short-term view) and sequential information (long-term view) are helpful for predicting action labels, and combining both signals could improve performance.

# In[ ]:

"""Organize animation data into snapshots as done above for CNN model. 
Then split the animations into segments as done above for the RNN model (where each segment here is a series
of snapshots)"""

# Reload the datasets
train_x, train_labels = get_x_labels(raw_train_data, labels_to_idxs)
test_x, test_labels = get_x_labels(raw_test_data, labels_to_idxs)

n_snapshot_frames = 100
n_segment_frames = 1000
train_x, train_labels = get_animation_snapshots(train_x, train_labels, n_snapshot_frames)
train_x, train_labels = split_animations(train_x, train_labels, n_segment_frames)


# In[ ]:

train_x.shape, train_labels.shape


# In[ ]:

"""Create hybrid CNN-RNN model"""

model = create_hybrid_cnn_rnn_model(segment_length=train_x.shape[1],
                                    n_snapshot_frames=train_x.shape[2],
                                    n_frame_features=train_x.shape[-2],
                                    n_shapes=train_x.shape[-1],
                                    n_labels=len(labels_to_idxs))


# In[ ]:

"""Train hybrid model"""

loss = model.fit(x=train_x, y=train_labels,
                 epochs=5, batch_size=50, verbose=0)
print(loss.history)


# In[ ]:

"""Divide test data into snapshots and segments as with training animations"""

test_x, test_labels = get_animation_snapshots(test_x, test_labels, n_snapshot_frames)
test_x, test_labels = split_animations(test_x, test_labels, n_segment_frames)


# In[ ]:

"""Evaluate trained model on test data"""

accuracy, perplexity = evaluate_prediction(model, test_x, test_labels)


# In[ ]:

accuracy, perplexity


# In[ ]:

"""Predict the label sequence probabilities for a given animation"""

test_animation = test_x[0]
test_animation_labels = test_labels[0]
pred_labels, pred_probs = predict_labels(model, test_animation, n_best=1, recurrent=True)


# In[ ]:

# Print predictions alongside gold label

list(zip([[idxs_to_labels[label_idx] for label_idx in labels] for labels in pred_labels],
         pred_probs,
         [[idxs_to_labels[label_idx] for label_idx in labels] for labels in test_animation_labels]))


# In[ ]:



