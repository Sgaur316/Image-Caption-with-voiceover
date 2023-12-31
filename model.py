import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    Encoder - loading ResNet and adjusting the last fully connected layer
    to the embed_size parameter and perform CNN features extraction.
    """
    
    def __init__(self, embed_size):
        """
        Class constructor 
        :param embed_size: is the dimension of embedding of the CNN features.
                           This parameter is also the input vector of the decoder.
        """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules) # use * to unpack list
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size) # perform BatchNorm1d in the last dense layer to reduce overfitting

    def forward(self, images):
        """
        PyTorch forward step to perform features extraction from the images.
        
        :param images: images correctly transformed to run features extraction
        :returns features: features vector
        """
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    """
    Decoder - Input embed_size-like vector and LSTM cells to output captions 
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Class constructor - define several hyperparamenters of the Decoder model.
    
        :param embed_size: shape of image vector and word embeddings
        :param hidden_size: number of features in hidden state of the RNN decoder
        :param vocab_size: size of vocabulary or output size
        :param num_layer: number of layers of the architecture
        
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # The LSTM takes embedded vectors as inputs
        # and outputs hidden states of hidden_size
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers,
                            batch_first = True)
        
        # the linear layer that maps the hidden state output dimension 
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        """
        Pytorch forward running over captions and features
        
        :param features: feature vector provided by the Encoder stage
        :param captions: caption strings used
        :returns outputs: features vector
        """

        captions = captions[:,:-1] 
        embeds = self.word_embeddings(captions)
        
        # concatenating features to embedding torch.cat 3D tensors
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)
        
        lstm_out, hidden = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        
        return outputs
    
        
    def sample(self, inputs, state=None, max_len=20):
        """
        Creates samples of captions for pre-processed image tensor (inputs).
        Returns predicted sentence (list of tensor ids of length max_len)
        """
        
        caption = [] # allocate empty list
        
        for i in range(max_len):
            
            lstm_out, state = self.lstm(inputs, state)

            lstm_out = lstm_out.squeeze(1) # change shape of the output
            outputs = self.linear(lstm_out) # get output
            
            # get the prediction with the maximum probabilities and point to the word
            target = outputs.max(1)[1] 
            
            # append result
            caption.append(target.item())
            
            # feed input to the next state
            inputs = self.word_embeddings(target).unsqueeze(1)
            
        return caption
