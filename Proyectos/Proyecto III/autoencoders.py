import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data as D
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import os
from math import log10

"""** Create Encoder**
The encoder with 4 sets of: Convolutions, ReLu activations and batch normalizations in layer1.
Layer2 has 2 sets of: convolutions, ReLu activations and batch normalizations.
The batch normalization substracts the mean over the n samples received as input :
$y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$
where $ \gamma $ and $ \beta $ are learnable parameters
"""

"""
Encoder class, reduces input dimensionality
"""
class Encoder(nn.Module):
    """
    Class constructor, creates both metalayers
    """
    def __init__(self):
        super(Encoder,self).__init__()
        #4 sets of: Convolutions, ReLu activations and batch normalizations
        self.layer1 = nn.Sequential(
                        #128x128 input dimensions
                        #Always applies
                        #1 input layer, 32 filters of 3x3, creating 32 activation maps, with ReLu activation and batch normalization
                        nn.Conv2d(3, 32, 3, padding = 1),   
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        #32 input layers, with 32 filters of 3x3, with ReLu activation and batch normalization
                        nn.Conv2d(32, 32, 3, padding = 1),   
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        #32 input layers, with 64 filters of 3x3, with ReLu activation and batch normalization
                        nn.Conv2d(32, 64, 3, padding = 1),  
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        #32 input layers, with 64 filters of 3x3, with ReLu activation and batch normalization
                        nn.Conv2d(64, 64, 3, padding = 1),  
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        #Max pooling to reduce its dimensionality to 64x64
                        nn.MaxPool2d(2,2)  
        )
        self.layer2 = nn.Sequential(
                        #Input of 14x14 dimensions
                        #A padding of 1 to preserve dimensionality in 3x3 filtering
                        #64 input layers, with 128 filters of 3x3, with ReLu and batch normalization
                        nn.Conv2d(64, 128, 3, padding = 1),  
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        #128 input layers, with 128 filters of 3x3, with ReLu and batch normalization
                        nn.Conv2d(128, 128, 3, padding = 1), 
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        #Max pooling,making 32x32 output dimensions
                        nn.MaxPool2d(2, 2),
                        #128 input layers, with 256 filters of 3x3, with ReLu activation function
                        nn.Conv2d(128, 256, 3, padding = 1),  
                        nn.ReLU()
        )
        
    """
    Forward pass
    @param x, input image of 128x128 RGB
    """           
    def forward(self,x):
        #Layer1 takes a 128x128 RGB input to 64 activation maps of 64x64 dimensions
        out = self.layer1(x)
#         print(out.size())
        #Layer2 takes 64 activation maps of 64x64 dimensions to 256 activation maps of 32x32
        out = self.layer2(out)
#         print(out.size())
        #flattens the output, for the WHOLE BATCH??
        out = out.view(batchSize, -1)
        return out

"""** Decoder **

The decoder is based in the Conv2DTranspose, which calculates a deconvolution kernel, able to increment output dimensionality.
"""

"""
Decoder class
"""
class Decoder(nn.Module):
    """
    Constructor
    The following are the transponse2d parameters:

    torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
    """
    def __init__(self):
        super(Decoder,self).__init__()
        #Instead of ConvTranspose2d, you can use nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) for bilinear filtering for instance. 
        self.layer1 = nn.Sequential(
                        #From the encoding part, 256 maps of 32x32 are received
                        #Always a grouping of 1 (all filters are applied to all input maps) and an output padding of 1 is used
                        #Takes 256 input maps, and creates 128 activation maps applying same number of filters of 3x3
                        #stride of 2, padding of 1 and output padding of 1, VER FORMULAS 
                        nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), 
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        #Takes 128 input maps, and creates 128 activation maps applying same number of filters of 3x3
                        #stride of 1, padding of 1 and output padding of 1
                        nn.ConvTranspose2d(128, 128, 3, 1, 1),   # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        #Takes 128 input maps, and creates 64 activation maps applying same number of filters of 3x3
                        #stride of 2, padding of 1 and output padding of 1
                        nn.ConvTranspose2d(128, 64, 3, 1, 1),    # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        #Takes 64 input maps, and creates 64 activation maps applying same number of filters of 3x3
                        #stride of 2, padding of 1 and output padding of 1
                        nn.ConvTranspose2d(64, 64, 3, 1, 1),     # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
                        #Takes 64 input maps, and creates 32 activation maps applying same number of filters of 3x3
                        #stride of 2, padding of 1 and output padding of 1
                        nn.ConvTranspose2d(64, 32, 3, 1, 1),     
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        #Takes 32 input maps, and creates 32 activation maps applying same number of filters of 3x3
                        #stride of 2, padding of 1 and output padding of 1
                        nn.ConvTranspose2d(32, 32, 3, 1, 1),    
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        #Takes 32 input maps, and creates 32 activation maps applying same number of filters of 3x3
                        #stride of 2, padding of 2 and output padding of 1
                        nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),    
                        nn.ReLU()
        )
        
    """
    Forward pass
    @param x, input plain tensor
    """
    def forward(self,x):
        #takes the input which is a plain tensor of 32x32x256 and modifies dimensionality
        out = x.view(batchSize, 256, 32, 32)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

"""
** Load data **
"""
class Cats_vs_dogs_DS(D.Dataset):

    def __init__(self, root):
        self.root = root
        self.filenames = []
        

        self.transform = transforms.Compose([
                transforms.ToTensor()
                ])

        
        self.filenames = glob.glob(root)
        
        self.len = len(self.filenames)

    def __getitem__(self, index):
        # le damos una imagen o ejemplo del dataset
        image = Image.open(self.filenames[index])
        return self.transform(image)

    def __len__(self):
        return self.len

def loadData(path):

  dataImg = Cats_vs_dogs_DS(path)
  
  loader = D.DataLoader(dataImg, batch_size=batchSize, shuffle=True)
  
  print('Data loaded corretly perro:', dataImg.len)
  
  return loader

"""** Train Model **

Trains the model with one batch per epoch.
"""

"""
Takes an image batch and adds noise
@param imagesBatch, 3d tensor with  batch of images
@param batchSize
"""
def addNoise(imagesBatch, batchSize):
    #creates a 3d tensor of  380x380 random values corresponding to noise
    noise = torch.zeros(batchSize, 3, 128, 128).normal_(0, 0.05)
    noisyBatch = imagesBatch + noise
    return noisyBatch
  
"""
Takes an image batch and adds noise
@param imagesBatch, 3d tensor with  batch of images
@param batchSize
@param amountof pixels to change
@param salt_vs_pepper probabilty of pixel to be change for salt
"""
def addNoise_Salt_Pepper(imagesBatch, batchSize, amount, salt_vs_pepper):
    
    for image in range(batchSize):
        image_ = imagesBatch[image].numpy()
        out = image_.copy()
        p = amount
        q = salt_vs_pepper
        flipped = np.random.choice([True, False], size=image_[0].shape,
                                   p=[p, 1 - p])
        salted = np.random.choice([True, False], size=image_[0].shape,
                                  p=[q, 1 - q])
        peppered = ~salted
        out[0][flipped & salted] = 1
        out[1][flipped & salted] = 1
        out[2][flipped & salted] = 1
        
        out[0][flipped & peppered] = 0
        out[1][flipped & peppered] = 0
        out[2][flipped & peppered] = 0
        imagesBatch[image] = torch.from_numpy(out)
        
"""
Takes an image batch and adds noise
@param imagesBatch, 3d tensor with  batch of images
@param batchSize
"""
def addNoiseSpeckle(imagesBatch, batchSize):
    noise = torch.zeros(batchSize, 3, 128, 128).normal_(0, 0.2)
    noisyBatch = imagesBatch + imagesBatch * noise
    return noisyBatch

"""
Trains the model with an squared error loss function
@param learning rate
@param epochs
@param batchSize
@param trainLoader
"""
def trainModel(learningRate, epochs, batchSize, trainLoader, encoder=None, decoder=None):
  #create encoder and decoder
  if (encoder == None and decoder == None):
    encoder = Encoder().cuda()
    decoder = Decoder().cuda()
  # In order to use multi parameters with one optimizer, we concatenate the parameters in one list
  parameters = list(encoder.parameters()) + list(decoder.parameters())
  #we use MSE as loss function
  lossFunction = nn.MSELoss()
  #create the optimizer, using Adam
  optimizer = torch.optim.Adam(parameters, lr = learningRate)

  # train encoder and decoder
  for i in range(epochs):
    
    
    for imagesBatch in trainLoader:
      
        #add Noise to batch
        imagesBatchNoise = imagesBatch + 0
        addNoise_Salt_Pepper(imagesBatchNoise, batchSize, 0.2, .5)
        
        #to cuda
        imagesBatch = Variable(imagesBatch).cuda()
        imagesBatchNoise = Variable(imagesBatchNoise).cuda()
        #reset optimizer
        optimizer.zero_grad()
        #forward pass
        outputBatch = encoder(imagesBatchNoise)
        outputBatch = decoder(outputBatch)
        #compute batch loss
        loss = lossFunction(outputBatch, imagesBatch)
        #compute gradients
        loss.backward()
        #optimize
        optimizer.step()
            
    loss1 = loss

    torch.save(encoder.state_dict(), encp)
    torch.save(decoder.state_dict(), decp)
    
    print("Loss: ", loss1.data.tolist(), " epoch: ", i)
       

  # check image with noise and denoised image\

  image = imagesBatch[0].cpu()
  inputImage = imagesBatchNoise[0].cpu()
  outputImage = outputBatch[0].cpu()

  transform = transforms.Compose([
                transforms.ToPILImage()
                ])

  tmp = transform(image)
  
  plt.imshow(tmp)
  plt.show()

  tmp1 = transform(inputImage)
  
  plt.imshow(tmp1)
  plt.show()

  tmp2 = transform(outputImage)
  
  plt.imshow(tmp2)
  plt.show()

  return (encoder, decoder)

"""** Test Model **

Test the model with an external dataset.

**Hyperparameter definition**

Learning rate, epochs, number of input, hidden and output units
contextConcatInputLayerSize is the number of units of hidden plus input units
"""

#number of epochs in total
epochs = 50
#Batch size: number of samples to be used per epoch
#More samples need more memory
batchSize = 50
#Learning rate
learningRate = 0.0002

encp = '/content/gatosyperros/encoder.model'
decp = '/content/gatosyperros/decoder.model'

path = '/content/gatosyperros/train128/*.jpg'

"""** Main function **"""

"""
Main function
"""
def main():
  
    encoder = Encoder()
    encoder.load_state_dict(torch.load(encp))
    encoder.eval()
    encoder.cuda()
    
    decoder = Decoder()
    decoder.load_state_dict(torch.load(decp))
    decoder.eval()
    decoder.cuda()
  
    trainLoader = loadData(path)
    (encoder, decoder) = trainModel(learningRate, epochs, batchSize, trainLoader, encoder, decoder)

    torch.save(encoder.state_dict(), encp)
    torch.save(decoder.state_dict(), decp)
  
main()

pathTest1 = '/content/fruits/*.jpg'

encp = '/content/encoder.model'
decp = '/content/decoder.model'

"""
Trains the model with an squared error loss function
@param encoder
@param decoder
@param testLoader
"""
def testModel(encoder, decoder, testLoader):
   
#we use MSE as loss function
    lossFunction = nn.MSELoss()
    epoch = 0
    avg_psnr = 0
    #loads one images and labes batch per epoch
    for imagesBatch in testLoader:
      
        #add Noise to batch
        imagesBatchNoise = addNoise(imagesBatch, batchSize)
      
        #to cuda
        imagesBatch = Variable(imagesBatch).cuda()
        imagesBatchNoise = Variable(imagesBatchNoise).cuda()

        #forward pass
        outputBatch = encoder(imagesBatchNoise)
        outputBatch = decoder(outputBatch)
        
        #compute batch loss
        loss = lossFunction(outputBatch, imagesBatch)            
        
        psnr = 10 * log10(1/ loss)
        avg_psnr += psnr
        
        epoch += 1;

    print("PSNR:", psnr / len(testLoader))
        
    # check image with noise and denoised image\
    image = imagesBatch[0].cpu()
    inputImage = imagesBatchNoise[0].cpu()
    outputImage = outputBatch[0].cpu()

    transform = transforms.Compose([
                  transforms.ToPILImage()
                  ])

    tmp = transform(image)
    plt.imshow(tmp)
    plt.show()

    tmp1 = transform(inputImage)
    plt.imshow(tmp1)
    plt.show()

    tmp2 = transform(outputImage)
    plt.imshow(tmp2)
    plt.show()

"""
Main function
"""
def main():
  
    encoder = Encoder()
    encoder.load_state_dict(torch.load(encp))
    encoder.eval()
    encoder.cuda()
    
    decoder = Decoder()
    decoder.load_state_dict(torch.load(decp))
    decoder.eval()
    decoder.cuda()
  
  
    testLoader1 = loadData(pathTest1)
  
    testModel(encoder, decoder, testLoader1)

main()
