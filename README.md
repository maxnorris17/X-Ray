# CNN to predict pneumonia from chest X-Ray images

Motivation: 

The goal of using CNNs in Radiology is to help Radiologists more accurately diagnose patients and to help patients who lack access to Radiologists diagnose themselves using their chest X-Ray images

Approach: 

I used partial transfer learning from VGG16 and a 24-layer CNN. I split my image data into a training set (N normal = 1341, N pnemonia = 3875), a cross validation set (N normal = N pnemonia = 8), and a test set (N normal = 234 normal, N pneumonia = 390).

Data: 

5863 infant X-Ray images (all images normalized and converted to 244x244x3) from Guangzhou Women and Children’s Medical Center in China. Dependent variable: Pneumonia (N training=3875) vs. Normal (N training=1341). Performed data augmentation (horizontal flips, rotation, random brightness changes) to correct for class imbalance.

Model: 

For the first four layers, which capture general details like blobs, patches, edges, ectetera, I load the pre-trained weights from VGG16 and fine tune them, instead of using random initialization. After that, I use convolutional, batch norm, and max pooling, layers, ending with a softmax layer for binary prediction.

How to run this code:
1. Set up an AWS account. Without a GPU testing this code froze my MacBook Pro. You will need to log in from terminal with the secure .pem file amazon sends you.
2. Set up an AWS instance (at first I read instructions that recommended instance type p2.xlarge, but TA Steven Chen said p3.2xlarge would be faster, the downside is it is more expensive since it uses more GPUs)
3. Upload a zip file to your p3.2xlarge AWS instance with a) the xray.py file b) the VGG16 weights .h5 file c) the chest_xray folder, which has instide a folder for train, val, and test. (The .h5 file and the chest_xray folder were both too large to upload to this GitHub repository.)
4. Unzip the compressed file using "unzip" in terminal
5. Instead of using separate .py files for each model you wish to run, I found it easier to learn how to use vim commands from terminal to edit the xray.py file to tune hyperparameters. I experimented with changing the batch size, the learning rate, the optimization method (RMSprop vs. Adam) and the number of epochs (the number of times the CNN runs through every training image).

Discussion and Future: 

The Stanford ML Group, which includes Professor Andrew Ng and Pranav Rajpurkar, has developed a CNN that not only classifies 14 different thoracic diseases, but also pinpoints the location in the chest that causes the CNN to make its prediction, as seen in their figure below. Their algorithm beats the "gold standard" of a committee of radiologists. Now they are working to deploy a website where anyone in the world who lacks access to a Radiologist can upload their own X-Ray image file to get a fast, free diagnosis.

Acknowledgements:


I would like to thank my project mentor, Cristian Bartolomé Aramburu, for his guidance throughout my project during my proposal and milestone. I would also like to thank the other TAs in CS 230, especially Steven Chen, for their time and assistance during Office Hours and Sarah Najmark for her excellent instruction during Friday TA Sections.


Works Cited:

[1] https://www.kaggle.com/aakashnain/beating-everything-with-depthwise-convolution?scriptVersionId=4028995 [Accessed: December 2, 2018]

[2] https://nihcc.app.box.com/v/ChestXray-NIHCC [Accessed: October 10, 2018]

[3] https://stanfordmlgroup.github.io/projects/chexnet/ [Accessed: October 15, 2018]

[4] https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/ [Accessed: December 2, 2018]
