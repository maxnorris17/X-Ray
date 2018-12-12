# CNN to predict pneumonia from chest X-Ray images

Motivation: The goal of using CNNs in Radiology is to help Radiologists ​more accurately diagnose patients and to help patients who lack access to Radiologists diagnose themselves using their chest X-Ray images​

Approach: I used partial transfer learning from VGG16 and a 24-layer CNN. I split my image data into a training set (N normal = 1341, N pnemonia = 3875), a cross validation set (N normal = N pnemonia = 8), and a test set (N normal = 234 normal, N pneumonia = 390).

Data: 5863 infant X-Ray images (all images normalized and converted to 244x244x3) from Guangzhou Women and Children’s Medical Center in China. Dependent variable: Pneumonia (N training=3875) vs. Normal (N training=1341). Performed data augmentation (horizontal flips, rotation, random brightness changes) to correct for class imbalance.​

Model: For the first few layers, which capture general details like blobs, patches, edges, ectetera, I load the weights from VGG16 and fine tune them, instead of using random initialization. After that, I use convolutional, batch norm, and max pooling, layers, ending with a softmax layer for binary prediction.

Discussion and Future: The Stanford ML Group, which includes Professor Andrew Ng and Pranav Rajpurkar, has developed a CNN that not only classifies 14 different thoracic diseases, but also pinpoints the location in the chest that causes the CNN to make its prediction, as seen in their figure below. Their algorithm beats the "gold standard" of a committee of radiologists. Now they are working to deploy a website where anyone in the world who lacks access to a Radiologist can upload their own X-Ray image file to get a fast, free diagnosis.

Acknowledgements:
I would like to thank my project mentor, Cristian Bartolomé Aråamburu, for his guidance throughout my project during my proposal and milestone. I would also like to thank the other TAs in CS 230, especially Steven Chen, for their time and assistance during Office Hours and Sarah Najmark for her excellent instruction during Friday TA Sections.​

Works Cited:​

[1] https://www.kaggle.com/aakashnain/beating-everything-with-depthwise-convolution?scriptVersionId=4028995 [Accessed: December 2, 2018]​

[2] https://nihcc.app.box.com/v/ChestXray-NIHCC [Accessed: October 10, 2018]​

[3] https://stanfordmlgroup.github.io/projects/chexnet/ [Accessed: October 15, 2018]​

[4] https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/ [Accessed: December 2, 2018]​
