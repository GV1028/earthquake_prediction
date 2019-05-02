# **Project Web Presence**

### Project Overview

  Forecasting fault failure is a fundamental but elusive goal in earthquake science. Earthquakes have huge consequences on life, property and socio-economic balance in the world. In this project, we aim to apply Machine Learning and Deep Learning techniques to accurately predict the time of earthquakes. Using such techniques, we hope to come up with an ad-hoc prediction system, by discovering patterns in seismic data obtained from laboratory earthquakes. We hypothesize that deep learning would significantly advance current methods of earthquake prediction and provide new insights on fault physics.

### Project Approach

The team members for this project are Chester Holtz and Vignesh Gokul. Both members have experience with machine learning. In particular, Vignesh has expertise in deep learning and neural network architectures. Earthquake prediction is a well-studied problem. However, there is a gap between the application traditional statistics-based modeling and modern machine learning-based methods. We plan to explore the application of a broad set of approaches and techniques from machine learning, statistics, and optimization. Some examples of our planned models include: deep neural networks (LSTM, CNN, Transformers), sparse quantile regression, and statistical modeling (Hawkes, Poisson). In addition to applying these techniques on the raw signal, we also plan to leverage our signal processing expertise with various pre-processing and analytical algorithms i.e. (robust pca, nonlinear manifold analysis, etc.). Our intent is to design algorithms that are effective for forecasting quakes, but also to make sure that they are efficient (fast, low footprint) enough to potentially run on embedded monitoring devices in the field. We will leverage a gold-standard synthetic & real dataset released by Los Alamos National Laboratory. The data is hosted here: https://www.kaggle.com/c/LANL-Earthquake-Prediction/data and consists of the raw acoustic waveform signal and other statistics.

### Project Goals (High Level)

* To develop a method for predicting the time to failure for the next labaratory earthquakes.

* To determine the potential of Deep Learning and Machine Learning techniques in finding patterns in seismic waves.

* To develop an ad-hoc mechanism for predicting earthquakes quickly


### Risks and How to Avoid Them

Earthquake prediction is not an easy task. In order to have realistic goals, we plan to start by predicting the time factor alone (Time before next earthquake) Since the data is very high-dimensional, one potential failure would be underfitting and not achieving convergence. We plan to explore models that can process high-dimensional signals with low processing power to overcome this aspect. We hope to formulate a schedule and that minimizes risk and guarantees deliverables while still leaving us with room for the exploration and development of more ambitious ideas.

# Group Management

We plan to make decisions by consensus and communicate via email and instant messages. For shared documents, we will utilize overleaf and google drive. For sharing code, we have set up a github repository and may consider leveraging a shared notebook in google colab. We plan to meet biweekly outside of class to review our progress and schedule. We planned our schedule in such a way that, we have some slack for every complex task. Missing out on milestone deadlines is possible and we try to accommodate that in our schedule. Specific details about who is responsible for which deliverables and milestones is given in the project schedule section.

# Project Development

For this project, both members of the team plan to develop code. No additional hardware or software will be required. For computation, we have google compute credit and plan to leverage goole’s colab environment with GPU & deep learning accelerator support. Our primary development language will be in Python. Documentation will primarily be through the use of structured Jupyter notebooks which offer the capability to run Python code, generate plots, and render markdown-formatted text in a single document.

# Project Schedule

### Milestones

<table>
  <tr>
    <td>Description / How will you demonstrate completion</td>
    <td>Due</td>
    <td>Person Responsible</td>
    <td>Priority</td>
  </tr>
  <tr>
    <td>Write Inception Net. Report network accuracy on test set with visualizations of network predictions. Post the code.</td>
    <td>Week 4</td>
    <td>Patrick</td>
    <td>Low</td>
  </tr>
  <tr>
    <td>Build pipeline for adding universal perturbations to a dataset. Show visualizations of the process.
</td>
    <td>Week 4</td>
    <td>Davis</td>
    <td>Medium</td>
  </tr>
  <tr>
    <td>Find a dataset to use for our experiments. Provide reasons for choosing the dataset. </td>
    <td>Week 5</td>
    <td>Patrick</td>
    <td>Medium</td>
  </tr>
  <tr>
    <td>Preprocess the dataset to fit properly within Inception Net framework. Post code.</td>
    <td>Week 5</td>
    <td>Davis</td>
    <td>Medium</td>
  </tr>
  <tr>
    <td>Generate Universal Adversarial Images using preprocessed dataset. Write code for getting accuracy of a pre-trained model given some input and labels. </td>
    <td>Week 6</td>
    <td>Patrick</td>
    <td>High</td>
  </tr>
  <tr>
    <td>Write the autoencoder, convolutional autoencoder, and PCA models. Create three datasets by running the adversarial images through the models to use as input for experiments. </td>
    <td>Week 6</td>
    <td>Davis</td>
    <td>Medium</td>
  </tr>
  <tr>
    <td>Run experiments on auto-encoder, convolutional auto-encoder, and pca models and report accuracy using Patrick's tester code. Write the MTL model. </td>
    <td>Week 7</td>
    <td>Davis</td>
    <td>Medium</td>
  </tr>
  <tr>
    <td>Formalize method for measuring adversarial robustness. Work on digesting mathematical formalization of our methods of increasing adversarial robustness.</td>
    <td>Week 8</td>
    <td>Patrick</td>
    <td>Medium</td>
  </tr>
  <tr>
    <td>Run experiments to see if multi-task learning improves a networks robustness. Report accuracy. </td>
    <td>Week 8</td>
    <td>Davis</td>
    <td>Medium</td>
  </tr>
  <tr>
    <td> REPORT WORK: See how well these techniques generalize across datasets and across network architectures. Report the results of the experiments done on other networks and other datasets.</td>
    <td>Week 9</td>
    <td>Patrick and Davis</td>
    <td>Medium</td>
  </tr>

  <tr>
    <td>Finalize Video Presentation Stuff.</td>
    <td>Week 10</td>
    <td>Patrick and Davis</td>
    <td>High</td>
  </tr>

</table>

# Deliverables

### Build a Control Model 

We decided to use a pre-trained convolutional neural network. We use the [inception v3 network](https://github.com/tensorflow/models/tree/master/inception) provided by tensorflow. Below is a visiualization of the inception v3 architecture

![Inception-v3 Architecture](https://4.bp.blogspot.com/-TMOLlkJBxms/Vt3HQXpE2cI/AAAAAAAAA8E/7X7XRFOY6Xo/s1600/image03.png)

The inception v3 network allows us to generate high resolution adversarail images. 

##### Accuracy of Control Model
Accuracy: 75.85%

##### Confusion Matrix

![](Results/Confusion_Matrices/control.png)

### Choose a Dataset

We will be using a subset of the validation set of the 2012 Large Scale Visual Recognition Challenge hosted by [Imagenet](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads).
The original validation set contains 50,000 images across 1,000 different categories. We reduced this to 2,000 images with 2 images form each category. Below are some examples of the images from the data set.

Label: Coral               |  Label: Jelly Fish        | Label:  Loom
:-------------------------:|:-------------------------:|:------------------:
![](http://farm1.static.flickr.com/87/280429814_d2b5216d99.jpg)  |  ![](http://i.ehow.co.uk/images/a06/85/qk/benthic-zone_-1.1-800X800.jpg) | ![](http://farm1.static.flickr.com/4/6027614_d778b6a4d5.jpg)


### Use Adversarial Filters to Generate a Dataset of Adversarial Images

One method of generating adversarial images is to apply a universal adversarial perturbation. Universal adversarial perturbations are filters which can be applied to an image to convert it into a adversial image which gets misclassified by the vast majority of networks. We took our original data of images and added one of the 6 universal filter to each of the images. We will use this dataset as one of our benchmarks for robustness. A weak network will get roughly random accuracy on the universal adversarial image data set but a network with 10% accuracy will have a ceratain level of certification, a network with 20%  accuracy will have an even higher level of certification and so on. Below are the visualizations of the universal perturbations. 

![](http://www.i-programmer.info/images/stories/News/2016/Nov/A/advers2.jpg)

##### Accuracy of Control Model on Univerasl Adversarial Images
Accuracy: 55.25%

##### Confusion Matrix

![](Results/Confusion_Matrices/universal.png)


### Use Gradient Ascent to Generate a Dataset of Adversarial Images

Another method of generating adversarial images is to choose a certain label that you want your images to be categorized as. For example you might want an image of a cat to be classified as a paper towel. We use gradient ascent to generate the approriate noise, so that when we add the noise to the original image it becomes an adversarial image. We took our original data set of images and for each image we chose another category at random and computed the noise necessary to make our original image be classified as the randomnly selected category. We will generate one of these gradient adversarial datasets for each of the models we create. We have a gradient adversarial dataset for our control inception v3 network, a gradient adversarial dataset for our inception v3 network with an added autoencoder layer, and a gradient adversarial dataset for our multiclass learning network. These data sets will be used to establish a relative measure of robustness. We will test the control network on the autoencoders adversarial dataset, and the autoencoder on the controls dataset. If the autoencoder has a higher accuracy rate then the autoencoder is considered more robust than the control. Eventually we will have standardized set of networks, each with their own established level of robustness. When a customer or boss asks about the robustness of your network you can say it has level four robustness because it is more robust than the forth network in the benchmark suit. 
Below are some examples of adversarial images generate using gradient ascent. 

![](readme_images/adv_balloon.png)
![](readme_images/adv_orca.png)

##### Accuracy of Control Model on Gradient Adversarial Images
Accuracy: 57.55%

##### Confusion Matrix

![](Results/Confusion_Matrices/gradient.png)



### Build a Modified Inception Network Which Does PCA Pre-Processing

##### Motivation
PCA (Principal Component Analysis) is used to reduce the dimensionality of a matrix. Adversarial images rely on adding specificallt targeted noise to an image. By pre-processing the inputs to a network using PCA we hope to remove the adversarial noise from our input images. This is one of our proposed methods for increasing the robustness of a network. Below our some examples of original images along with the PCA version of those images.

#### Examples of Images

Original              |  PCA Verion with 99% of Varience       
:-------------------------:|:-------------------------:
![](readme_images/18579429_10154702481664779_1273974298_n.png)  |  ![](readme_images/18600730_10154702482644779_1634726341_n.png) 

Original              |  PCA Verion with 99% of Varience       
:-------------------------:|:-------------------------:
![](readme_images/18622863_10154702482209779_1002029237_n.png)  |  ![](readme_images/18579322_10154702483074779_1301879580_n.png) 

##### Accuracy After Doing PCA
Accuracy on original images: 23.35%

Accuracy on universal adversarial images: 22.10%
Universal Robustness : 0.9465

##### Confusion Matrix
On Unperturbed Images              |  On Universal Adversarial Images |      
:-------------------------:|:-------------------------:|
![](Results/Confusion_Matrices/control_pca.png) | ![](Results/Confusion_Matrices/data_universal.png)

### Build a Modified Inception Network Which Adds Random Noise Before Classifying

##### Motivation
Adversarial Images add minor variations to the image's pixel values to push the image as quickly as possible to a different classification space (for example an image could be pushed from the dog classification space to the cat classification space). To prevent the adversarial perturbations from altering a human's perception of the image, the perturbations must remain small (changing each pixel by less than 5). Adding random noise to the image before classification causes the image to make random move in classification space. Adversarial images tend to exist in the sharp protrusions of a classification space so moving by moving randomnly we are more likely to move back into the correct category than to move further away. See the following diagrams for a visiualization. 

Take a simplified example of a classifier that takes two inputs – height and average number of seeds – and uses that to classify a plant as an orchid or not an orchid. Here an orchid is represented as blue and a not orchid is represented as orange. 

Take the following abnormal classification space: 

![](readme_images/Weird_Space.png)

The point circled in red is the closet blue area to the surrounding orange area, so many adversarial examples would be drawn to that point. Adversarial examples look for the shortest distance they have to travel to be classified as the other category. However if were to move in a random direction from that point we are more likely to return to the orange category than to move further into the blue space. This is the rational behind adding random noise. However, by the bigger step we take in a random direction the more likely we are to move into another classification space. Especially when we have many different categories. We did a grid search to find the ideal trade off between an increase in adversarial robustness and a loss in accuracy. 

#### Examples of Images

Unperturbed Images         |  Unperturbed Images with Noise     
:-------------------------:|:-------------------------:
![](readme_images/unperturbed1.JPEG)  |  ![](readme_images/unpuerturbed_noisy_1.JPEG) 

Universal Adversarial Images |  Universal Adversarial Images with Noise      
:-------------------------:|:-------------------------:
![](readme_images/universal_1.JPEG)  |  ![](readme_images/universal_noisy_1.JPEG) 

Gradient Adversarial Images |  Gradient Adversarial Images with Noise      
:-------------------------:|:-------------------------:
![](readme_images/gradient_1.JPEG)  |  ![](readme_images/gradient_noisy_1.JPEG)


##### Accuracy and Robustness After Adding Random Noise
Accuracy of Noisy Models on Original Images | 
:-------------------------:|
![](readme_images/Noisy_Model_on_unperturbed.png)  |


Accuracy of Noisy Models on Universal Adversarial Images | Robustness of Noisy Models on Universal Adversarial Images |
:----------------------------------------:| :-----------------------------------------------:|
![](readme_images/Universal_Accuracy.png)  | ![](readme_images/Universal_Robustness.png) |

Accuracy of Noisy Model on Gradient Adversarial Images of Control Model | Accuracy of Control Model on Gradient Adverarial Images of Noisy Model |
:--------------------------------:|:------------------------: |
Accuracy: 55.85% | Accuracy: ?????? |

##### Confusion Matrix
On Unperturbed Images              |  On Universal Adversarial Images |      
:-------------------------:|:-------------------------:|
![](Results/Confusion_Matrices/control_noisy_30.png) | ![](Results/Confusion_Matrices/universal_noisy_30.png)


### Build a Modified Inception Network Which Downsamples the Input Images

##### Motivation
Another way to reduce the dimesionality of the input images is to downsample. Downsampling reduces a 1,000 x 1,000 pixel image to some smaller n x n pixel image and then expands it back to a 1,000 x 1,000 pixel image again. This blurs the image which theorectically gets rid of the minor adversarial perturbations.


##### Accuracy

Downsampling by 50%                                | Downsampling by 75% |
:------------------------------------------------: | :-----------------: |
Accuracy on original images: 62.95%                | Accuracy on original images: 70.45% |
Accuracy on universal adversarial images: 54.30%   | Accuracy on universal adversarial images: 53.70% |
Robustness of Downsampling by 50% : 0.8625          | Robustness of Downsampling by 50% : 0.7622 |


##### Confusion Matrix
On Unperturbed Images              |  On Universal Adversarial Images |      
:-------------------------:|:-------------------------:|
![](Results/Confusion_Matrices/control_75.png) | ![](Results/Confusion_Matrices/universal_75.png)

  





