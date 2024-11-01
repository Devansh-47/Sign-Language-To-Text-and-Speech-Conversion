**Sign-Language-To-Text-and-Speech-Conversion**

**ABSTRACT:** 

 Sign language is one of the oldest and most natural form of language for communication, hence we have come up with a real time method using neural networks for finger spelling based American sign language. Automatic human gesture recognition from camera images is an interesting topic for developing vision. We propose a convolution neural network (CNN) method to recognize hand gestures of human actions from an image captured by camera. The purpose is to recognize hand gestures of human task activities from a camera image. The position of hand and orientation are applied to obtain the training and testing data for the CNN. The hand is first passed through a filter and after the filter is applied where the hand is passed through a classifier which predicts the class of the hand gestures. Then the calibrated images are used to train CNN. 
 
 
 The Final Outcome Of Our Project...
 
 https://user-images.githubusercontent.com/99630855/201496149-e7004402-16b2-4d72-8e1a-ff20c422c565.mp4



**Introduction:**

 American sign language is a predominant sign language Since the only disability D&M people have been communication related and they cannot use spoken languages hence the only way for them to communicate is through sign language. Communication is the process of exchange of thoughts and messages in various ways such as speech, signals, behavior and visuals. Deaf and dumb(D&M) people make use of their hands to express different gestures to express their ideas with other people. Gestures are the nonverbally exchanged messages and these gestures are understood with vision. This nonverbal communication of deaf and dumb people is called sign language. 

In our project we basically focus on producing a model which can recognise Fingerspelling based hand gestures in order to form a complete word by combining each gesture. The gestures we aim to train are as given in the image below. 


![Spanish_SL](https://user-images.githubusercontent.com/99630855/201489493-585ffe5c-f460-402a-b558-0d03370b4f92.jpg)

**Requirements:**

More than 70 million deaf people around the world use sign languages to communicate. Sign language allows them to learn, work, access services, and be included in the communities.  

It is hard to make everybody learn the use of sign language with the goal of ensuring that people with disabilities can enjoy their rights on an equal basis with others. 

So, the aim is to develop a user-friendly human computer interface (HCI) where the computer understands the American sign language This Project will help the dumb and deaf people by making their life easy. 


**Objective:**
To create a computer software and train a model using CNN which takes an image of hand gesture of American Sign Language and shows the output of the particular sign language in text format converts it into audio format. 


**Scope:**
This System will be Beneficial for Both Dumb/Deaf People and the People Who do not understands the Sign Language. They just need to do that with sign Language gestures and this system will identify what he/she is trying to say after identification it gives the output in the form of Text as well as Speech format. 

**Modules:**

**Data Acquisition :**

The different approaches to acquire data about the hand gesture can be done in the following ways: 

It uses electromechanical devices to provide exact hand configuration, and position. Different glove-based approaches can be used to extract information. But it is expensive and not user friendly. 

In vision-based methods, the computer webcam is the input device for observing the information of hands and/or fingers. The Vision Based methods require only a camera, thus realizing a natural interaction between humans and computers without the use of any extra devices, thereby reducing costs.  The main challenge of vision-based hand detection ranges from coping with the large variability of the human hand’s appearance due to a huge number of hand movements, to different skin-color possibilities as well as to the variations in viewpoints, scales, and speed of the camera capturing the scene. 

 
![Screenshot (224)](https://user-images.githubusercontent.com/99630855/201489523-0804652e-1a38-4242-ad69-8bfafb25f55a.png)

 

**Data pre-processing and Feature extraction:**

In this approach for hand detection, firstly we detect hand from image that is acquired by webcam and for detecting a hand we used media pipe library which is used for image processing. So, after finding the hand from image we get the region of interest (Roi) then we cropped that image and convert the image to gray image using OpenCV library after we applied the gaussian blur .The filter can be easily applied using open computer vision library also known as OpenCV. Then we converted the gray image to binary image using threshold and Adaptive threshold methods. 

We have collected images of different signs of different angles  for sign letter A to Z. 

 ![fdfScreenshot (227)](https://user-images.githubusercontent.com/99630855/201489564-04b0416d-f976-4946-80d3-01bab6897ce3.png) 

- in this method there are many loop holes like your hand must be ahead of clean soft background and that is in proper lightning condition then only this method will give good accurate results but in real world we dont get good background everywhere and we don’t get good lightning conditions too. 

So to overcome this situation we try different approaches then we reached at one interesting solution in which firstly we detect hand from frame using mediapipe and get the hand landmarks of hand present in that image then we draw and connect those landmarks in simple white image  

Mediapipe Landmark System: 

![2410344](https://user-images.githubusercontent.com/99630855/201489741-3649959e-df4d-4c32-898a-8f994be92ca2.png)

![a12](https://user-images.githubusercontent.com/99630855/201490095-96402d48-b289-4ff3-9738-ed99ffcffca6.jpg)

![a23](https://user-images.githubusercontent.com/99630855/201490105-87b17583-45c5-4e3b-82d1-0c9a6f98fc55.jpg)

![7](https://user-images.githubusercontent.com/99630855/201490124-dc41d7ad-313f-47b7-b50c-0f9db3155e0d.jpg)

![b11](https://user-images.githubusercontent.com/99630855/201490119-55ff1b2d-1826-4bc6-994e-8c8c528c8c35.jpg)

![b16](https://user-images.githubusercontent.com/99630855/201490122-46d87005-ccb6-46ac-9dcf-185a569d6958.jpg)

![127](https://user-images.githubusercontent.com/99630855/201490130-b0aae39b-a623-4cf8-b41d-0611c02637ed.jpg)
 

Now we get this landmark points and draw it in plain white background using opencv library 

-By doing this we tackle the situation of background and lightning conditions because the mediapipe labrary will give us landmark points in any background and mostly in any lightning conditions. 


![2022-10-31](https://user-images.githubusercontent.com/99630855/201489669-1b262755-23f8-4e02-91ba-393aa6482620.png)
![2022-10-31 (1)](https://user-images.githubusercontent.com/99630855/201489673-08a8dad8-30a4-426a-8f62-02190416191d.png)

 ![hhee2022-10-31 (2)](https://user-images.githubusercontent.com/99630855/201496302-f67b360a-1ef5-4486-8ff7-cc56cee30b97.png)


-we have collected 180 skeleton images of Alphabets from A to Z 

**Gesture Classification :**

**Convolutional Neural Network (CNN)**

CNN is a class of neural networks that are highly useful in solving computer vision problems. They found inspiration from the actual perception of vision that takes place in the visual cortex of our brain. They make use of a filter/kernel to scan through the entire pixel values of the image and make computations by setting appropriate weights to enable detection of a specific feature. CNN is equipped with layers like convolution layer, max pooling layer, flatten layer, dense layer, dropout layer and a fully connected neural network layer. These layers together make a very powerful tool that can identify features in an image. The starting layers detect low level features that gradually begin to detect more complex higher-level features 

 

Unlike regular Neural Networks, in the layers of CNN, the neurons are arranged in 3 dimensions: width, height, depth. 

The neurons in a layer will only be connected to a small region of the layer (window size) before it, instead of all of the neurons in a fully-connected manner. 

Moreover, the final output layer would have dimensions(number of classes), because by the end of the CNN architecture we will reduce the full image into a single vector of class scores. 

CNN 

**Convolutional Layer:**

In convolution layer I have taken a small window size [typically of length 5*5] that extends to the depth of the input matrix. 

The layer consists of learnable filters of window size. During every iteration I slid the window by stride size [typically 1], and compute the dot product of filter entries and input values at a given position. 

As I continue this process well create a 2-Dimensional activation matrix that gives the response of that matrix at every spatial position. 

That is, the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some colour. 
![cnn](https://user-images.githubusercontent.com/99630855/201490154-1416d8ad-c7df-42a2-a296-5d56bad1d5c5.png)

**Pooling Layer:**

We use pooling layer to decrease the size of activation matrix and ultimately reduce the learnable parameters. 

There are two types of pooling: 

   **a. Max Pooling:**

In max pooling we take a window size [for example window of size 2*2], and only taken the maximum of 4 values. 

Well lid this window and continue this process, so well finally get an activation matrix half of its original Size. 

  **b. Average Pooling:** 

In average pooling we take average of all Values in a window. 

pooling 

![pooling](https://user-images.githubusercontent.com/99630855/201490158-22a8a043-c2fe-4082-8fb5-a6c173061b58.jpg)

**Fully Connected Layer:**

In convolution layer neurons are connected only to a local region, while in a fully connected region, well connect the all the inputs to neurons. 

Fully Connected Layer 

![fullyConnectedLayer](https://user-images.githubusercontent.com/99630855/201490169-00b17306-e355-4d2e-88e5-3fbd4c7b3f17.png)
 

The preprocessed 180 images/alphabet will feed the keras CNN model.  

Because we got bad accuracy in 26 different classes thus, We divided whole 26 different alphabets into 8 classes in which every class contains similar alphabets: 
[y,j] 

[c,o] 

[g,h] 

[b,d,f,I,u,v,k,r,w] 

[p,q,z] 

[a,e,m,n,s,t] 

All the gesture labels will be assigned with a  

probability. The label with the highest probability will treated to be the predicted label. 

So when model will classify [aemnst] in one single class using mathematical operation on hand landmarks we will classify further into single alphabet a or e or m or n or s or t. 

-Finally, we got **97%** Accuracy (with and without clean background and proper lightning conditions) through our method. And if the background is clear and there is good lightning condition then we got even **99%** accurate results 

![2022-11-01 (2)](https://user-images.githubusercontent.com/99630855/201489689-3adeacf0-ca19-471d-8942-cf7effc6296a.png)
![2022-11-01 (3)](https://user-images.githubusercontent.com/99630855/201489695-d14822c4-3a48-41c3-9cde-8fac4d835f65.png)
![2022-11-01 (4)](https://user-images.githubusercontent.com/99630855/201489697-168c8ca3-e4b3-4fc5-9e3a-97b239971c27.png)
![2022-11-01 (5)](https://user-images.githubusercontent.com/99630855/201489700-78b38657-16c0-45ac-88ec-ef06ada7870e.png)

**Text To Speech Translation:**

The model translates known gestures into words. we have used pyttsx3 library to convert the recognized words into the appropriate speech. The text-to-speech output is a simple workaround, but it's a useful feature because it simulates a real-life dialogue. 

**Project Requirements :**

**Hardware Requirement:**

Webcam 

**Software Requirement:**

Operating System: Windows 8 and Above 

IDE: PyCharm 

Programming Language: Python 3.9 5 

Python libraries: OpenCV, NumPy, Keras,mediapipe,Tensorflow 


**System Diagrams:**


**System Flowchart:**

![system flowchart](https://user-images.githubusercontent.com/99630855/201490238-224f65aa-071f-473a-8c23-a9d60e0a47d8.png)


**Use-case diagram:**

![Untitled Diagram drawio](https://user-images.githubusercontent.com/99630855/201490218-85f4c194-0496-4dfb-b920-e486256bd6b7.png)
 
**DFD diagram:**

![Flowcharts (2)](https://user-images.githubusercontent.com/99630855/201490221-f543fa6d-75ba-4db0-bc35-ee8c06e25018.png)
![Flowcharts (1)](https://user-images.githubusercontent.com/99630855/201490226-966bcc44-8149-433d-ab3b-b0a23deb1c91.png)
 

**Sequence diagram:**

![sequence2](https://user-images.githubusercontent.com/99630855/201490230-b903c365-7a4c-4972-8268-5687060b9cd0.png)

 
