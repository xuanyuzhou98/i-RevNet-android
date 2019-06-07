# OnDeviceTrain
Train Neural Network on mobile devices

# Our goal
1. Amount of data is small: Pretrained model + few shot learning: the training data in devices is small.
2. Explore more data types: Currently only using images as input.  Support machine translation etc.
3. Moving from algorithm to application: Write a mobile application with on device training.

# Motivations for On-Device Training
1. Customization: Customized model weights for each device
2. Economy: No data transmission overhead. 
3. Privacy: Data does not leave devices

# Constraints of On-Device Training
1. Memory Limit: 
Total Memory = Parameter Memory + Gradient Memory + Layer Activations Memory. 
2. Speed Limit (FLOPS):
Not a hard limit. We don’t need training to be finished in real-time.
3. Energy/Battery Limit:
Not a hard limit. We can limit training to be only executed during charging.


# Papers for reference
1. [i-RevNet: Deep Invertible Networks](https://arxiv.org/pdf/1802.07088.pdf)

2. [The Reversible Residual Network: Backpropagation Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)

3. [Sample Efficient Adaptive Text-To-Speech](https://arxiv.org/pdf/1809.10460.pdf)

4. [Wavenet: A Generative Model For Raw Audio](https://arxiv.org/pdf/1609.03499.pdf)

5. [Low-Memory Neural Network Training: A Technical Report](https://arxiv.org/pdf/1904.10631.pdf)

6. [Weight Standardization, Training with Batch Size 1](https://arxiv.org/pdf/1903.10520.pdf)

# Useful Resources
1. The library deeplearning4j:
https://github.com/deeplearning4j/deeplearning4j

2. PyTorch code for i-revnet:
https://github.com/jhjacobsen/pytorch-i-revnet

3. Tensorflow code for revnet:
https://github.com/renmengye/revnet-public

4. Deep neural networks for voice conversion (voice style transfer) in Tensorflow:
https://github.com/andabi/deep-voice-conversion

5. Surface Inspection defect detection dataset:
https://github.com/abin24/Surface-Inspection-defect-detection-dataset

6. Lumber grading dataset (image tagging):
http://www.ee.oulu.fi/~olli/Projects/Lumber.Grading.html

7. A blog about Training on the device:
https://machinethink.net/blog/training-on-device/

