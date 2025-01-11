# UAV DIO
## Unmanned Aerial Vehicle Deep Inertial Odometry

[Nathan A. Z. Xavier](http://lattes.cnpq.br/2088578568009855),
[Elcio H. Shiguemori](http://lattes.cnpq.br/7243145638158319),
[Marcos R. O. A. Maximo](http://lattes.cnpq.br/1610878342077626)

A solution combining 
<p align="center">
<img src="https://github.com/nathanxavier/UAV_DIO/blob/701b1211c8c59c28623a351f632b636eb7d26b24/Figures/Engineering%20Analysis%20with%20Boundary%20Elements.png">
</p>

This repository contains the implementation for our paper on **deep inertial navigation for UAV pose estimation**, leveraging deep neural networks to address the challenge of self-localization without global positioning.
- **Deep Learning for Inertial Navigation:** Accurate UAV pose estimation using real outdoor flight data.  
- **Cross-Platform Transfer:** Seamless performance transition from a real hexacopter to a simulated quadcopter.  
- **High Accuracy:** Achieving an R-squared value of 70% in a simulated UAV flight cycle with limited error.  
- **Noise Reduction:** Outperforming traditional Kalman filters in mitigating noise from off-the-shelf inertial sensors.

This repository provides all the necessary code for training, evaluation, and reproducing the results described in the paper.

## Installation
Dependencies:
- Python >= 3.8
- TensorFlow >= 2.4
- Other dependencies listed in `requirements.txt`

![Preview](https://github.com/tmKamal/hosted-images/blob/master/under-construction/Document.gif?raw=true)<br/>  
