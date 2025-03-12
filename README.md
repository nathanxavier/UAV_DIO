# **UAV DIO**
## ***Unmanned Aerial Vehicle Deep Inertial Odometry***

[Nathan A. Z. Xavier](http://lattes.cnpq.br/2088578568009855),
[Elcio H. Shiguemori](http://lattes.cnpq.br/7243145638158319),
[Marcos R. O. A. Maximo](http://lattes.cnpq.br/1610878342077626)

<p align="center">
<img src="https://github.com/nathanxavier/UAV_DIO/blob/701b1211c8c59c28623a351f632b636eb7d26b24/Figures/Engineering%20Analysis%20with%20Boundary%20Elements.png">
</p>


# **Overview**
This repository contains the implementation for our paper on **deep inertial navigation for UAV pose estimation**, leveraging deep neural networks to address the challenge of self-localization without global positioning.

## ***Highlights:***
- **Deep Learning for Inertial Navigation:** Accurate UAV pose estimation using real outdoor flight data.  
- **Cross-Platform Transfer:** Seamless performance transition from a real hexacopter to a simulated quadcopter.  
- **High Accuracy:** Achieving an R-squared value of 70% in a simulated UAV flight cycle with limited error.  
- **Noise Reduction:** Outperforming traditional Kalman filters in mitigating noise from off-the-shelf inertial sensors.

This repository provides all the necessary code for training, evaluation, and reproducing the results described in the paper.


# **Installation**
## ***Python Dependencies:***
- Python >= 3.8
- TensorFlow >= 2.4
- Other dependencies listed in `requirements.txt`

## ***CoppeliaSim Dependencies:***
The repository uses **RemoteAPI** in **CoppeliaSim** (formerly V-REP) to enable external applications to communicate with the simulator over a network. This Remote API is available in multiple languages, including **Python**, **C++**, **Lua**, **Java**, and others.

For more detailed instructions on simulator use, please look at the official [CoppeliaSim User Manual](https://manual.coppeliarobotics.com/index.html).


# **Training and Evaluation Data**
The training data is from a proprietary dataset provided by the Brazilian Air Force, consisting of operational, training, and research flight data. While not publicly available, the UAV used in this study shares similarities with those in the [NTU VIRAL dataset](https://doi.org/10.1177/02783649211052312) and the [CLOUD dataset](https://www.dynsyslab.org/cloud-dataset).

## ***Testing Stage on a Real Flight***
The first testing stage simulated a real flight using the same UAV from the training data, conducted during the Exercício Geral Integrado de Resposta à Emergência e Segurança Física Nuclear in 2023 in Angras dos Reis, Brazil, coordinated by the Brazilian Air Force. The flight lasted over 6 minutes, with the UAV reaching 120 meters, performing a circular motion for nearly 4 minutes, and landing in the final minute. The trajectory includes movement in the north, east, height, and yaw rotation. The proposed DIO techniques were selected based on their higher R² scores and suitable inference times.

<p align="center">
  <img src="https://github.com/nathanxavier/UAV_DIO/blob/f0c464c43296a785207d4464a172ed4d65704c5a/Figures/FLY_Teste.png">
  <br>
  <em>Flight trajectory in a real flight.</em>
  <br><br>
  <img src="https://github.com/nathanxavier/UAV_DIO/blob/35acbf377ab9084f21e4c81fc4b0f58ea261d02c/Figures/FLY_Compara.png">
  <br>
  <em>Comparison of position estimation errors across techniques.</em>
</p>

The EKF quickly drifts within the first minute, and when the UAV moves upward, DIO solutions show increased errors in the north and east, with poor height prediction. The Updated 9-Axis IONEt has the largest errors (30m and 25m), while the Updated AbolDeepIO has the smallest (29m and 20m). Yaw prediction is most accurate with the EKF and IONet, with errors of 20 and 30 degrees, while the Updated 9-Axis IONEt and Updated AbolDeepIO show a near-linear increase, reaching 100 degrees after 4 minutes. Overall, all DIO techniques limited the inertial navigation error, with a maximum error of around 200 meters after 3 minutes without GNSS.

## ***Testing Stage on CoppeliaSim***
The testing stage was conducted in the CoppeliaSim simulator using a standard quadcopter model. The UAV, equipped with an accelerometer and gyroscope, performed a flight simulation with synthetic sensor errors. The simulation involved sinusoidal orbits along all three axes, starting from an altitude of -1m. Each DIO technique estimated position and yaw orientation using dead reckoning, and the results were compared to a Kalman filter approach.

<p align="center">
  <img src="https://github.com/nathanxavier/UAV_DIO/blob/f0c464c43296a785207d4464a172ed4d65704c5a/Figures/Coppelia_Teste.png">
  <br>
  <em>Flight trajectory during the simulation.</em>
  <br><br>
  <img src="https://github.com/nathanxavier/UAV_DIO/blob/35acbf377ab9084f21e4c81fc4b0f58ea261d02c/Figures/Coppelia_Compara.png">
  <br>
  <em>Comparison of position estimation errors across techniques.</em>
</p>

In the one-minute simulated flight, the EKF showed a rapid increase in error in the north and east directions. The Updated Nine-Axis IONet technique followed the same error slope as the EKF in the first 30 seconds but had lower error afterward. The Updated IONet technique had the lowest error in the north, while the Updated AbolDeepIO had the lowest error in the east. None of the techniques were able to accurately predict height movement, as expected. For yaw rotation, the EKF consistently predicted the angle accurately, while the DIO techniques exhibited gradual drift due to the lack of continuous corrections. The Updated AbolDeepIO showed the lowest yaw error, while the Updated IONet displayed the highest. The test highlighted that, while the DIO techniques and EKF performed similarly for short periods, continuous corrections are crucial for precise inertial navigation, particularly for yaw estimation. The methods also demonstrated good cross-platform compatibility, performing well when switching from a hexacopter to a quadcopter, revealing both strengths and limitations in dynamic flight conditions.
