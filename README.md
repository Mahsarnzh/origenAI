# Well Oil Prediction

# PINNs for Lid Driven Cavity Flow using Navier Stokes Equations

I employed two distinct approaches for predicting velocities in the x and y directions as well as pressure. Initially, I used the deepxde library and associated packages, graciously provided by Dr. George Karniadakis, the originator of the PINNs method. Subsequently, I opted for a second method where I independently implemented the PINNs methodology from scracth.

As a side note, the "Navier_Stokes.py" file focuses on solving the lid-driven cavity flow using numerical methods. Although not directly involved in the PINNs, its inclusion significantly contributed to my understanding of the problem's structure.


Contours of predictions using deepxde library, PINN_dpxde.py:
![Sample Image](https://github.com/Mahsarnzh/origenAI/blob/main/lid_driven_cavity_PINNS/PINNs_Scratch_Contour_01.png)

Contours of predictions using my method, PINN_scratch.py:
![Sample Image](https://github.com/Mahsarnzh/origenAI/blob/main/lid_driven_cavity_PINNS/PINNs_Scratch_Contour_02.png)


Welcome to my project! This repository contains an awesome project that contains two parts:
1. Various deep learning methods for predicting WOPR data using LSTM and CNN.
2. Two methods for predicting velocitie and pressure using PINNs.

## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

- Python 3.11
- pip

  
For mac users:

### Installation
Clone the repository:

   ```
   git clone https://github.com/Mahsarnzh/origenAI.git
   ```

### Navigate to the project directory

  ```
  cd origenAI
  ```

### Create a virtual environment:

  ```
  python3.11 -m venv venv
  ```

### Activate the virtual environment:

  ```
  source venv/bin/activate
  ```


### Install dependencies:
  ```
  pip install -r requirements.txt
  ```
