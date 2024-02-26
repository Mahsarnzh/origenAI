# PINNs for Lid Driven Cavity Flow using Navier Stokes Equations

I employed two distinct approaches for predicting velocities in the x and y directions as well as pressure. Initially, I used the deepxde library and associated packages, graciously provided by Dr. George Karniadakis, the originator of the PINNs method. Subsequently, I opted for a second method where I independently implemented the PINNs methodology from scracth.

As a side note, the "Navier_Stokes.py" file focuses on solving the lid-driven cavity flow using numerical methods. Although not directly involved in the test, its inclusion significantly contributed to my understanding of the problem's structure.


Contours of predictions using deepxde library:
![Sample Image](https://github.com/Mahsarnzh/origenAI/blob/main/Question_2/contour_01.png)

Contours of predictions using my method, PINN_scratch.py:
![Sample Image](https://github.com/Mahsarnzh/origenAI/blob/main/Question_2/PINNs_Scratch_Contour_02.png)
