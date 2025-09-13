# Exploring Deep Learning and Data Representations for the Prediction of Erosion Channels

## Abstract
- Importance: 
  - Accurate forecasting evolution of erosion channels is essential for protecting buildings, dams and other earth structures.
- In the paper, we consider a simplified channel system and two machine learning challenges based on thousands of high-fidelity simulations:
  - single-frame prediction of a channel's ultimate direction (left, right, split)
  - Long-horizon synthesis of its full trajectory from an early image or image sequence.
- Final direction prediction:
  - Lightweight convolutional network:
    - Unexpectedly robust when inputs are aggressively down-sampled or binarized.
  - Hybrid Autoencoder-CatBoost pipeline:
    - Better early-warning accuracy
- Path prediction:
  - Recurrent video model (PredRNN-V2):
    - Excels at intersection-over-union
  - Latent-space autoencoder:
    - Predicts finer path geometry deeper into the future, especially for binary images.
* Findings quanitfy how model architecture, training-set size, and image fidelity appect actionable lead time in erosion monitoring.


## CSS Concepts
1. Applied Computing ->
   1. Physical Sciences and Engineering
   2. Physics
2. Computing Methodoligies ->
   1. Neural Networks
   2. Supervised Learning

## Keywords
1. Deep Learning
2. Erosion Channel Direction Prediction
3. Erosion Path Prediction

## Introduction
* Channel networks are complex systems. Shaping surface and subsurface landscapes through erosion, sediment transport, and deposition.
* Systems Like:
  1. Branching patterns of river basins
  2. Intricate pathways within natural aquifers
  3. Engineered porous media
* Surface channels are often shaped gradually over time persistent river currents.
* Extreme events like dam failures or floods can produce rapid and dramatic morphological changes.
* In subsurface, fluid flow through porous media can erode and reorganize internal stuctures, altering hydrolic properties and connectivity.
* Chemical dissolution further contributes to the development of voids and channels
* Emergence and evolution of such channels are governed by a highly nonlinear interplay of hydrodynamic forces, material het- erogeneity, and chemical interactions, rendering their development inherently difficult to predict.
* Influenced not only by local material evolution but also by the spatial distribution of water sources. 
* Despite diversity of the systems, they are governed by shared physical principles
* A Laplacian framework, for example, has been employed to model two-dimensional groundwater flow under the Dupuit approximation
* This formulation enabled the development of physics-based models capable of reproducing erosion patterns observed in controlled laboratory experiments under defined boundary conditions 
* significant challenges remain in extending these models to natural settings, where complete boundary and initial conditions are rarely known. 
* Addressing the challenges will advance ability to forecast landscape evolution, optimize the design of engineered systems for water management, environmental remediation, and resource extraction.
* Fluid motion in porous media obeys Darcy’s law
* Erosion begins where |v𝑓| exceeds a critical threshold
* We adopt the hybrid erosion model of Kudrollietal.
  * partitions the porous matrix into immobile grains, mobile grains, and fluid
* Hydrodynamic stress mobilizes grains, which advect with the flow at reduced speed and redeposit when slowed or obstructed.
* Interfaces of contrasting porosity thereforeconcentrate erosion and seed channel growth.
* systematic data-driven forecasting of erosion channels is still rare
* Several ML methods were applied to identify the primary drivers of soil erosion
* prior studies cover a broad and diverse set of problems, they differ substantially from the erosion forecasting challenges we address here
* most closely related prior work is the master’s thesis by Lyu:
  * Used a physics based simulation model to address (i) a binary erosion type classification task (uniform vs branched patterns) and (ii) long-horizon prediction of uniform river network images with a spatiotemporal Long Short-Term Memory (SPLSTM) model. 
* their study showed the efficacy of the specific SPLSTM model they used for the prediction of erosion networks.
* To our knowledge, no published work has rigorously tested whether modern deep networks can recover both the ultimate direc- tion and the detailed path of channel growth from early-time snapshots under a strictly leak-free protocol. 


## Problem Formulation
* We study a simplified channel system in which a single conduit originates at the bottom–centre of a square porous domain and grows upward, eventually veering left, right, or splitting into both directions. 
* domain has two fluid inlets (top corners) and one outlet (bottom centre). 
* Channel evolution is recorded as a sequence of 200 grayscale frames, each of size 200 × 200 pixels.
* Visible growth typically begins near frame 20 and is essentially complete by frame 110;
* final frame serves as the class label for a run.
* We consider two complementary ways to represent the image data:
  * First retains the raw grayscale output, preserving information about surrounding porous matrix.
  * Second converts each frame to a binary map that isolates channel pixels and sets background to zero.
* Which representation ultimately proves more informative is an empirical question addressed in our study.
* Regardless of representation, we cast channel evolution into two concrete prediction problems.
  * 1. Final direction Prediction: given a single intermediate frame, predict whether the run will end left, right, or split.
  * 2. Path forecasting: given the first 𝑘 frames, generate the channel image at any future step 𝑡 > 𝑘, predicting its trajectory.




# Questions:
1. Outcomes of incorrect erosion prediction
2. Light weight convolutional network:
   1. Downsampled, binarized
3. Hybrid Autoencoder-Catboost pipeline
4. Recurrent video model (PredRNN-v2)
   1. Intersection over union
5. Latent space autoencoder
   1. Predict finer path geometry esp for binary images
6. Diff between sediment transport and deposition
7. Engineered porous media
8. fluid flow through porous media can erode and reorganize internal stuctures, altering hydrolic properties and connectivity.
9. Chemical dissolution
   1.  particularly in carbonate-rich environments,
10. rendering their development inherently difficult to predict. What are factors that apply here?
11. formation and evolution of these channel systems are influenced not only by local material evolution but also by the spatial distribution of water sources.
12. Despite diversity of systems, they are governed by shared physical principles, making them amenable to unifying theoretical frameworks. 
13. Laplacian framework, Dupuit approximation
14. environmental remediation
15. Darcey's Law
    1.  pore pressure (p)
    2.  evolving permeability (k)
    3.  Local fluid velicity is v_f = -k * Delta P
16. hybrid erosion model of Kudrollietal.
17. * Hydrodynamic stress mobilizes grains, which advect with the flow at reduced speed and redeposit when slowed or obstructed.
18. * Interfaces of contrasting porosity thereforeconcentrate erosion and seed channel growth.
19. systematic data-driven forecasting of erosion channels is still rare
20. A comprehensive review of recent advances and ongoing challenges in ML for porous media organizes the work in six application domains: heat exchangers and thermal storage, energy storage and combustion, electrochemical systems, hydrocarbon reservoirs, carbon capture and sequestration, and groundwater 
21. most closely related prior work is the master’s thesis by Lyu:
  * Used a physics based simulation model to address 
  * (i) a binary erosion type classification task (uniform vs branched patterns) and 
  * (ii) long-horizon prediction of uniform river network images with a spatiotemporal Long Short-Term Memory (SPLSTM) model. 
22. spatiotemporal Long Short-Term Memory (SPLSTM) model
23. Figure 2 plots the total grayscale intensity in the right half of each frame against that in the left half at steps 30, 90, and 200. Early in the run the three outcome classes overlap almost completely; only by the final frame does a partial separation emerge, implying that coarse pixel-sum statistics confer little discriminative power.