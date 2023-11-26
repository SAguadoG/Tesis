# Tesis doctoral 
## Sergio Aguado González
## Universidad de Cádiz

## Abstract

With the high demand and usage of electronic devices, renewable energies, and the development of so-called Smart Grids connected to the grid, the available electrical waveform has significantly deviated from nominal production values (shape, amplitude, and/or frequency), resulting in negative consequences for both the electrical system and consumers.

Consequently, the detection and classification of power quality disturbances (PQDs) have gained importance in recent years, providing instrumental means and techniques to aid in network design and operation improvement, as well as the promotion of good consumption practices.
Many studies have demonstrated high reliability using intelligent classifiers with different methods, although, in most of these studies, experiments are conducted with simulated values, including significant signal noise. However, when attempting to use PQD classifiers on real data, reliability drops drastically and/or results are not shown directly.

This thesis will focus on the study of disturbances in power lines, their automatic detection and classification, both for simple and individual disturbances and combined disturbances. To achieve this, feature extraction techniques based on the calculation of temporal and frequency parameters will be used, along with soft computing techniques from the field of artificial intelligence, specifically Deep Learning and Neural Networks.

The proposed tasks, initially, to achieve the research objectives are related to two main aspects: signal analysis/feature extraction and signal classification with disturbances. These tasks are detailed in the Methodology section.

The general working scheme, assumed by the research community, involves obtaining a feature vector (feature extraction stage) characteristic of a waveform segment. Subsequently, a properly trained neural network classifies the segment or vector into a disturbance category. In a first approach, the training, validation, and testing of the neural network are performed with synthetic waveforms, i.e., computer-generated using parametric mathematical models aligned with the EN 50160 standard. The disturbance categories considered are sag, swell, interruption, impulsive transient, and oscillatory transient, which are part of the two major families of electrical disturbances: continuous phenomena and transient phenomena. The set of waveforms is later expanded to include other artificial waveforms obtained through computer simulation of electrical networks and, finally, with real cases. The possibility of multiple disturbances, i.e., waves affected by two or more different disturbance categories, will also be taken into account.

For the study with real cases, time series measurements mainly published in the "IEEE Datasets" database, which are part of the knowledge base of the research group PAIDI-TIC-168 (Computational Instrumentation and Industrial Electronics - ICEI), will be used.

The most relevant expected results in this study are as follows:

- Identify the set of features or characteristics of the supply signal that best characterizes the type of disturbance present.
- Achieve a reliable and robust automatic classifier in both Matlab/Simulink and open-source software (Python) with simulated and real data.
- Compare both classifiers with the intention of assessing whether open-source software can match or improve computation times, resource consumption, and error rates.
- Study and comparison of implementation solutions that the networks offer in both systems.