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
    <td>Background and reference review. Both students will deliver a list of key references and a brief summary of the overall approaches & future trends.</td>
    <td>Week 4</td>
    <td>Vignesh and Chester</td>
    <td>Low</td>
  </tr>
  <tr>
    <td>Exploratory analysis & feature selection (Feature Engineering and Analysis). Jupyter Notebook with results </td>
    <td>Week 5</td>
    <td>Chester</td>
    <td>High</td>
  </tr>
  <tr>
    <td>Exploratory analysis & feature selection (Explore Deep Learning Models (no hand-engineered features)). Jupyter Notebook with results </td>
    <td>Week 5</td>
    <td>Vignesh</td>
    <td>High</td>
  </tr>

  <tr>
    <td>Model Construction. Build classical machine learning models to fit data.  </td>
    <td>Week 6</td>
    <td>Chester</td>
    <td>High</td>
  </tr>
  <tr>
    <td>Model Construction. Build Deep Learning Models.  </td>
    <td>Week 6</td>
    <td>Vignesh</td>
    <td>High</td>
  </tr>

  <tr>
    <td>Evaluation and Results. Evaluate all the models built and generate results. </td>
    <td>Week 7</td>
    <td>Vignesh and Chester</td>
    <td>Medium</td>
  </tr>

  <tr>
    <td>Reproducibility. Work on documentation and code cleaning to make sure results are reproducible.</td>
    <td>Week 8</td>
    <td>Vignesh and Chester</td>
    <td>Medium</td>
  </tr>

  <tr>
    <td>Draft final write up </td>
    <td>Week 9</td>
    <td>Vignesh and Chester</td>
    <td>Medium</td>
  </tr>

</table>

  





