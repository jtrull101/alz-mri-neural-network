# alz-mri-neural-network
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<br />
<div align="center">
  <a href="https://github.com/jtrull101/alz-mri-neural-network">
    <img src="images/006-11.jpg" alt="Logo" width="80" height="80">
    <br>
    <font size="-50">
      <a href="https://www.vecteezy.com/free-vector/brain">Brain Vectors by Vecteezy</a>
    </font>
  </a>
  
  
  <h3 align="center">Alzheimers MRI Scan Neural Network</h3>

  <p align="center">
    Simple Convolutional Neural Network intended to diagnose fictitious MRI images, labeling new inputs with an estimated Impairment level ranging from No Impairment to Moderate Impairment.
    <br>
    · <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    · <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</div>



<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#why-make-this-project">Why Make This Project?</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



## Why Make This Project?

![Product Name Screen Shot](images/screenshot.jpg)

In a deeply personal quest to combat a devastating and pervasive disease, I embarked on a journey to develop a neural network model to read fictitious MRI data and detect the early signs of Alzheimer's disease. This endeavor is motivated by a profound family history that has been haunted by Alzheimer's for generations. The emotional toll of Alzheimer's, coupled with the urgent need for early diagnosis and intervention, fueled my determination to make a difference.

My journey began by delving into the world of medical imaging and artificial intelligence. I gathered fictitious MRI data, sourced from [kaggle.com](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy), which mirrors the complexities of real-world medical images, to construct a neural network model. This model has been meticulously trained to analyze subtle patterns and anomalies within the brain, with a specific focus on identifying the early indicators of Alzheimer's disease.

My hope is that this project will serve as a beacon of hope for others facing Alzheimer's, a testament to the power of technology, and a tribute to the loved ones who have inspired it. Together, we can shine a light on this dark path and take meaningful steps toward early diagnosis, treatment, and ultimately, a cure for Alzheimer's disease.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

Below are the major Python frameworks used for this project

<ul>
  <li>[Tensorflow](https://www.tensorflow.org/)</li>
  <li>[Keras](https://keras.io/)</li>
  <li>[Flask](https://flask.palletsprojects.com/en/3.0.x/)</li>
</ul>
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started
Installing this software is as simple as the few steps below:

<ol>
  <li>First, pull this git repo to a local directory and cd into that directory</li>
  <li>Next, upgrade pip:
    
    python -m pip install --upgrade pip
  <li>Then pip install the included requirements.txt file</li>

    pip install -r requirements.txt
    
</ol>
  Now you are ready to run tests or the front-end application.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

This software utility offers a simple and accessible way to predict Alzheimer's disease progression using MRI data. Here's how to use it:

Starting the Web User Interface (UI):

    ./start_front_end.sh

  Sending MRI Files:
  <br>
  <p>
  To predict Alzheimer's disease progression, send MRI image files to the http://127.0.0.1:5000/predict endpoint of the Flask server. Note these MRI files should be one of the fictitious MRIs present in the data/test/ directory. This model has zero ability to predict actual MRI images.
  </p>
  Interpreting Predictions:
  <br>
  <p>
  The utility will return a prediction for the diagnosis, categorizing it into one of four levels of impairment:
  </p>
  <ul>
    <li>No Impairment</li>  
    <li>Very Mild Impairment</li>
    <li>Mild Impairment</li>  
    <li>Moderate Impairment</li>  
  </ul>
  
  <br>
  Using the Web UI:
  <br>
  <p>
  Accessing the Web Interface:
        Open your web browser and navigate to http://127.0.0.1:5000. This will take you to a basic Graphical User Interface (GUI).
  </p>
  <br>
  Selecting Impairment Categories:
  <br>
  <p>
  On the web UI, you will find four buttons, each corresponding to a different level of impairment. By clicking on one of these buttons, you can initiate a prediction for that specific impairment category.
  </p>
  <br>
  Testing the Model:
  <p>
  The utility will randomly select an MRI image from the training set for the chosen impairment category and run it through the predictive model.

  This utility provides a convenient and user-friendly way to predict Alzheimer's disease progression, making it accessible to both professionals and non-experts. By following these simple steps, you can quickly assess the likelihood of Alzheimer's disease  in MRI images and gain insights into its progression.
  </p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Jonathan Trull - jttrull0@gmail.com

Project Link: [https://github.com/jtrull101/alz-mri-neural-network](https://github.com/jtrull101/alz-mri-neural-network)

[![LinkedIn][linkedin-shield]][linkedin-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/jonathan--trull
![Tests](https://github.com/jtrull101/alz-mri-neural-network/actions/workflows/tests.yml/badge.svg)
