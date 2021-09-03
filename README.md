# ai-for-healthcare-hackathon
Team Medical Explorers @ DERBI AI for Healthcare Hackathon Code Base

We used a total of 6 datasets for our project - 4 datasets with Optical Coherence Tomography (OCT) scans of the retina and 2 datasets with Fundoscopic images of the retina. Of these datasets, 3 were provided officially as a part of the Hackathon while 3 were obtained through external sources with appropriate permissions, compliance and attribution.

We developed a web application for showcasing the model developed for predicting diseases based on the ensemble model developed using the 2 best developed models shown. Web App was developed in Flask, with frontend developed using Bootstrap and Javascript used to improve responsiveness of the app. Users can insert any image and the model will predict the corresponding class with probability score as well as display the gradient heat map for the given image. Users can also view segmentation of any OCT image to see segmentation of the different retinal layers to identify key regions.

FINAL_kermany18+duke+tehran_xxx notebooks & Hackathon GUI folder have the code & Flask scripts for the implemented GUI.

Link to all pretrained models: https://www.kaggle.com/itsmariodias/pretrained-datasets

To Run GUI:
open terminal in /Hackathon GUI and enter "python app.py"

In browser type 'http://127.0.0.1:5002/' to access webpage.

There are 3 pages to showcase 3 different implementations.
Click on Browse and select appropriate image. Image will be displayed on screen
Then click Predict. After 2-4 seconds predictions will be given and heatmap is generated.
For segmentation the equivalent segment mask will be given instead.
Click clear, browse or reload page to perform predictions for other images.

Models with their weights can be found at https://www.kaggle.com/itsmariodias/pretrained-datasets
 

