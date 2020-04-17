# Disaster Response Messages - Multi-class classifier

Prepare, analyze and visualize disaster data provided by figure8 to build a Machine Learning model that classifies disaster messages based on their content. 

This project was developed as part of Udacity Data Science Nanodegree.

To classify a message, the user can input text in the text-input slot and then press __Classify Message__. The classes that are assigned as most appropriate to the message will be highlighted in green.

![Classify messages example](figures/example.png)

## Installation Requirements

The project uses scikit-learn and nltk with Python 3.*. A way to install the two packages locally can be to run the following commands:

`conda install -c intel scikit-lean`

`conda install -c anaconda nltk`

## Data
The data (and code) can be accessed within the repository in the _app/data_. The two csv files were provided by figure8 in partnership with Udacity for the Data Science Nanodegree.
The overview of the used data can be seen in the bellow chart:

![Training Data Overview](figures/pie-chart-messages.png)

## Files

* _app_ : The files in the _app_ folder are used to run the web application (as seen in the above screenshot). 

* _notebooks_ : The two provided jupyter notebooks contain the same code as the python scripts in the _app/data_ and _app/models_ folders, but they are structured in a step-by-step manner to process the data in an ETL pipeline, and then to train, evaluate and save the Machine Learning model, respectively.

* _figures_: contains the two images in the README file. 

## Running the App

* Run the following commands in the project's root directory to set up a new database and model.

    * To run ETL pipeline that cleans data and stores in database:

        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    * To run ML pipeline that trains classifier and saves it (a trained model is not available due to repository upload limitations):

        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl` 

* Run the following command in the app's directory to run your web app. 
    `python run.py`

* Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements

Must give credit to Figure Eight for the data and the stub files were provided through Udacity Data Science Nanodegree. 