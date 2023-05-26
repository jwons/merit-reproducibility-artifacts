# Reproducibility Artifacts for "Integrated Reproducibility with Self-describing Machine Learning Models"

The paper presents MERIT, a reproducibility system fully integrated into the Tribuo ML library. To use the most up-to-date version of MERIT yourself, you can add Tribuo to your Java projects using Maven, Gradle, or from source - [installation and documentation can be found here](https://tribuo.org). Otherwise, this document explains how to execute our evaluation with the version of MERIT we used. 

Our evaluation uses Docker and consists of three Jupyter notebooks that each train and then reproduce a different set of models. In total, they will train and then reproduce 48 models, and save the results to a csv file.

The results from when we originally ran this evaluation are stored in the top-level directory titled `author-results`, with the models we trained stored in the top-level directory `author-models`.

## Table of Contents
1. [Intro](#reproducibility-artifacts-for-integrated-reproducibility-with-self-describing-machine-learning-models)
2. [Table of Contents](#table-of-contents)
3. [Estimated Time and Resources](#estimated-time-and-resources)
4. [Reproducing the results](#reproducing-the-results)
5. [Building the Docker image (if pulling the image doesn't work)](#building-the-docker-image-if-pulling-the-image-doesnt-work)
6. [Using MERIT and Tribuo on your own](#using-merit-and-tribuo-on-your-own)
7. [Explanation of directory structure](#directory-structure)


## Estimated Time and Resources

These instructions have been tested on a machine with 32GB RAM and an Intel Xeon E-2276M CPU @ 2.80GHz
At a minimum, you need to be able to provide the JVM with ~4GB of heap space. 

We reproduced this work using the following instructions on a fresh machine in ~10 minutes. Included within the time was downloading the image, running the experiments, and checking the results. 

# Reproducing the results

The simplest way to reproduce the results is to pull the pre-built Docker image from Docker Hub. This image is almost 4GB so it might take a few minutes to download. To pull the image: use a computer [with Docker installed](https://docs.docker.com/get-docker/), open a command-line prompt and run the following command. 
```
docker pull jwonsil/merit-artifact
```
(If you'd rather build the image yourself, check the next section.)

Once you have pulled this image, start the local Jupyter Lab server using the following command.
```
docker run --rm -p 8888:8888 jwonsil/merit-artifact
```
It will take a few seconds to start up, once it has there will be instructions on how to access it. 
```
To access the server, open this file in a browser:
        file:///home/jupyter/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://1a6bda217368:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
     or http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```
Ctrl+Click the bottom-most link that starts with `http://127.0.0.1:8888` to be taken to the Jupyter Lab instance. 

Navigate to the `eval` folder and there will be 3 notebooks along with some folders. Running each notebook from top to bottom in a linear fashion will train and then reproduce all 48 models. Some of these models might take a few minutes to train and re-train. 

Once all the notebooks have completed, open the `results` directory to see a csv file for each notebook. Within these files are columns named `Equivalent Evaluation` and `Model Prov Diff`. The models were all successfully reproduced if  `Equivalent Evaluation` is all true and `Model Prov Diff` only contains timestamps. 

Closing the terminal with the server running, or pressing ctrl+c within in will stop the server. 

## Building the Docker image (if pulling the image doesn't work)

You should not need to build the image yourself, but in case you want to try here are the instructions. Pulling the pre-built image is likely faster, and more guaranteed to work in the off chance one of the sites hosting a dependency linked in the Dockerfiles no longer works.

These steps are for linux machines with Docker installed, they build the image and will likely take a while. 
```
cd tribuo-env
docker build . -t tribuo-env
cd ..

docker build . -t tribuo-notebook
```

To run the image as a container:
```
docker run --rm -p 8888:8888 tribuo-notebook
```
Then browse to localhost:8888 in your browser. Ctrl+c in the terminal where you ran the container kills the notebook server. 

# Using MERIT and Tribuo on your own
Tribuo and MERIT can be easily included into Java projects through Maven, Gradle, or linking to prebuilt jars. Visit [tribuo.org](https://tribuo.org) to learn how. 

# Directory Structure

```
📂 Repository Root
├── 📂author-models # All the models we trained during our evaluation
│   ├── 📂classification 
│   │   ├── 📜3-nn.model
│   │   ...
│   │   └── 📜rf.model
│   └── 📂regression
│       ├── 📜3-nn.model
│       ...
│       └── 📜rf-reg.model
├── 📂author-results # Our results from running these experiments
│   ├── 📜configResults.csv
│   ├── 📜multilabelResults.csv
│   └── 📜results.csv
├── 📂eval # The main directory for our evaluation, contains the notebooks, data, intermediate files, and results.
│   ├── 📂configs
│   │   ├── 📜all-classification-config.xml
│   │   ...
│   │   └── 📜mnist-config.xml
│   ├── 📂data # This is populated once the container is built
│   ├── 📂models # Trained models are saved here
│   ├── 📂results # The results of the experiments are saved here
│   ├── 📜reproduce-models.ipynb # Experimental script
│   ├── 📜reproduce-multilabel-config.ipynb # Experimental script
│   └── 📜reproduce-from-configs.ipynb # Experimental script
├── 📂example-provenance # Contains example provenance objects that are too big to put in the paper. Each one is about ~2000 lines of "pretty printed" provenance 
│   ├── 📜ExtraTreesModelProvenance.txt
│   └── 📜RandomForestModelProvenance.txt
├── 📂 reproduce-serialized # This directory contains a Java program for reproducing serialized models for the cross-architecture eval in the from of a unit test. 
│   ├── 📜pom.xml
│   └── 📂src
│       ├── 📂main
│       │   └── 📂java
│       │       └── 📜TestReproduction.java
│       └── 📂test
│           ├── 📂java
│           │   └── 📜TestReproductionTest.java
│           └── 📂resources
│               ├── 📂data
│               │   ├── 📜bezdekIris.data
│               │   └── 📜winequality-red.csv
│               └── 📂models
│                   ├── 📂classification
│                   │   ├── 📜3-nn.model
│                   │   ...
│                   │   └── 📜rf.model
│                   └── 📂regression
│                       ├── 📜3-nn.model
│                       ...
│                       └── 📜rf-reg.model
├── 📂tribuo-env
│   └── 📜Dockerfile # Builds an environment tribuo can run in, lengthy build time as it install many dependencies
├── 📜Dockerfile # Copies this repo into the tribuo-env container, occurs quickly as it uses the prebuilt image.
├── 📜LICENSE
└── 📜README.md

```
