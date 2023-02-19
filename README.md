# Reproducibility Artifacts for Integrated Reproducibility with Self-describing Machine Learning Models

Our evaluation uses Docker and consists of three Jupyter notebooks that each train and then reproduce a different set of models. In total, they will reproduce 42 models, and save the results to a csv file. 

# Reproducing the results

The simplest way to reproduce the results is to pull the pre-built Docker image from Docker hub. This image almost 4GB so might take a few minutes to download. Using a computer with Docker installed, open a command-line prompt and run the following command. 
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

Navigate to the `eval` folder and there will be 3 notebooks along with some folders. Running each notebook from top to bottom in a linear fashion will train and then reproduce all 42 models. Some of these models might take a few minutes to train and re-train. 

Once all the notebooks have completed, open the `results` directory to see a csv file for each notebook. Within these files are columns named `Equivalent Evaluation` and `Model Prov Diff`. The models were all successfully reproduced if  `Equivalent Evaluation` is all true and `Model Prov Diff` only contains timestamps. 

Closing the terminal with the server running, or pressing ctrl+c within in will stop the server. 

## Building the Docker image (if pulling image doesn't work)

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
