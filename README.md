# tribuo-reproducibility-artifacts

Instrucions to bring up Jupyter Notebook with Tribuo

These steps build the image and will take a while.
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
