FROM tribuo-env

RUN yum -y install bzip2

# Copy in the whole directory so we can use the patch files
COPY results ./results
COPY configResults.csv ./configResults.csv

RUN chown -R jupyter:jupyter results \
	&& chown -R jupyter:jupyter tribuo \
	&& chown -R jupyter:jupyter olcut 

# Reproduciblity Jar
RUN cp tribuo/Reproducibility/target/tribuo-reproducibility-4.2.0-SNAPSHOT-jar-with-dependencies.jar ./results 

# Irises classification
RUN cd results \
	&& cp ../tribuo/Classification/Experiments/target/tribuo-classification-experiments-4.2.0-SNAPSHOT-jar-with-dependencies.jar ./ \
	&& cp ../tribuo/Json/target/tribuo-json-4.2.0-SNAPSHOT-jar-with-dependencies.jar ./ \
	&& wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data

# Regression 
RUN cd results \ 
	&& cp ../tribuo/Regression/SGD/target/tribuo-regression-sgd-4.2.0-SNAPSHOT-jar-with-dependencies.jar ./ \
	&& cp ../tribuo/Regression/XGBoost/target/tribuo-regression-xgboost-4.2.0-SNAPSHOT-jar-with-dependencies.jar ./ \
	&& cp ../tribuo/Regression/RegressionTree/target/tribuo-regression-tree-4.2.0-SNAPSHOT-jar-with-dependencies.jar ./ \
	&& wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

# Configuration
RUN cd results \
	&& wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
	&& wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Clustering 
RUN cd results \
	&& cp ../tribuo/Clustering/KMeans/target/tribuo-clustering-kmeans-4.2.0-SNAPSHOT-jar-with-dependencies.jar ./

# Anomaly Detection with LibSVM
RUN cd results \
	&& cp ../tribuo/AnomalyDetection/LibSVM/target/tribuo-anomaly-libsvm-4.2.0-SNAPSHOT-jar-with-dependencies.jar ./
	
# Feature extraction
RUN cd results \
	&& cp ../tribuo/Interop/ONNX/target/tribuo-onnx-4.2.0-SNAPSHOT-jar-with-dependencies.jar ./ \
	&& wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz \
	&& mkdir 20news \
	&& cd 20news \
	&& tar -zxf ../20news-bydate.tar.gz 

RUN cd results \
	&& wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/yeast_train.svm.bz2 \
	&& wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/yeast_test.svm.bz2 \ 
	&& bzip2 -d yeast_train.svm.bz2 \
	&& bzip2 -d yeast_test.svm.bz2
	

USER jupyter

# Jupyter lab will be exposed on port 8888
# It is important to still execute docker run -p 8888:8888 <image_name> 
# As this command will not actually make the connection
EXPOSE 8888

CMD ["jupyter", "lab","--ip=0.0.0.0"]


