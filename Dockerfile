FROM tribuo-env

RUN yum -y install bzip2


COPY eval ./eval
COPY author-results ./author-results
COPY reproduce-serialized ./eval/reproduce-serialized
COPY author-models ./author-models

RUN chown -R jupyter:jupyter eval \
	&& chown -R jupyter:jupyter tribuo \
	&& chown -R jupyter:jupyter author-models

# Download Data
RUN cd eval/data \
	&& wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data \
	&& wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv \
	&& wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
	&& wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz \
	&& wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/yeast_train.svm.bz2 \
	&& wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel/yeast_test.svm.bz2 \ 
	&& bzip2 -d yeast_train.svm.bz2 \
	&& bzip2 -d yeast_test.svm.bz2 \
	&& wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz \
	&& mkdir 20news \
	&& cd 20news \
	&& tar -zxf ../20news-bydate.tar.gz 

# Copy data to necessary location
RUN cp eval/data/winequality-red.csv eval/reproduce-serialized/src/test/resources/data/ \
	&& cp eval/data/bezdekIris.data eval/reproduce-serialized/src/test/resources/data/ 

RUN cd tribuo/tutorials \
	&& wget https://repo1.maven.org/maven2/com/fasterxml/jackson/datatype/jackson-datatype-jsr310/2.14.0/jackson-datatype-jsr310-2.14.0.jar

USER jupyter

# Jupyter lab will be exposed on port 8888
# It is important to still execute docker run -p 8888:8888 <image_name> 
# As this command will not actually make the connection
EXPOSE 8888

CMD ["jupyter", "lab","--ip=0.0.0.0"]


