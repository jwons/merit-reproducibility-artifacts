FROM tribuo-env

RUN yum -y install bzip2

# Copy in the whole directory so we can use the patch files
COPY eval ./eval
COPY author-results ./author-results

RUN chown -R jupyter:jupyter eval \
	&& chown -R jupyter:jupyter tribuo \
	&& chown -R jupyter:jupyter olcut 

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

USER jupyter

# Jupyter lab will be exposed on port 8888
# It is important to still execute docker run -p 8888:8888 <image_name> 
# As this command will not actually make the connection
EXPOSE 8888

CMD ["jupyter", "lab","--ip=0.0.0.0"]


