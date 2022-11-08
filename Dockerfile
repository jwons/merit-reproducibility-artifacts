FROM tribuo-env

RUN yum -y install bzip2

# Copy in the whole directory so we can use the patch files
COPY eval ./eval
COPY author-results ./author-results

RUN chown -R jupyter:jupyter eval \
	&& chown -R jupyter:jupyter tribuo 

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

RUN cd tribuo/tutorials \
	&& wget https://repo1.maven.org/maven2/com/fasterxml/jackson/datatype/jackson-datatype-jsr310/2.14.0/jackson-datatype-jsr310-2.14.0.jar

#%jars ../Classification/Experiments/target/tribuo-classification-experiments-5.0.0-SNAPSHOT-jar-with-dependencies.jar
#%jars ../Json/target/tribuo-json-5.0.0-SNAPSHOT-jar-with-dependencies.jar
#%jars ./jackson-datatype-jsr310-2.14.0.jar

#import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
#import com.fasterxml.jackson.datatype.jsr310.*; 

#ObjectMapper objMapper = new ObjectMapper();
#objMapper.registerModule(new JsonProvenanceModule());
#objMapper.registerModule(new JavaTimeModule());
#objMapper = objMapper.enable(SerializationFeature.INDENT_OUTPUT);

USER jupyter

# Jupyter lab will be exposed on port 8888
# It is important to still execute docker run -p 8888:8888 <image_name> 
# As this command will not actually make the connection
EXPOSE 8888

CMD ["jupyter", "lab","--ip=0.0.0.0"]


