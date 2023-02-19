FROM dokken/centos-8

# Necessary for using conda
ENV PATH="/usr/local/bin/conda/bin:${PATH}"
ARG PATH="/usr/local/bin/conda/bin:${PATH}"

# Install necessary dependencies 
RUN yum install -y wget git unzip libgomp 

RUN wget https://dlcdn.apache.org/maven/maven-3/3.8.7/binaries/apache-maven-3.8.7-bin.zip \
	&& unzip apache-maven-3.8.7-bin.zip \
	&& rm apache-maven-3.8.7-bin.zip

ENV PATH="/apache-maven-3.8.7/bin:${PATH}"
 

# Install OpenJDK 17
RUN wget https://download.java.net/java/GA/jdk17.0.2/dfd4a8d0985749f896bed50d7138ee7f/8/GPL/openjdk-17.0.2_linux-x64_bin.tar.gz \
    && tar xvf openjdk-17*_bin.tar.gz \
    && mv jdk-17.0.2 /opt/ \
    && rm openjdk-17*_bin.tar.gz 
   
# Ensure System and Maven use Java 17
ENV JAVA_VER 17.0.2
ENV JAVA_HOME /opt/jdk-17.0.2
ENV PATH $PATH:$JAVA_HOME/bin
RUN update-alternatives --install /usr/bin/java java /opt/jdk-17.0.2/bin/java 1 \ 
    && update-alternatives --set java /opt/jdk-17.0.2/bin/java

# Install conda, which is used to install jupyterlab the notebook environment used by tribuo tutorials
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local/bin/conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda install -y -c conda-forge jupyterlab nodejs

# Install IJava: the java notebook kernel Tribuo uses to work with Jupyter
RUN wget \
    https://github.com/SpencerPark/IJava/releases/download/v1.3.0/ijava-1.3.0.zip \
    && mkdir ijava \
    && unzip -d ijava ijava-1.3.0.zip \
    && cd ijava \
    && python install.py --sys-prefix \
    && cd .. \
    && rm -rd ijava*

# Create a user so as not to run in privileged mode 
RUN useradd -ms /bin/bash jupyter
WORKDIR /home/jupyter

# Install OLCUT 
#RUN git clone https://github.com/oracle/olcut.git \
#	&& cd olcut \
#	&& mvn -X clean install \

# Install tribuo
RUN git clone https://github.com/jwons/tribuo.git \
	&& cd tribuo \ 
    && git checkout reproducibility \
	&& mvn clean install
	
# adding eval scripts goes in the next stage
