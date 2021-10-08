FROM centos:8

# Necessary for using conda
ENV PATH="/usr/local/bin/conda/bin:${PATH}"
ARG PATH="/usr/local/bin/conda/bin:${PATH}"

# Install necessary dependencies 
RUN yum install -y wget git unzip libgomp maven 

# Install OpenJDK 16
RUN wget https://download.java.net/java/GA/jdk16.0.2/d4a915d82b4c4fbb9bde534da945d746/7/GPL/openjdk-16.0.2_linux-x64_bin.tar.gz \
    && tar xvf openjdk-16*_bin.tar.gz \
    && mv jdk-16.0.2 /opt/ \
    && rm openjdk-16*_bin.tar.gz 
   
# Ensure System and Maven use Java 16
ENV JAVA_VER 16.0.2
ENV JAVA_HOME /opt/jdk-16.0.2
ENV PATH $PATH:$JAVA_HOME/bin
RUN update-alternatives --install /usr/bin/java java /opt/jdk-16.0.2/bin/java 1 \ 
    && update-alternatives --set java /opt/jdk-16.0.2/bin/java

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
RUN git clone https://github.com/oracle/olcut.git \
	&& cd olcut \
	&& mvn clean install 

# Install and patch tribuo
RUN git clone https://github.com/jwons/tribuo.git \
	&& cd tribuo \ 
	&& git checkout fm \
	&& mvn clean install \

# adding eval scripts goes in the next stage

