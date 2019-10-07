FROM python:3.7
MAINTAINER joezuntz@googlemail.com
#Joe's note to himself.  Compile this with: docker build -t joezuntz/cosmosis-base .
#then docker push joezuntz/cosmosis-base

# Basic compilers and tools dependencies
RUN apt-get update -y && apt-get install -y gcc g++ gfortran \
    cmake swig wget make libopenblas-dev libchealpix-dev \
    pkg-config curl \
    && apt-get clean all

# Manual installation of mpich seems to be required to work on NERSC
RUN mkdir /opt/mpich && cd /opt/mpich \
    && wget http://www.mpich.org/static/downloads/3.3.1/mpich-3.3.1.tar.gz \
    && tar xvzf mpich-3.3.1.tar.gz &&  cd mpich-3.3.1 && ./configure --disable-wrapper-rpath && make -j4 \
    && make install && rm -rf /opt/mpich


# other dependencies, some for matplotlib
RUN apt-get update  && \
	apt-get install -y  libcfitsio-dev  libfftw3-dev   git  libgsl-dev \
    && apt-get clean all


# Need to install manually because we want MPI-enabled version
RUN mkdir /opt/hdf && cd /opt/hdf \
    && wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz \
    && tar xvf hdf5-1.10.5.tar.gz \
    && cd hdf5-1.10.5 \
    && ./configure --enable-parallel --prefix=/usr/local CC=mpicc \
    && make \
    && make install \
    && rm -rf /opt/hdf



# Also need a manual install of mpi4py so that it uses the right libraries - pip seems not to work
RUN mkdir /opt/mpi4py && cd /opt/mpi4py \
    && wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.2.tar.gz \
    && tar -zxvf mpi4py-3.0.2.tar.gz && cd mpi4py-3.0.2 && python setup.py install \
    && rm -rf /opt/mpi4py


# Some env vars
ENV HDF5_MPI ON
ENV HDF5_USE_FILE_LOCKING FALSE
ENV LD_LIBRARY_PATH /usr/local/lib
ENV C_INCLUDE_PATH /usr/local/include

# Ceci

RUN pip install --no-cache-dir --upgrade pip

# fitsio needs this to be installed first
RUN pip install --no-cache-dir numpy==1.17

# Basic python requirements
RUN pip install --no-cache-dir \
                Cython==0.29.13 \
                nose==1.3.7 \
                pyparsing==2.2.0 \
                pyyaml==5.1.2 \
                scikit-learn==0.21.3 \
                setuptools==38.5.2 \
                astropy==3.2.1 \
                scipy==1.3.1 \
                fitsio==0.9.11 \
                healpy==1.12.9 \
                treecorr \
                descformats \
                sacc \
                mlz_desc

# Install h5py, recompiling so that we use HDF5 with MPI
RUN pip install --no-cache-dir --no-binary=h5py h5py==2.9.0

RUN pip install --no-cache-dir  git+https://github.com/LSSTDESC/ceci@minirunner

# Manual namaster install
RUN mkdir -p /opt/namaster && cd /opt/namaster \
    && git clone https://github.com/LSSTDESC/NaMaster \
    && git clone https://github.com/Libsharp/libsharp \
    && cd libsharp \
    && autoconf \
    && ./configure --enable-pic  --disable-openmp \
    && make \
    && cp auto/lib/* /usr/local/lib/ \
    && cp auto/include/* /usr/local/include/ \
    && cd ../NaMaster \
    && ./configure --disable-openmp \
    && make  \
    && make install \
    && python setup.py install \
    && rm -rf /opt/namaster

# Manual CCL install
RUN cd /opt \
    && git clone https://github.com/LSSTDESC/CCL.git \
    && cd CCL \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install \
    && cd .. \
    && python setup.py install \
    && rm -rf /opt/CCL


WORKDIR /opt/txpipe
CMD ["/bin/bash", "-l"]
