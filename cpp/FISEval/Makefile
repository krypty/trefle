CC=g++
CFLAGS=-W -Wall -pedantic -fPIC -fopenmp -shared -march=native -ffast-math -isystem cpp/eigen/
LDFLAGS=
EXEC=fiseval.so
BUILD_PATH=./cpp/build
SRC_PATH=./cpp/src
SOURCES=$(SRC_PATH)/fis.cpp $(SRC_PATH)/fiseval_wrapper.cpp
EIGEN_VERSION=3.3.4

all: $(EXEC)

prepare: eigen

eigen:
	wget https://github.com/eigenteam/eigen-git-mirror/archive/$(EIGEN_VERSION).tar.gz -O eigen.tar.gz
	mkdir -p cpp/eigen
	tar -xvzf eigen.tar.gz eigen-git-mirror-$(EIGEN_VERSION)/Eigen -C cpp/eigen --strip-components=1
	mv Eigen cpp/eigen/.
	rm eigen.tar.gz

fiseval.so:
	$(CC) $(CFLAGS) -o $(BUILD_PATH)/fiseval_wrapper.so $(SOURCES) $(LDFLAGS)

clean:
	rm -rf $(BUILD_PATH)/*.so

mrproper: clean
	rm -rf $(EXEC)

