CC=g++
CFLAGS=-O3 -ftree-vectorize -mavx -W -std=c++17 -Wall -pedantic -fPIC -fopenmp -shared -mtune=native -march=native -mfpmath=sse -DDEBUG  -isystem cpp/vendor/eigen/
# CFLAGS=-W -std=c++17 -Wall -pedantic -fPIC -fopenmp -shared -mtune=native -march=native -mfpmath=sse -DDEBUG  -isystem cpp/vendor/eigen/
CFLAGS_TESTS=-W -std=c++17 -Wall -pedantic -fPIC -fopenmp -march=native -ffast-math -isystem cpp/vendor/eigen/ -Icpp/vendor/catch/
LDFLAGS=
EXEC=fiseval.so
BUILD_PATH=./cpp/build
SRC_PATH=./cpp/src
SOURCES=$(SRC_PATH)/fis.cpp
SOURCES_TESTS=$(wildcard ./cpp/tests/*.cpp)

# EXTERNAL LIBS VERSIONS
EIGEN_VERSION=3.3.4
CATCH_VERSION=2.2.1

all: $(EXEC)

prepare: eigen catch2

eigen:
	wget https://github.com/eigenteam/eigen-git-mirror/archive/$(EIGEN_VERSION).tar.gz -O eigen.tar.gz
	mkdir -p cpp/vendor/eigen
	tar -xvzf eigen.tar.gz eigen-git-mirror-$(EIGEN_VERSION)/Eigen -C cpp/vendor/eigen --strip-components=1
	mv Eigen cpp/vendor/eigen/.
	rm eigen.tar.gz

catch2:
	mkdir -p cpp/vendor/catch
	wget https://github.com/catchorg/Catch2/releases/download/v${CATCH_VERSION}/catch.hpp -O cpp/vendor/catch/catch.hpp

fiseval.so:
	$(CC) $(CFLAGS) -o $(BUILD_PATH)/fiseval_wrapper.so $(SOURCES) $(LDFLAGS)

clean:
	rm -rf $(BUILD_PATH)/*.so

mrproper: clean
	rm -rf $(EXEC)

test:
	$(CC) $(CFLAGS_TESTS) -o $(BUILD_PATH)/run_tests $(SOURCES) $(SOURCES_TESTS) $(LDFLAGS) && $(BUILD_PATH)/run_tests --success
