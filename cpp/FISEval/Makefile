CC=g++
CFLAGS=-W -Wall -pedantic -fPIC -fopenmp -shared -march=native -ffast-math
LDFLAGS=
EXEC=fiseval.so
BUILD_PATH=./cpp/build
SRC_PATH=./cpp/src
SOURCES=$(SRC_PATH)/fis.cpp $(SRC_PATH)/fiseval_wrapper.cpp

all: $(EXEC)

fiseval.so:
	$(CC) $(CFLAGS) -o $(BUILD_PATH)/fiseval_wrapper.so $(SOURCES) $(LDFLAGS)

clean:
	rm -rf $(BUILD_PATH)/*.so

mrproper: clean
	rm -rf $(EXEC)

