CC = clang++
prefix = /usr/local
CPPFLAGS = -Wall -O3 -std=c++14 -I.. -I. -I$(prefix)/include -fPIC\
		   -Xpreprocessor -fopenmp -I$(prefix)/opt/libomp/include\
		   -I$(prefix)/include/eigen3 -Wno-unknown-warning-option


hard_sphere.o: hard_sphere.cpp hard_sphere.hpp
	$(CC) $(CPPFLAGS) -c hard_sphere.cpp -o hard_sphere.o


test: test.cpp hard_sphere.o
	rm -f test.xyz
	$(CC) $(CPPFLAGS) test.cpp hard_sphere.o -o test
	time ./test
	rm test test.xyz
