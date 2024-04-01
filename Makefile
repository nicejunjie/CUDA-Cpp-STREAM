CC = pgc++
ARCH=sm_90a
MODEL = -DZERO_COPY
CPPFLAGS=-DNTIMES=20 $(MODEL)


stream : stream.cu Makefile
	nvcc $(CPPFLAGS) -std=c++11 -ccbin=$(CC) stream.cu -arch=$(ARCH) -o stream 

.PHONY: clean
clean :
	rm -f stream
