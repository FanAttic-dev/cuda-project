TARGET = framework
SRCS = framework.cu
FLAGS = -O3 -lineinfo

all: $(TARGET)

$(TARGET): distclean
	nvcc $(FLAGS) -o $@ $(SRCS)
	./$@

clean:
	-$(RM) *.o

distclean: clean
	-$(RM) $(TARGET)

.PHONY: all clean distclean
