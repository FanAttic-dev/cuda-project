TARGET = matmult

all: $(TARGET)

$(TARGET): $(TARGET:=.o)
	nvcc -O3 -o $@ $< --run

$(TARGET:=.o): $(TARGET:=.cu)
	nvcc -O3 -o $@ $< -c

clean:
	-$(RM) *.o

distclean: clean
	-$(RM) $(TARGET)

.PHONY: all clean distclean
