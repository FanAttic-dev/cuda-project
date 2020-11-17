TARGET = matmult
FLAGS = -O3

all: $(TARGET)

tiled: $(TARGET)
	./$<

naive: FLAGS += -D "NAIVE"
naive: $(TARGET)
	./$<

$(TARGET): .FORCE
	nvcc $(FLAGS) -o $@ $(TARGET:=.cu)

clean:
	-$(RM) *.o

distclean: clean
	-$(RM) $(TARGET)

.PHONY: .FORCE all tiled naive clean distclean

.FORCE:
