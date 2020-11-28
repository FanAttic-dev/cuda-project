TARGET = vecsum
FLAGS = -O3 -lineinfo

all: $(TARGET)

$(TARGET): .FORCE
	nvcc $(FLAGS) -o $@ $(TARGET:=.cu)
	./$@

clean:
	-$(RM) *.o

distclean: clean
	-$(RM) $(TARGET)

.PHONY: all .FORCE clean distclean

.FORCE:
