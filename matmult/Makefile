TARGET = matmult
FLAGS = -O3 -lineinfo

all: $(TARGET)

large_blocks: FLAGS += -D "LARGE_BLOCKS"
large_blocks: $(TARGET)
	./$<

tiled: FLAGS += -D "TILED"
tiled: $(TARGET)
	./$<

naive: $(TARGET)
	./$<

$(TARGET): .FORCE
	nvcc $(FLAGS) -o $@ $(TARGET:=.cu)

clean:
	-$(RM) *.o

distclean: clean
	-$(RM) $(TARGET)

.PHONY: all naive .FORCE tiled large_blocks clean distclean

.FORCE:
