CC := nvcc
CE := .cu
HE := .hpp
FLAGS := -lm -ccbin g++ -m64 --threads 0 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -G

SRC_DIR := src/
BUILD_DIR := build/

TARGET := main

SRCS := $(shell find $(SRC_DIR) -name '*$(CE)')
HEADERS := $(shell find $(SRC_DIR) -name '*$(HE)')

OBJS = $(patsubst $(SRC_DIR)%$(CE), $(BUILD_DIR)%.o, $(SRCS))

BUILD_DIRS = $(patsubst $(SRC_DIR)%, $(BUILD_DIR)%, $(dir SRCS))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $^ $(FLAGS) -o $(BUILD_DIR)$@

build/convolution.o: src/convolution.cu
	mkdir -p $(BUILD_DIR)$(patsubst $(SRC_DIR)%,%, $(dir $<))
	$(CC) $< $(FLAGS) -c -o $@

$(BUILD_DIR)%.o: $(SRC_DIR)%.$(CE)
	mkdir -p $(BUILD_DIR)$(patsubst $(SRC_DIR)%,%, $(dir $<))
	$(CC) $< $(FLAGS) -c -o $@
