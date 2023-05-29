CC := nvcc
CE := .cu
HE := .hpp
FLAGS := -lm

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
