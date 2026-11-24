CC ?= gcc
CFLAGS := -O3 -std=c11 -Wall -Wextra -Isrc -MMD -MP
LDLIBS := -lm

BIN_DIR := bin
BUILD_DIR := build

COMMON_SRCS := $(wildcard src/common/*.c)
COMMON_OBJS := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(COMMON_SRCS))

SERIAL_SRCS := $(wildcard src/serial/*.c)
SERIAL_OBJS := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(SERIAL_SRCS))
SERIAL_BIN := $(BIN_DIR)/quad_serial

DEPS := $(COMMON_OBJS:.o=.d) $(SERIAL_OBJS:.o=.d)

.PHONY: all serial clean

all: serial

serial: $(SERIAL_BIN)

$(SERIAL_BIN): $(COMMON_OBJS) $(SERIAL_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD_DIR)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

-include $(DEPS)
