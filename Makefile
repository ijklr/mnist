CC=gcc
CFLAGS=-O2 -std=c11 -Wall -Wextra
TARGET=single_layer_perceptron.exe
SRC=single_layer_perceptron.c

.PHONY: all build run log plot clean

all: build

build:
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) -lm

run: build
	@echo "Running $(TARGET)"
	@./$(TARGET)

log: build
	@python scripts/run_and_log.py --exe ./$(TARGET) --out logs/training.csv

plot:
	@python scripts/plot_training.py logs/training.csv

clean:
	rm -f $(TARGET) logs/*.csv plots/*.png
