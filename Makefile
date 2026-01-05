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
	@python scripts/plot_training.py logs/training.csv --out plots/single_layer_loss.png --title "Single-layer Perceptron Training Loss" --subtitle "Single-layer Perceptron — LR=0.1, Epochs=2000"

clean:
	rm -f $(TARGET) logs/*.csv plots/*.png
