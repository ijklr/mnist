#ifndef MNIST_H
#define MNIST_H

#include <stddef.h>
#include <stdint.h>

typedef struct
{
    uint32_t count;
    uint32_t rows;
    uint32_t cols;
    uint8_t *pixels; /* count * rows * cols bytes */
} MnistImages;

typedef struct
{
    uint32_t count;
    uint8_t *labels; /* count bytes */
} MnistLabels;

int mnist_load_images(const char *path, MnistImages *out);
int mnist_load_labels(const char *path, MnistLabels *out);

void mnist_free_images(MnistImages *images);
void mnist_free_labels(MnistLabels *labels);

/* Compute required buffer size for mnist_render_ascii; returns 0 on success. */
int mnist_ascii_buffer_size(const MnistImages *images, size_t *out_size);

/* Render an image into the provided buffer using '.', '+', '#'. */
int mnist_render_ascii(const MnistImages *images, uint32_t index, char *dest, size_t dest_size);

#endif /* MNIST_H */
