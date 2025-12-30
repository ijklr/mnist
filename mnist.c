#include "mnist.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum
{
    MNIST_IMAGE_MAGIC = 2051,
    MNIST_LABEL_MAGIC = 2049
};

static uint32_t read_be32(FILE *f, int *err)
{
    /* Read a big-endian 32-bit value and propagate errno-style failures */
    uint8_t buf[4];
    if (fread(buf, 1, sizeof(buf), f) != sizeof(buf))
    {
        *err = errno ? errno : EIO;
        return 0;
    }
    return (uint32_t)buf[0] << 24 | (uint32_t)buf[1] << 16 | (uint32_t)buf[2] << 8 |
           (uint32_t)buf[3];
}

static int alloc_pixels(uint32_t count, uint32_t rows, uint32_t cols, uint8_t **out_pixels,
                        size_t *out_total)
{
    if (rows == 0 || cols == 0)
    {
        return EINVAL;
    }

    /* Guard against overflow: count * rows * cols */
    if (cols != 0 && rows > SIZE_MAX / cols)
    {
        return EOVERFLOW;
    }
    size_t rows_cols = (size_t)rows * cols;
    if (rows_cols != 0 && count > SIZE_MAX / rows_cols)
    {
        return EOVERFLOW;
    }

    size_t total_pixels = rows_cols * count;
    uint8_t *pixels = malloc(total_pixels);
    if (!pixels)
    {
        return ENOMEM;
    }

    *out_pixels = pixels;
    *out_total = total_pixels;
    return 0;
}

int mnist_load_images(const char *path, MnistImages *out)
{
    memset(out, 0, sizeof(*out));

    /* Parse IDX image header: magic, count, rows, cols */
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        return errno;
    }

    int err = 0;
    uint32_t magic = read_be32(f, &err);
    uint32_t count = read_be32(f, &err);
    uint32_t rows = read_be32(f, &err);
    uint32_t cols = read_be32(f, &err);

    if (err)
    {
        fclose(f);
        return err;
    }
    if (magic != MNIST_IMAGE_MAGIC)
    {
        fclose(f);
        return EINVAL;
    }

    size_t total_pixels = 0;
    err = alloc_pixels(count, rows, cols, &out->pixels, &total_pixels);
    if (err)
    {
        fclose(f);
        return err;
    }

    /* Allocate and read raw pixel bytes (one byte per pixel) */
    size_t read = fread(out->pixels, 1, total_pixels, f);
    fclose(f);
    if (read != total_pixels)
    {
        free(out->pixels);
        memset(out, 0, sizeof(*out));
        return EIO;
    }

    out->count = count;
    out->rows = rows;
    out->cols = cols;
    return 0;
}

int mnist_load_labels(const char *path, MnistLabels *out)
{
    memset(out, 0, sizeof(*out));

    /* Parse IDX label header: magic, count */
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        return errno;
    }

    int err = 0;
    uint32_t magic = read_be32(f, &err);
    uint32_t count = read_be32(f, &err);

    if (err)
    {
        fclose(f);
        return err;
    }
    if (magic != MNIST_LABEL_MAGIC)
    {
        fclose(f);
        return EINVAL;
    }

    /* Allocate and read label bytes */
    out->labels = malloc(count);
    if (!out->labels)
    {
        fclose(f);
        return ENOMEM;
    }

    size_t read = fread(out->labels, 1, count, f);
    fclose(f);
    if (read != count)
    {
        free(out->labels);
        memset(out, 0, sizeof(*out));
        return EIO;
    }

    out->count = count;
    return 0;
}

void mnist_free_images(MnistImages *images)
{
    free(images->pixels);
    memset(images, 0, sizeof(*images));
}

void mnist_free_labels(MnistLabels *labels)
{
    free(labels->labels);
    memset(labels, 0, sizeof(*labels));
}

int mnist_ascii_buffer_size(const MnistImages *images, size_t *out_size)
{
    if (!images || images->rows == 0 || images->cols == 0)
    {
        return EINVAL;
    }
    /* Need rows*cols chars plus newline per row and trailing NUL */
    if (images->cols != 0 && images->rows > (SIZE_MAX - 1) / images->cols)
    {
        return EOVERFLOW;
    }
    size_t pixels = (size_t)images->rows * images->cols;
    if (images->rows > SIZE_MAX - pixels - 1)
    {
        return EOVERFLOW;
    }

    *out_size = pixels + images->rows + 1;
    return 0;
}

int mnist_render_ascii(const MnistImages *images, uint32_t index, char *dest, size_t dest_size)
{
    if (!images || !dest || index >= images->count)
    {
        return EINVAL;
    }

    size_t required = 0;
    int err = mnist_ascii_buffer_size(images, &required);
    if (err)
    {
        return err;
    }
    if (dest_size < required)
    {
        return ENOMEM;
    }

    /* Render one image as ASCII using a coarse brightness map */
    uint32_t rows = images->rows;
    uint32_t cols = images->cols;
    const uint8_t *img = images->pixels + (size_t)index * rows * cols;
    char *cursor = dest;

    for (uint32_t r = 0; r < rows; r++)
    {
        for (uint32_t c = 0; c < cols; c++)
        {
            uint8_t px = img[r * cols + c];
            *cursor++ = px > 128 ? '#' : px > 64 ? '+' : '.';
        }
        *cursor++ = '\n';
    }

    *cursor = '\0';
    return 0;
}
