#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mnist.h"

static void print_error(const char *prefix, int err) { fprintf(stderr, "%s: %s\n", prefix, strerror(err)); }

static void print_load_error(const char *kind, int err)
{
    if (err == EINVAL)
    {
        fprintf(stderr, "Failed to load %s: invalid MNIST IDX header or magic\n", kind);
    }
    else
    {
        print_error(kind, err);
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <images idx file> <labels idx file>\n", argv[0]);
        return 1;
    }

    MnistImages images = {0};
    MnistLabels labels = {0};
    int exit_code = 0;

    int err = mnist_load_images(argv[1], &images);
    if (err)
    {
        print_load_error("images", err);
        return 1;
    }

    err = mnist_load_labels(argv[2], &labels);
    if (err)
    {
        print_load_error("labels", err);
        mnist_free_images(&images);
        return 1;
    }
    if (labels.count != images.count)
    {
        fprintf(stderr, "Label count (%u) does not match image count (%u)\n", labels.count,
                images.count);
        mnist_free_images(&images);
        mnist_free_labels(&labels);
        return 1;
    }

    printf("Loaded %u images (%ux%u) and %u labels\n", images.count, images.rows, images.cols,
           labels.count);

    size_t ascii_size = 0;
    int render_err = mnist_ascii_buffer_size(&images, &ascii_size);
    if (!render_err)
    {
        char *ascii = malloc(ascii_size);
        if (!ascii)
        {
            render_err = ENOMEM;
        }
        else
        {
            render_err = mnist_render_ascii(&images, 0, ascii, ascii_size);
            if (!render_err)
            {
                fputs(ascii, stdout);
            }
            free(ascii);
        }
    }

    if (render_err)
    {
        print_error("Could not render sample", render_err);
        exit_code = 1;
    }

    mnist_free_images(&images);
    mnist_free_labels(&labels);
    return exit_code;
}
