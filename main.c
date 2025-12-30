#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mnist.h"

static void print_error(const char *prefix, int err)
{
    fprintf(stderr, "%s: %s\n", prefix, strerror(err));
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

    int err = mnist_load_images(argv[1], &images);
    if (err)
    {
        print_error("Failed to load images", err);
        return 1;
    }

    err = mnist_load_labels(argv[2], &labels);
    if (err)
    {
        print_error("Failed to load labels", err);
        mnist_free_images(&images);
        return 1;
    }

    printf("Loaded %u images (%ux%u) and %u labels\n", images.count, images.rows, images.cols,
           labels.count);

    size_t ascii_size = 0;
    err = mnist_ascii_buffer_size(&images, &ascii_size);
    if (!err)
    {
        char *ascii = malloc(ascii_size);
        if (!ascii)
        {
            err = ENOMEM;
        }
        else
        {
            err = mnist_render_ascii(&images, 0, ascii, ascii_size);
            if (!err)
            {
                fputs(ascii, stdout);
            }
            free(ascii);
        }
    }

    if (err)
    {
        print_error("Could not render sample", err);
    }

    mnist_free_images(&images);
    mnist_free_labels(&labels);
    return 0;
}
