#define _POSIX_C_SOURCE 200809L

#include "mnist.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int tests_run = 0;
static int failures = 0;

#define ASSERT_TRUE(cond, msg)                                                                   \
    do                                                                                           \
    {                                                                                            \
        if (!(cond))                                                                             \
        {                                                                                        \
            fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg);                      \
            failures++;                                                                          \
            return;                                                                              \
        }                                                                                        \
    } while (0)

#define ASSERT_EQ(expected, actual, fmt)                                                         \
    do                                                                                           \
    {                                                                                            \
        if ((expected) != (actual))                                                              \
        {                                                                                        \
            fprintf(stderr, "[FAIL] %s:%d: expected " fmt ", got " fmt "\n", __FILE__, __LINE__, \
                    (expected), (actual));                                                       \
            failures++;                                                                          \
            return;                                                                              \
        }                                                                                        \
    } while (0)

static void write_be32(FILE *f, uint32_t value)
{
    uint8_t buf[4] = {
        (uint8_t)(value >> 24), (uint8_t)(value >> 16), (uint8_t)(value >> 8), (uint8_t)value};
    fwrite(buf, 1, sizeof(buf), f);
}

static int create_images_file(uint32_t magic, uint32_t count, uint32_t rows, uint32_t cols,
                              const uint8_t *pixels, size_t pixel_count, char *out_path,
                              size_t out_size)
{
    size_t expected = (size_t)count * rows * cols;
    if (expected != pixel_count)
    {
        return EINVAL;
    }

    char tmpl[] = "tmp-img-XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd == -1)
    {
        return errno;
    }

    FILE *f = fdopen(fd, "wb");
    if (!f)
    {
        close(fd);
        return errno;
    }

    write_be32(f, magic);
    write_be32(f, count);
    write_be32(f, rows);
    write_be32(f, cols);
    fwrite(pixels, 1, pixel_count, f);
    fclose(f);

    strncpy(out_path, tmpl, out_size);
    out_path[out_size - 1] = '\0';
    return 0;
}

static int create_labels_file(uint32_t magic, uint32_t count, const uint8_t *labels,
                              char *out_path, size_t out_size)
{
    char tmpl[] = "tmp-labels-XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd == -1)
    {
        return errno;
    }

    FILE *f = fdopen(fd, "wb");
    if (!f)
    {
        close(fd);
        return errno;
    }

    write_be32(f, magic);
    write_be32(f, count);
    fwrite(labels, 1, count, f);
    fclose(f);

    strncpy(out_path, tmpl, out_size);
    out_path[out_size - 1] = '\0';
    return 0;
}

static void test_load_and_render(void)
{
    uint8_t pixels[] = {0, 70, 130, 255};
    uint8_t labels[] = {7};
    char img_path[64];
    char lbl_path[64];

    int err = create_images_file(2051, 1, 2, 2, pixels, sizeof(pixels), img_path, sizeof(img_path));
    ASSERT_EQ(0, err, "%d");
    err = create_labels_file(2049, 1, labels, lbl_path, sizeof(lbl_path));
    ASSERT_EQ(0, err, "%d");

    MnistImages images;
    MnistLabels lbls;

    err = mnist_load_images(img_path, &images);
    ASSERT_EQ(0, err, "%d");
    err = mnist_load_labels(lbl_path, &lbls);
    ASSERT_EQ(0, err, "%d");

    ASSERT_EQ(1u, images.count, "%u");
    ASSERT_EQ(2u, images.rows, "%u");
    ASSERT_EQ(2u, images.cols, "%u");
    ASSERT_EQ(0u, images.pixels[0], "%u");
    ASSERT_EQ(70u, images.pixels[1], "%u");
    ASSERT_EQ(130u, images.pixels[2], "%u");
    ASSERT_EQ(255u, images.pixels[3], "%u");
    ASSERT_EQ(1u, lbls.count, "%u");
    ASSERT_EQ(7u, lbls.labels[0], "%u");

    size_t buf_size = 0;
    err = mnist_ascii_buffer_size(&images, &buf_size);
    ASSERT_EQ(0, err, "%d");

    char *buf = malloc(buf_size);
    ASSERT_TRUE(buf != NULL, "malloc buffer");

    err = mnist_render_ascii(&images, 0, buf, buf_size);
    ASSERT_EQ(0, err, "%d");
    ASSERT_TRUE(strcmp(buf, ".+\n##\n") == 0, "ASCII rendering mismatch");

    free(buf);
    mnist_free_images(&images);
    mnist_free_labels(&lbls);
    remove(img_path);
    remove(lbl_path);
}

static void test_rejects_bad_magic(void)
{
    uint8_t pixels[] = {0, 1, 2, 3};
    char img_path[64];

    int err = create_images_file(9999, 1, 2, 2, pixels, sizeof(pixels), img_path, sizeof(img_path));
    ASSERT_EQ(0, err, "%d");

    MnistImages images;
    err = mnist_load_images(img_path, &images);
    ASSERT_TRUE(err == EINVAL, "expected EINVAL for bad magic");

    remove(img_path);
}

static void run_test(const char *name, void (*fn)(void))
{
    int before = failures;
    fn();
    if (failures == before)
    {
        printf("[PASS] %s\n", name);
    }
    else
    {
        printf("[FAIL] %s\n", name);
    }
    tests_run++;
}

int main(void)
{
    run_test("load_and_render", test_load_and_render);
    run_test("rejects_bad_magic", test_rejects_bad_magic);

    if (failures)
    {
        fprintf(stderr, "%d/%d tests failed\n", failures, tests_run);
        return 1;
    }

    printf("All %d tests passed\n", tests_run);
    return 0;
}
