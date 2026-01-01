#include "mlp.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple float comparison for small expected values. */
static void assert_close(float got, float expect, float tol)
{
    assert(fabsf(got - expect) <= tol);
}

static void seed_and_clear(Mlp *net, size_t in, size_t hidden, size_t out)
{
    srand(0);
    int ok = mlp_init(net, in, hidden, out);
    assert(ok == 0);
}

static void test_forward_known_values(void)
{
    /* Checks deterministic forward activations/logits and argmax class. */
    Mlp net;
    seed_and_clear(&net, 2, 2, 2);

    /* Fixed parameters to make the forward pass deterministic. */
    net.w1[0] = 1.0f;
    net.w1[1] = 2.0f;
    net.w1[2] = -1.0f;
    net.w1[3] = 0.5f;
    net.b1[0] = 0.1f;
    net.b1[1] = -0.2f;

    net.w2[0] = 0.3f;
    net.w2[1] = -0.7f;
    net.w2[2] = 0.8f;
    net.w2[3] = 0.2f;
    net.b2[0] = 0.05f;
    net.b2[1] = -0.05f;

    float input[2] = {1.0f, 1.0f};
    float hidden[2] = {0};
    float logits[2] = {0};
    mlp_forward(&net, input, hidden, logits);

    assert_close(hidden[0], 3.1f, 1e-4f);
    assert_close(hidden[1], 0.0f, 1e-6f);
    assert_close(logits[0], 0.98f, 1e-4f);
    assert_close(logits[1], 2.43f, 1e-4f);
    assert(mlp_predict(logits, 2) == 1);

    mlp_free(&net);
}

static void test_backward_updates_bias_toward_label(void)
{
    /* Ensures backward pass nudges the correct-class bias when hidden=0. */
    Mlp net;
    seed_and_clear(&net, 2, 2, 2);

    /* Start from zero params to make the gradient math trivial. */
    memset(net.w1, 0, sizeof(float) * net.hidden_size * net.input_size);
    memset(net.b1, 0, sizeof(float) * net.hidden_size);
    memset(net.w2, 0, sizeof(float) * net.output_size * net.hidden_size);
    memset(net.b2, 0, sizeof(float) * net.output_size);

    float input[2] = {1.0f, 0.0f};
    float hidden[2] = {0};
    float logits[2] = {0};
    mlp_forward(&net, input, hidden, logits);

    float lr = 0.1f;
    float loss = mlp_backward(&net, input, hidden, logits, 0, lr);
    assert(loss > 0.0f);

    /* With zero hidden activations, only the output biases move. */
    assert_close(net.b2[0], 0.05f, 1e-6f);
    assert_close(net.b2[1], -0.05f, 1e-6f);
    assert_close(net.w1[0], 0.0f, 1e-6f);
    assert_close(net.w2[0], 0.0f, 1e-6f);

    mlp_free(&net);
}

static float evaluate_accuracy(const Mlp *net, const float *images, const uint8_t *labels,
                               size_t count)
{
    float *hidden = malloc(sizeof(float) * net->hidden_size);
    float *logits = malloc(sizeof(float) * net->output_size);
    assert(hidden && logits);

    size_t correct = 0;
    size_t input_size = net->input_size;
    for (size_t i = 0; i < count; i++)
    {
        mlp_forward(net, images + i * input_size, hidden, logits);
        correct += mlp_predict(logits, net->output_size) == labels[i];
    }

    free(hidden);
    free(logits);
    return (float)correct / (float)count;
}

static void test_train_improves_on_tiny_dataset(void)
{
    /* Verifies end-to-end training improves accuracy on a toy dataset. */
    Mlp net;
    seed_and_clear(&net, 2, 2, 2);

    /* Deterministic, slightly asymmetric starting weights. */
    net.w1[0] = 0.4f;
    net.w1[1] = -0.2f;
    net.w1[2] = -0.3f;
    net.w1[3] = 0.1f;
    net.w2[0] = 0.2f;
    net.w2[1] = 0.05f;
    net.w2[2] = -0.15f;
    net.w2[3] = 0.25f;
    memset(net.b1, 0, sizeof(float) * net.hidden_size);
    memset(net.b2, 0, sizeof(float) * net.output_size);

    /* Two samples, linearly separable. */
    float images[4] = {
        1.0f, 0.0f, /* label 0 */
        0.0f, 1.0f  /* label 1 */
    };
    uint8_t labels[2] = {0, 1};

    float acc_before = evaluate_accuracy(&net, images, labels, 2);

    TrainConfig cfg = {
        .epochs = 5,
        .batch_size = 1,
        .learning_rate = 0.2f};
    int rc = mlp_train(&net, images, labels, 2, &cfg);
    assert(rc == 0);

    float acc_after = evaluate_accuracy(&net, images, labels, 2);
    assert(acc_after >= acc_before);
    assert(acc_after > 0.5f); /* Expect at least 1/2 samples correct after training. */

    mlp_free(&net);
}

int main(void)
{
    /* Lightweight runner that reports each test name and status. */
    struct
    {
        const char *name;
        void (*fn)(void);
    } tests[] = {
        {"forward_known_values", test_forward_known_values},
        {"backward_bias_update", test_backward_updates_bias_toward_label},
        {"train_improves_tiny_dataset", test_train_improves_on_tiny_dataset},
    };

    size_t count = sizeof(tests) / sizeof(tests[0]);
    for (size_t i = 0; i < count; i++)
    {
        printf("[TEST] %s ... ", tests[i].name);
        fflush(stdout);
        tests[i].fn();
        puts("PASS");
    }

    return 0;
}
