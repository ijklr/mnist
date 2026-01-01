#include "mlp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Simple RNG helper: random float in [-scale, scale]. */
static float rand_uniform(float scale)
{
    return scale * (2.0f * rand() / (float)RAND_MAX - 1.0f);
}

int mlp_init(Mlp *net, size_t input_size, size_t hidden_size, size_t output_size)
{
    memset(net, 0, sizeof(*net));
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;

    size_t w1_count = hidden_size * input_size;
    size_t w2_count = output_size * hidden_size;

    net->w1 = calloc(w1_count, sizeof(float));
    net->b1 = calloc(hidden_size, sizeof(float));
    net->w2 = calloc(w2_count, sizeof(float));
    net->b2 = calloc(output_size, sizeof(float));
    if (!net->w1 || !net->b1 || !net->w2 || !net->b2)
    {
        mlp_free(net);
        return -1;
    }

    /* Xavier-style uniform init */
    float scale1 = 1.0f / (float)sqrt((double)input_size);
    float scale2 = 1.0f / (float)sqrt((double)hidden_size);
    for (size_t i = 0; i < w1_count; i++)
    {
        net->w1[i] = rand_uniform(scale1);
    }
    for (size_t i = 0; i < w2_count; i++)
    {
        net->w2[i] = rand_uniform(scale2);
    }

    return 0;
}

void mlp_free(Mlp *net)
{
    free(net->w1);
    free(net->b1);
    free(net->w2);
    free(net->b2);
    memset(net, 0, sizeof(*net));
}

void mlp_forward(const Mlp *net, const float *input, float *hidden_out, float *logits_out)
{
    /* hidden = ReLU(W1*x + b1) */
    for (size_t h = 0; h < net->hidden_size; h++)
    {
        float sum = net->b1[h];
        const float *w_row = net->w1 + h * net->input_size;
        for (size_t i = 0; i < net->input_size; i++)
        {
            sum += w_row[i] * input[i];
        }
        float activated = sum > 0.0f ? sum : 0.0f;
        hidden_out[h] = activated;
    }

    /* logits = W2*hidden + b2 */
    for (size_t o = 0; o < net->output_size; o++)
    {
        float sum = net->b2[o];
        const float *w_row = net->w2 + o * net->hidden_size;
        for (size_t h = 0; h < net->hidden_size; h++)
        {
            sum += w_row[h] * hidden_out[h];
        }
        logits_out[o] = sum;
    }
}

static float softmax_normalize(float *logits, size_t n)
{
    /* Subtract max for stability */
    float max_logit = logits[0];
    for (size_t i = 1; i < n; i++)
    {
        if (logits[i] > max_logit)
        {
            max_logit = logits[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
    {
        logits[i] = expf(logits[i] - max_logit);
        sum += logits[i];
    }

    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++)
    {
        logits[i] *= inv;
    }
    return max_logit + logf(sum);
}

float mlp_backward(Mlp *net, const float *input, const float *hidden, const float *logits,
                   uint8_t label, float learning_rate)
{
    /* Copy logits to mutable buffer for softmax; small on stack for 10 classes. */
    float probs_buf[32];
    size_t n = net->output_size;
    for (size_t i = 0; i < n; i++)
    {
        probs_buf[i] = logits[i];
    }
    softmax_normalize(probs_buf, n);

    /* Cross-entropy loss with one-hot target */
    float loss = -logf(probs_buf[label] + 1e-9f);

    /* grad_logits = probs - one_hot(label) */
    probs_buf[label] -= 1.0f;

    /* Gradients for W2 and b2 */
    for (size_t o = 0; o < net->output_size; o++)
    {
        float g = probs_buf[o]; /* dL/dlogit */
        net->b2[o] -= learning_rate * g;
        float *w_row = net->w2 + o * net->hidden_size;
        for (size_t h = 0; h < net->hidden_size; h++)
        {
            w_row[h] -= learning_rate * g * hidden[h];
        }
    }

    /* Backprop into hidden: g_hidden = ReLU'(hidden) * (W2^T * grad_logits) */
    float g_hidden[1024]; /* fits typical small hidden sizes; adjust if you grow the net */
    for (size_t h = 0; h < net->hidden_size; h++)
    {
        float sum = 0.0f;
        for (size_t o = 0; o < net->output_size; o++)
        {
            sum += net->w2[o * net->hidden_size + h] * probs_buf[o];
        }
        g_hidden[h] = hidden[h] > 0.0f ? sum : 0.0f;
    }

    /* Gradients for W1 and b1 */
    for (size_t h = 0; h < net->hidden_size; h++)
    {
        float g = g_hidden[h];
        net->b1[h] -= learning_rate * g;
        float *w_row = net->w1 + h * net->input_size;
        for (size_t i = 0; i < net->input_size; i++)
        {
            w_row[i] -= learning_rate * g * input[i];
        }
    }

    return loss;
}

uint8_t mlp_predict(const float *logits, size_t output_size)
{
    size_t best = 0;
    for (size_t i = 1; i < output_size; i++)
    {
        if (logits[i] > logits[best])
        {
            best = i;
        }
    }
    return (uint8_t)best;
}

int mlp_train(Mlp *net, const float *images, const uint8_t *labels, size_t count,
              const TrainConfig *cfg)
{
    /* This simple loop ignores batch_size; add batching/shuffling as needed. */
    (void)cfg->batch_size;
    size_t input_size = net->input_size;
    float *hidden = malloc(sizeof(float) * net->hidden_size);
    float *logits = malloc(sizeof(float) * net->output_size);
    if (!hidden || !logits)
    {
        free(hidden);
        free(logits);
        return -1;
    }

    for (size_t epoch = 0; epoch < cfg->epochs; epoch++)
    {
        float total_loss = 0.0f;
        size_t correct = 0;

        for (size_t idx = 0; idx < count; idx++)
        {
            const float *x = images + idx * input_size;
            uint8_t y = labels[idx];

            mlp_forward(net, x, hidden, logits);
            total_loss += mlp_backward(net, x, hidden, logits, y, cfg->learning_rate);

            /* For quick feedback, treat logits as scores pre-softmax */
            if (mlp_predict(logits, net->output_size) == y)
            {
                correct++;
            }
        }

        float avg_loss = total_loss / (float)count;
        float acc = (float)correct / (float)count;
        /* Hook for logging; replace with your own logger/printf as desired */
        (void)avg_loss;
        (void)acc;
    }

    free(hidden);
    free(logits);
    return 0;
}
