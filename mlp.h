#ifndef MLP_H
#define MLP_H

#include <stddef.h>
#include <stdint.h>

/* Minimal two-layer MLP tailored for MNIST-sized inputs (28x28 -> hidden -> 10). */
typedef struct
{
    size_t input_size;
    size_t hidden_size;
    size_t output_size;

    /* Parameters stored as flat arrays in row-major order */
    float *w1; /* hidden_size x input_size */
    float *b1; /* hidden_size */
    float *w2; /* output_size x hidden_size */
    float *b2; /* output_size */
} Mlp;

typedef struct
{
    size_t epochs;
    size_t batch_size;
    float learning_rate;
} TrainConfig;

/* Lifecycle */
int mlp_init(Mlp *net, size_t input_size, size_t hidden_size, size_t output_size);
void mlp_free(Mlp *net);

/* Forward pass: computes logits; optional scratch buffer for hidden activations. */
void mlp_forward(const Mlp *net, const float *input, float *hidden_out, float *logits_out);

/* Loss + backward update using in-place SGD; label is 0-9. */
float mlp_backward(Mlp *net, const float *input, const float *hidden, const float *logits,
                   uint8_t label, float learning_rate);

/* Inference utility */
uint8_t mlp_predict(const float *logits, size_t output_size);

/* Training loop over the full dataset (images normalized to [0,1]). */
int mlp_train(Mlp *net, const float *images, const uint8_t *labels, size_t count,
              const TrainConfig *cfg);

#endif /* MLP_H */
