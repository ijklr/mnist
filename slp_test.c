/*
 * =======================================================================
 * SINGLE LAYER PERCEPTRON - TEST SUITE
 * =======================================================================
 *
 * This file tests the single layer perceptron implementation with:
 * 1. Sigmoid function correctness
 * 2. Forward pass with known weights
 * 3. Training convergence on OR gate
 * 4. Training convergence on AND gate
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Test configuration */
#define NUM_FEATURES 2
#define EPSILON 1e-12
#define TOL 1e-4f

/* Sigmoid function - copied from main implementation */
static double sigmoid(double z) {
    if (z >= 0) {
        double ez = exp(-z);
        return 1.0 / (1.0 + ez);
    } else {
        double ez = exp(z);
        return ez / (1.0 + ez);
    }
}

/* Forward pass - copied from main implementation */
static double forward(const double x[NUM_FEATURES],
                      const double w[NUM_FEATURES],
                      double b, int D) {
    double z = b;
    for (int j = 0; j < D; j++) {
        z += w[j] * x[j];
    }
    return sigmoid(z);
}

/* Prediction - copied from main implementation */
static int predict(const double x[NUM_FEATURES],
                   const double w[NUM_FEATURES],
                   double b, int D) {
    return forward(x, w, b, D) >= 0.5;
}

/* Helper function for float comparison */
static void assert_close(double got, double expect, double tol) {
    assert(fabs(got - expect) <= tol);
}

/* Test 1: Sigmoid function behavior */
static void test_sigmoid_function(void) {
    /* Test sigmoid(0) = 0.5 */
    assert_close(sigmoid(0.0), 0.5, TOL);

    /* Test sigmoid of large positive value approaches 1 */
    assert(sigmoid(10.0) > 0.99);

    /* Test sigmoid of large negative value approaches 0 */
    assert(sigmoid(-10.0) < 0.01);

    /* Test sigmoid is monotonically increasing */
    assert(sigmoid(-1.0) < sigmoid(0.0));
    assert(sigmoid(0.0) < sigmoid(1.0));
}

/* Test 2: Forward pass with known weights */
static void test_forward_known_weights(void) {
    const int D = NUM_FEATURES;
    double w[NUM_FEATURES] = {1.0, 2.0};
    double b = -1.5;

    /* Test case 1: input [0, 0] */
    double x1[NUM_FEATURES] = {0.0, 0.0};
    double z1 = -1.5;  /* b only */
    double expected1 = sigmoid(z1);
    assert_close(forward(x1, w, b, D), expected1, TOL);

    /* Test case 2: input [1, 0] */
    double x2[NUM_FEATURES] = {1.0, 0.0};
    double z2 = 1.0 * 1.0 + 0.0 * 2.0 - 1.5;  /* -0.5 */
    double expected2 = sigmoid(z2);
    assert_close(forward(x2, w, b, D), expected2, TOL);

    /* Test case 3: input [0, 1] */
    double x3[NUM_FEATURES] = {0.0, 1.0};
    double z3 = 0.0 * 1.0 + 1.0 * 2.0 - 1.5;  /* 0.5 */
    double expected3 = sigmoid(z3);
    assert_close(forward(x3, w, b, D), expected3, TOL);

    /* Test case 4: input [1, 1] */
    double x4[NUM_FEATURES] = {1.0, 1.0};
    double z4 = 1.0 * 1.0 + 1.0 * 2.0 - 1.5;  /* 1.5 */
    double expected4 = sigmoid(z4);
    assert_close(forward(x4, w, b, D), expected4, TOL);
}

/* Test 3: Prediction thresholding */
static void test_predict_threshold(void) {
    const int D = NUM_FEATURES;
    double w[NUM_FEATURES] = {1.0, 1.0};

    /* When b = 0, sigmoid(0) = 0.5, should predict 1 */
    double x1[NUM_FEATURES] = {0.0, 0.0};
    assert(predict(x1, w, 0.0, D) == 1);

    /* Large positive weighted sum should predict 1 */
    double x2[NUM_FEATURES] = {10.0, 10.0};
    assert(predict(x2, w, 0.0, D) == 1);

    /* Large negative weighted sum should predict 0 */
    double b_neg = -100.0;
    assert(predict(x1, w, b_neg, D) == 0);
}

/* Test 4: Training on OR gate */
static void test_train_or_gate(void) {
    const int N = 4;
    const int D = NUM_FEATURES;
    const double learning_rate = 0.1;
    const int epochs = 2000;

    /* OR gate training data */
    double X[4][NUM_FEATURES] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double y[4] = {0, 1, 1, 1};

    /* Initialize weights to zero */
    double w[NUM_FEATURES] = {0.0, 0.0};
    double b = 0.0;

    /* Training loop */
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < N; i++) {
            double y_hat = forward(X[i], w, b, D);
            double dz = (y_hat - y[i]);

            for (int j = 0; j < D; j++) {
                w[j] -= learning_rate * dz * X[i][j];
            }
            b -= learning_rate * dz;
        }
    }

    /* After training, all predictions should be correct */
    int all_correct = 1;
    for (int i = 0; i < N; i++) {
        int pred = predict(X[i], w, b, D);
        if (pred != (int)y[i]) {
            all_correct = 0;
            break;
        }
    }

    assert(all_correct);
}

/* Test 5: Training on AND gate */
static void test_train_and_gate(void) {
    const int N = 4;
    const int D = NUM_FEATURES;
    const double learning_rate = 0.1;
    const int epochs = 2000;

    /* AND gate training data */
    double X[4][NUM_FEATURES] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double y[4] = {0, 0, 0, 1};

    /* Initialize weights to zero */
    double w[NUM_FEATURES] = {0.0, 0.0};
    double b = 0.0;

    /* Training loop */
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < N; i++) {
            double y_hat = forward(X[i], w, b, D);
            double dz = (y_hat - y[i]);

            for (int j = 0; j < D; j++) {
                w[j] -= learning_rate * dz * X[i][j];
            }
            b -= learning_rate * dz;
        }
    }

    /* After training, all predictions should be correct */
    int all_correct = 1;
    for (int i = 0; i < N; i++) {
        int pred = predict(X[i], w, b, D);
        if (pred != (int)y[i]) {
            all_correct = 0;
            break;
        }
    }

    assert(all_correct);
}

/* Test 6: Loss decreases during training */
static void test_loss_decreases(void) {
    const int N = 4;
    const int D = NUM_FEATURES;
    const double learning_rate = 0.1;

    /* OR gate training data */
    double X[4][NUM_FEATURES] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double y[4] = {0, 1, 1, 1};

    /* Initialize weights to zero */
    double w[NUM_FEATURES] = {0.0, 0.0};
    double b = 0.0;

    /* Compute initial loss */
    double initial_loss = 0.0;
    for (int i = 0; i < N; i++) {
        double y_hat = forward(X[i], w, b, D);
        initial_loss += -(y[i] * log(y_hat + EPSILON) +
                          (1.0 - y[i]) * log(1.0 - y_hat + EPSILON));
    }
    initial_loss /= N;

    /* Train for 500 epochs */
    for (int e = 0; e < 500; e++) {
        for (int i = 0; i < N; i++) {
            double y_hat = forward(X[i], w, b, D);
            double dz = (y_hat - y[i]);

            for (int j = 0; j < D; j++) {
                w[j] -= learning_rate * dz * X[i][j];
            }
            b -= learning_rate * dz;
        }
    }

    /* Compute final loss */
    double final_loss = 0.0;
    for (int i = 0; i < N; i++) {
        double y_hat = forward(X[i], w, b, D);
        final_loss += -(y[i] * log(y_hat + EPSILON) +
                        (1.0 - y[i]) * log(1.0 - y_hat + EPSILON));
    }
    final_loss /= N;

    /* Loss should decrease significantly */
    assert(final_loss < initial_loss);
    assert(final_loss < 0.1);  /* Should converge to very low loss */
}

int main(void) {
    /* Test runner */
    struct {
        const char *name;
        void (*fn)(void);
    } tests[] = {
        {"sigmoid_function", test_sigmoid_function},
        {"forward_known_weights", test_forward_known_weights},
        {"predict_threshold", test_predict_threshold},
        {"train_or_gate", test_train_or_gate},
        {"train_and_gate", test_train_and_gate},
        {"loss_decreases", test_loss_decreases},
    };

    size_t count = sizeof(tests) / sizeof(tests[0]);
    for (size_t i = 0; i < count; i++) {
        printf("[TEST] %s ... ", tests[i].name);
        fflush(stdout);
        tests[i].fn();
        puts("PASS");
    }

    printf("\nAll %zu tests passed!\n", count);
    return 0;
}
