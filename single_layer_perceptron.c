#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double sigmoid(double z) {
    // numerically stable-ish sigmoid
    if (z >= 0) {
        double ez = exp(-z);
        return 1.0 / (1.0 + ez);
    } else {
        double ez = exp(z);
        return ez / (1.0 + ez);
    }
}

int main(void) {
    // Example: OR gate dataset (linearly separable)
    // x has 2 features, y in {0,1}
    const int N = 4;
    const int D = 2;
    double X[N][D] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double y[N] = {0, 1, 1, 1};

    // Parameters
    double w[D];
    double b = 0.0;

    // init weights small
    for (int j = 0; j < D; j++) w[j] = 0.0;

    double lr = 0.1;
    int epochs = 2000;

    for (int e = 0; e < epochs; e++) {
        double loss = 0.0;

        for (int i = 0; i < N; i++) {
            // forward: z = w·x + b
            double z = b;
            for (int j = 0; j < D; j++) z += w[j] * X[i][j];

            double yhat = sigmoid(z);

            // cross-entropy loss (avoid log(0))
            double eps = 1e-12;
            loss += -(y[i] * log(yhat + eps) + (1.0 - y[i]) * log(1.0 - yhat + eps));

            // gradient for logistic regression:
            // dL/dz = yhat - y
            double dz = (yhat - y[i]);

            // update
            for (int j = 0; j < D; j++) w[j] -= lr * dz * X[i][j];
            b -= lr * dz;
        }

        if ((e % 200) == 0) {
            printf("epoch %d loss %.6f w=(%.3f, %.3f) b=%.3f\n", e, loss / N, w[0], w[1], b);
        }
    }

    // test
    for (int i = 0; i < N; i++) {
        double z = b;
        for (int j = 0; j < D; j++) z += w[j] * X[i][j];
        double yhat = sigmoid(z);
        printf("x=(%.0f,%.0f) -> yhat=%.3f pred=%d\n", X[i][0], X[i][1], yhat, (yhat >= 0.5));
    }

    return 0;
}
