/*
 * ============================================================================
 * SINGLE LAYER PERCEPTRON - A Beginner's Guide to Neural Networks
 * ============================================================================
 *
 * This program implements the simplest form of a neural network:
 * a single neuron that learns to classify data into two categories.
 *
 * WHAT IS A PERCEPTRON?
 * A perceptron is a single neuron that:
 * 1. Takes inputs (features like x1, x2)
 * 2. Multiplies each by a learned weight (w1, w2)
 * 3. Adds them together with a bias term (b)
 * 4. Applies an activation function (sigmoid) to get output between 0 and 1
 *
 * MATHEMATICAL MODEL:
 *   z = w1*x1 + w2*x2 + b     (weighted sum)
 *   y_hat = sigmoid(z)         (prediction between 0 and 1)
 *
 * LEARNING PROCESS:
 * The network "learns" by adjusting weights to minimize prediction errors:
 * 1. Make a prediction
 * 2. Calculate how wrong it was (loss)
 * 3. Adjust weights in the direction that reduces the error (gradient descent)
 * 4. Repeat many times (epochs)
 *
 * THIS EXAMPLE:
 * We train the perceptron to learn the OR logic gate:
 *   Input (0,0) -> Output 0
 *   Input (0,1) -> Output 1
 *   Input (1,0) -> Output 1
 *   Input (1,1) -> Output 1
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// --- Dataset Configuration ---
#define NUM_SAMPLES 4      // How many training examples we have
#define NUM_FEATURES 2     // How many input features each example has

// --- Training Hyperparameters ---
// These control how the learning process works
#define LEARNING_RATE 0.1  // How big a step to take when adjusting weights
                           // Too large: might overshoot, too small: learns slowly
#define EPOCHS 2000        // How many times to go through the entire dataset
#define EPSILON 1e-12      // Small value to prevent log(0) in loss calculation
#define PRINT_INTERVAL 200 // Print progress every N epochs

// --- Function Prototypes ---
static double sigmoid(double z);
static double forward(const double x[NUM_FEATURES], const double w[NUM_FEATURES], double b, int D);
static int predict(const double x[NUM_FEATURES], const double w[NUM_FEATURES], double b, int D);

// ============================================================================
// ACTIVATION FUNCTION
// ============================================================================

/*
 * sigmoid - The Sigmoid Activation Function
 *
 * PURPOSE:
 *   Converts any input value to a probability between 0 and 1
 *   This is crucial for binary classification (yes/no, 0/1 decisions)
 *
 * FORMULA:
 *   sigmoid(z) = 1 / (1 + e^(-z))
 *
 * BEHAVIOR:
 *   - Large positive z -> output approaches 1
 *   - Large negative z -> output approaches 0
 *   - z = 0 -> output = 0.5
 *
 * IMPLEMENTATION NOTE:
 *   We use a numerically stable version that avoids overflow for large |z|
 *   Instead of always computing e^(-z), we choose the safer computation
 */
static double sigmoid(double z) {
    if (z >= 0) {
        // For positive z, compute as: 1 / (1 + e^(-z))
        double ez = exp(-z);
        return 1.0 / (1.0 + ez);
    } else {
        // For negative z, compute as: e^z / (1 + e^z) to avoid overflow
        double ez = exp(z);
        return ez / (1.0 + ez);
    }
}

// ============================================================================
// NEURAL NETWORK OPERATIONS
// ============================================================================

/*
 * forward - Forward Propagation (Making a Prediction)
 *
 * PURPOSE:
 *   Computes the network's prediction for a given input
 *
 * PROCESS:
 *   1. Compute weighted sum: z = b + w[0]*x[0] + w[1]*x[1] + ... + w[D-1]*x[D-1]
 *      - This is the "dot product" of weights and inputs, plus bias
 *   2. Apply sigmoid activation: y_hat = sigmoid(z)
 *
 * PARAMETERS:
 *   x - Input features (e.g., [0, 1] for one training example)
 *   w - Learned weights (one per input feature)
 *   b - Learned bias (shifts the decision boundary)
 *   D - Number of features (dimensions)
 *
 * RETURNS:
 *   Predicted probability between 0 and 1
 */
static double forward(const double x[NUM_FEATURES], const double w[NUM_FEATURES], double b, int D) {
    // Step 1: Compute weighted sum (linear combination)
    double z = b;  // Start with the bias term
    for (int j = 0; j < D; j++) {
        z += w[j] * x[j];  // Add each weighted input
    }

    // Step 2: Apply activation function to get probability
    return sigmoid(z);
}

/*
 * predict - Make a Binary Decision
 *
 * PURPOSE:
 *   Converts probability output to a binary class (0 or 1)
 *
 * DECISION RULE:
 *   If probability >= 0.5, predict class 1
 *   Otherwise, predict class 0
 */
static int predict(const double x[NUM_FEATURES], const double w[NUM_FEATURES], double b, int D) {
    return forward(x, w, b, D) >= 0.5;
}

// ============================================================================
// MAIN PROGRAM - TRAINING AND TESTING
// ============================================================================

int main(void) {
    // -------------------------------------------------------------------------
    // STEP 1: PREPARE THE DATASET
    // -------------------------------------------------------------------------
    // We're using the OR logic gate as our training data
    // This is a simple, linearly separable problem perfect for learning
    const int N = NUM_SAMPLES;    // Number of training examples
    const int D = NUM_FEATURES;   // Number of input features

    // Training data:
    // X[i] = input features for example i
    // y[i] = correct output for example i
    double X[N][D] = {
        {0, 0},  // Input: both false -> Output: false (0)
        {0, 1},  // Input: first false, second true -> Output: true (1)
        {1, 0},  // Input: first true, second false -> Output: true (1)
        {1, 1}   // Input: both true -> Output: true (1)
    };
    double y[N] = {0, 1, 1, 1};

    // -------------------------------------------------------------------------
    // STEP 2: INITIALIZE MODEL PARAMETERS
    // -------------------------------------------------------------------------
    // These are what the network will learn during training
    double w[D];  // Weights - one for each input feature
    double b = 0.0;  // Bias - shifts the decision boundary

    // Start with all weights at zero (could also use random initialization)
    for (int j = 0; j < D; j++) {
        w[j] = 0.0;
    }

    printf("Starting training with:\n");
    printf("  Learning rate: %.2f\n", LEARNING_RATE);
    printf("  Epochs: %d\n", EPOCHS);
    printf("  Dataset: OR gate (%d samples, %d features)\n\n", N, D);

    // -------------------------------------------------------------------------
    // STEP 3: TRAINING LOOP (Learning Phase)
    // -------------------------------------------------------------------------
    /*
     * In each epoch:
     * 1. Make predictions for all training examples
     * 2. Calculate the loss (how wrong we are)
     * 3. Compute gradients (direction to improve)
     * 4. Update weights using gradient descent
     */
    for (int e = 0; e < EPOCHS; e++) {
        double loss = 0.0;  // Tracks total error for this epoch

        // Process each training example
        for (int i = 0; i < N; i++) {
            // FORWARD PASS: Make a prediction
            double y_hat = forward(X[i], w, b, D);

            // COMPUTE LOSS: Binary Cross-Entropy
            // Measures how different our prediction is from the true label
            // Formula: -[y*log(y_hat) + (1-y)*log(1-y_hat)]
            // - When y=1, we want y_hat close to 1 (log(y_hat) → 0)
            // - When y=0, we want y_hat close to 0 (log(1-y_hat) → 0)
            loss += -(y[i] * log(y_hat + EPSILON) + (1.0 - y[i]) * log(1.0 - y_hat + EPSILON));

            // COMPUTE GRADIENT: How much to change weights
            // For binary cross-entropy with sigmoid, the gradient is simply: y_hat - y
            // This is mathematically elegant and computationally efficient!
            double dz = (y_hat - y[i]);

            // BACKWARD PASS: Update parameters using gradient descent
            // New weight = Old weight - learning_rate * gradient
            // We move weights in the opposite direction of the gradient (hence the minus sign)
            for (int j = 0; j < D; j++) {
                w[j] -= LEARNING_RATE * dz * X[i][j];  // Update weight j
            }
            b -= LEARNING_RATE * dz;  // Update bias
        }

        // Print progress periodically to see if we're learning
        if ((e % PRINT_INTERVAL) == 0) {
            printf("Epoch %4d | Loss: %.6f | w=(%.3f, %.3f) | b=%.3f\n",
                   e, loss / N, w[0], w[1], b);
        }
    }

    // -------------------------------------------------------------------------
    // STEP 4: EVALUATION (Testing Phase)
    // -------------------------------------------------------------------------
    // Now that training is complete, let's see how well the network learned
    printf("\n");
    printf("============================================================\n");
    printf("TRAINING COMPLETE! Final parameters:\n");
    printf("  w1 = %.4f, w2 = %.4f, b = %.4f\n", w[0], w[1], b);
    printf("============================================================\n\n");

    printf("Testing the learned OR gate:\n");
    printf("-----------------------------------------------------------\n");
    printf("  Input (x1, x2)  |  Probability  |  Prediction  |  Actual\n");
    printf("-----------------------------------------------------------\n");

    for (int i = 0; i < N; i++) {
        double y_hat = forward(X[i], w, b, D);  // Get probability
        int pred = predict(X[i], w, b, D);      // Get binary prediction

        // Check if prediction matches the actual label
        const char* result = (pred == (int)y[i]) ? "✓" : "✗";

        printf("      (%.0f, %.0f)      |     %.4f     |      %d       |    %d   %s\n",
               X[i][0], X[i][1], y_hat, pred, (int)y[i], result);
    }
    printf("-----------------------------------------------------------\n");

    printf("\nHow the network makes decisions:\n");
    printf("  Decision boundary: %.3f*x1 + %.3f*x2 + %.3f = 0\n", w[0], w[1], b);
    printf("  If the weighted sum > 0 and sigmoid > 0.5, predict 1\n");
    printf("  Otherwise, predict 0\n");

    return 0;
}

/*
 * ============================================================================
 * LEARNING EXERCISES
 * ============================================================================
 *
 * To deepen your understanding, try these modifications:
 *
 * 1. CHANGE THE LOGIC GATE:
 *    - Try AND gate: y = {0, 0, 0, 1}
 *    - Try NOR gate: y = {1, 0, 0, 0}
 *    - Try XOR gate: y = {0, 1, 1, 0}  <- This won't work! Why?
 *      (Hint: XOR is not linearly separable, needs multiple layers)
 *
 * 2. EXPERIMENT WITH LEARNING RATE:
 *    - Set LEARNING_RATE to 0.01 (very small) - What happens to training speed?
 *    - Set LEARNING_RATE to 1.0 (large) - Does it still converge?
 *    - Find the optimal learning rate for fastest convergence
 *
 * 3. VISUALIZE THE DECISION BOUNDARY:
 *    - The line w1*x1 + w2*x2 + b = 0 separates the two classes
 *    - Points above the line: predicted as 1
 *    - Points below the line: predicted as 0
 *
 * 4. ADD RANDOM INITIALIZATION:
 *    - Instead of w[j] = 0.0, try w[j] = (rand() / (double)RAND_MAX) - 0.5
 *    - This gives small random weights between -0.5 and 0.5
 *    - Does it change the final result?
 *
 * 5. IMPLEMENT BATCH GRADIENT DESCENT:
 *    - Currently we update weights after each example (stochastic)
 *    - Try accumulating gradients for all examples, then updating once per epoch
 *    - Compare convergence speed
 *
 * 6. ADD MORE FEATURES:
 *    - Try a 3-input logic gate with NUM_FEATURES = 3
 *    - You'll need 8 training examples (2^3 possible inputs)
 */
