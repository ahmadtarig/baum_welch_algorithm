public class BaumWelch {
    private final String observationSequence[]; // the observation sequence
    private final String observations[]; // the observation (0)
    private final String states[]; // the states (S)
    private float transitionMatrix[][]; // the transition matrix (P)
    private float emissionMatrix[][]; // the emission matrix (B)
    private float initialProbabilities[]; // the initial probabilities (pi)

    public BaumWelch(String observationSequence[], String states[], String observations[], float transitionMatrix[][],
            float initialProbabilities[], float emissionMatrix[][]) {
        this.observationSequence = observationSequence;
        this.observations = observations;
        this.states = states;
        this.transitionMatrix = transitionMatrix;
        this.initialProbabilities = initialProbabilities;
        this.emissionMatrix = emissionMatrix;
    }

    public BaumWelch(String observationSequence[], String states[], String observations[]) {
        this.observationSequence = observationSequence;
        this.observations = observations;
        this.states = states;

        // Initialize transitionMatrix, emissionMatrix, initialProbabilities
        transitionMatrix = new float[states.length][states.length];
        emissionMatrix = new float[observations.length][states.length];
        initialProbabilities = new float[states.length];

        // Fill transitionMatrix, emissionMatrix, initialProbabilities with random
        // values
        fillMatrix(transitionMatrix, states.length, states.length);
        fillMatrix(emissionMatrix, observations.length, states.length);
        fillVector(initialProbabilities, states.length);
    }

    private void baumWelch() {
        // Your Baum-Welch implementation goes here.
        // You may use the provided transitionMatrix, emissionMatrix, and
        // initialProbabilities.
        // Update transitionMatrix, emissionMatrix, and initialProbabilities as per the
        // Baum-Welch algorithm.
        // Return the updated transitionMatrix, emissionMatrix, and
        // initialProbabilities.
        // You may assume that the provided sequence is valid.
        // You may use any necessary libraries or classes for your implementation.
        // You should also handle any potential errors or exceptions that may occur
        // during the Baum-Welch algorithm.
        // Return the updated transitionMatrix, emissionMatrix, and
        // initialProbabilities, or an error message if an error occurs.

        // print the transitionMatrix and emissionMatrix
        System.out.println("Transition Matrix: ");
        this.displayMetrics(transitionMatrix);

        System.out.println("\nEmission Matrix: ");
        this.displayMetrics(transpose(this.emissionMatrix));

        // create the xi, forward, backward and gamma matrix
        float[][][] xi = new float[this.observationSequence.length - 1][this.states.length][this.states.length];
        float[][] forwardMatrix = new float[this.states.length][this.observationSequence.length];
        float[][] backwardMatrix = new float[this.states.length][this.observationSequence.length];
        float[][] gammaMatrix = new float[this.states.length][this.observationSequence.length];

        // step 1:
        // Calculate the forward and backward probabilities using the forward-backward
        // algorithm.

        // calculate alpa 1 and beta n-1
        for (int state = 0; state < this.states.length; state++) {
            forwardMatrix[state][0] = initialProbabilities[state]
                    * emissionMatrix[this.positionOf(this.observationSequence[0])][state];
            backwardMatrix[state][this.observationSequence.length - 1] = 1;
        }

        // calculate the forward matrix
        for (int t = 1; t < this.observationSequence.length; t++) {
            for (int state = 0; state < this.states.length; state++) {
                float sum = 0;
                for (int prevState = 0; prevState < this.states.length; prevState++) {
                    sum += forwardMatrix[prevState][t - 1] * transitionMatrix[prevState][state];
                }
                forwardMatrix[state][t] = sum * emissionMatrix[this.positionOf(this.observationSequence[t])][state];
            }
        }

        // print forward matrix
        System.out.println("\n\nThe forward transformation matrix : ");
        this.displayMetrics(forwardMatrix);

        // calculate the backward
        for (int t = this.observationSequence.length - 2; t >= 0; t--) {
            for (int state = 0; state < this.states.length; state++) {
                float sum = 0;
                for (int nextState = 0; nextState < this.states.length; nextState++) {
                    sum += transitionMatrix[state][nextState]
                            * emissionMatrix[this.positionOf(this.observationSequence[t + 1])][nextState]
                            * backwardMatrix[nextState][t + 1];
                }
                backwardMatrix[state][t] = sum;
            }
        }

        // print backward matrix
        System.out.println("\n\nThe backward transformation matrix : ");
        this.displayMetrics(backwardMatrix);

        // calculate and display the backword probabilities
        float probabilitForword = 0f;
        for (int state = 0; state < this.states.length; state++)
            probabilitForword += forwardMatrix[state][forwardMatrix[0].length - 1];

        // calculate ad display the backword probabilities
        float probabilitBackword = 0f;
        for (int state = 0; state < this.states.length; state++)
            probabilitBackword += backwardMatrix[state][0];

        // print the probabilities
        System.out.println("\nUsing forward algorithm the probability of sequence "
                + this.victorInString(observationSequence) + " is " + probabilitForword);
        System.out.println("Using backword algorithm the probability of sequence "
                + this.victorInString(observationSequence) + " is " + probabilitBackword + "\n");

        // step 2:
        // calculate the gamma and xi matrix
        float divisor;
        for (int t = 0; t < this.observationSequence.length; t++) {
            for (int state = 0; state < this.states.length; state++) {
                divisor = 0f;
                for (int divisorState = 0; divisorState < this.states.length; divisorState++)
                    divisor += forwardMatrix[divisorState][t] * backwardMatrix[divisorState][t];
                gammaMatrix[state][t] = forwardMatrix[state][t] * backwardMatrix[state][t];
                gammaMatrix[state][t] /= divisor;
            }
        }

        // print gamma matrix
        System.out.println("\nThe gamma transformation matrix : ");
        this.displayMetrics(gammaMatrix);

        // calculate the transformation matrix
        for (int t = 0; t < this.observationSequence.length - 1; t++) {
            float denominator = 0;
            for (int i = 0; i < this.states.length; i++) {
                for (int j = 0; j < this.states.length; j++) {
                    denominator += forwardMatrix[i][t] * transitionMatrix[i][j] *
                            emissionMatrix[this.positionOf(this.observationSequence[t + 1])][j] *
                            backwardMatrix[j][t + 1];
                }
            }

            for (int i = 0; i < this.states.length; i++) {
                for (int j = 0; j < this.states.length; j++) {
                    xi[t][i][j] = (forwardMatrix[i][t] * transitionMatrix[i][j] *
                            emissionMatrix[this.positionOf(this.observationSequence[t + 1])][j] *
                            backwardMatrix[j][t + 1]) / denominator;
                }
            }
        }

        // print xi matrix
        System.out.println("\nJoint probability sequence (xi) for each time step:");
        for (int t = 0; t < this.observationSequence.length - 1; t++) {
            System.out.println("Time " + t + " to " + (t + 1) + ":");
            this.displayMetrics(xi[t]);
            System.out.println();
        }

        // step 3:
        // calculate the new transition matrix
        float[][] newTransitionMatrix = new float[this.states.length][this.states.length];
        for (int i = 0; i < this.states.length; i++) {
            for (int j = 0; j < this.states.length; j++) {
                float sum = 0f;
                float denominator = 0f;
                for (int t = 0; t < this.observationSequence.length - 1; t++) {
                    sum += xi[t][i][j];
                    denominator += gammaMatrix[i][t];
                }
                newTransitionMatrix[i][j] = sum / denominator;
            }
        }

        // print new transition matrix
        System.out.println("\nNew transition matrix : ");
        this.displayMetrics(newTransitionMatrix);

        // step 4:
        // calculate the new emission matrix
        float newEmissionMatrix[][] = new float[observations.length][states.length];
        for (int state = 0; state < this.states.length; state++) {
            for (int observed = 0; observed < this.observations.length; observed++) {
                float sum = 0f;
                float denominator = 0f;
                for (int t = 0; t < this.observationSequence.length; t++) {
                    if (this.observationSequence[t].equals(this.observations[observed]))
                        sum += gammaMatrix[state][t];
                    denominator += gammaMatrix[state][t];
                }
                newEmissionMatrix[observed][state] = sum / denominator;
            }
        }

        // print new emission matrix
        System.out.println("\nNew emission matrix : ");
        this.displayMetrics(transpose(newEmissionMatrix));

        // step 5:
        // calculate the new initial probabilities
        float[] newInitialProbabilities = new float[this.states.length];
        for (int state = 0; state < this.states.length; state++) {
            float denominator = 0.0f;
            for (int innerState = 0; innerState < this.states.length; innerState++) {
                denominator += gammaMatrix[innerState][0];
            }
            newInitialProbabilities[state] = gammaMatrix[state][0] / denominator;
        }

        // print new initial probabilities
        System.out.println("\nNew initial probabilities : [" + this.victorInString(newInitialProbabilities) + "]\n");

        // step 6:
        // update all probabilities
        this.initialProbabilities = newInitialProbabilities;
        this.transitionMatrix = newTransitionMatrix;
        this.emissionMatrix = newEmissionMatrix;
    }

    // fill matrix with the random values
    private void fillMatrix(float matrix[][], int rows, int cols) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i][j] = (float) Math.random();
    }

    // fill victor with the random number
    private void fillVector(float vector[], int size) {
        for (int i = 0; i < size; i++)
            vector[i] = (float) Math.random();
    }

    // update probability take repet number as input
    public void updateProbabilities(int numberOfUpdates) {
        for (int update = 0; update < numberOfUpdates; update++) {
            System.out.println("update number " + update + ":");
            this.baumWelch();
            System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
        }
    }

    // transform float victor to string
    private String victorInString(float[] vector) {
        String s = new String();
        for (float item : vector)
            s += item + " ";
        return s;
    }

    // transform string victor to string
    private String victorInString(String[] vector) {
        String s = new String();
        for (String item : vector)
            s += item + " ";
        return s;
    }

    // print all matrix components
    private void displayMetrics(float matrix[][]) {
        // print all metrics components
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }

    // transpose matrix
    private static float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] transpose = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transpose[j][i] = matrix[i][j];
            }
        }
        return transpose;
    }

    // find position of observation
    private int positionOf(String observed) {
        for (int i = 0; i < this.observations.length; i++)
            if (this.observations[i] == observed)
                return i;
        return -1;
    }

    public static void main(String[] args) {
        BaumWelch bw = new BaumWelch(
                new String[] { "0", "1", "0" },
                new String[] { "A", "B" },
                new String[] { "0", "1" },
                new float[][] { { 0.99f, 0.01f }, { 0.01f, 0.99f } },
                new float[] { 0.99f, 0.01f },
                new float[][] { { 0.8f, 0.1f }, { 0.2f, 0.9f } });

        bw.updateProbabilities(1);
    }

}