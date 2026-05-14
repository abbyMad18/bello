
import java.util.*;
import java.io.*;

public class Network {

    Neuron first = new Neuron();
    Neuron second = new Neuron();

    Neuron third = new Neuron();
    Neuron fourth = new Neuron();

    Neuron out = new Neuron();

    /*
     * Data flows:
     * inputs → hidden layer 1 → output layer
     */
    public double predict(double x1, double x2) {

        // Pass inputs into hidden neurons
        double firstOut = first.predict(x1, x2);
        double secondOut = second.predict(x1, x2);

        // Second hidden layer
        double thirdOut = third.predict(firstOut, secondOut);
        double fourthOut = fourth.predict(firstOut, secondOut);

        // Hidden outputs become inputs to final neuron
        return out.predict(thirdOut, fourthOut);
    }

    /**
     * TRAINING FUNCTION
     */
    public void train(ArrayList<double[]> data, ArrayList<Double> answers) {

        double lr = 0.1;

        for (int epoch = 0; epoch < 10000; epoch++) {

            double totalLoss = 0;

            for (int i = 0; i < data.size(); i++) {

                double x1 = data.get(i)[0];
                double x2 = data.get(i)[1];
                double correctAnswer = answers.get(i);

                // =========================
                // FORWARD PASS
                // =========================

                // First hidden layer
                double z1 = first.rawValue(x1, x2);
                double firstOut = first.predict(x1, x2);

                double z2 = second.rawValue(x1, x2);
                double secondOut = second.predict(x1, x2);

                // Second hidden layer
                double z3 = third.rawValue(firstOut, secondOut);
                double thirdOut = third.predict(firstOut, secondOut);

                double z4 = fourth.rawValue(firstOut, secondOut);
                double fourthOut = fourth.predict(firstOut, secondOut);

                // Output layer
                double z5 = out.rawValue(thirdOut, fourthOut);
                double pred = out.predict(thirdOut, fourthOut);

                // =========================
                // LOSS
                // =========================

                double error = pred - correctAnswer;
                totalLoss += error * error;

                // =========================
                // BACKPROPAGATION
                // =========================

                double predictionLoss = 2 * error;
                double outputDerivative = Neuron.sigmoidDerivative(z5);

                // Output neuron gradients
                double grad_out_w1 = predictionLoss * outputDerivative * thirdOut;
                double grad_out_w2 = predictionLoss * outputDerivative * fourthOut;
                double grad_out_b = predictionLoss * outputDerivative;

                // Second hidden layer gradients
                double dz3 = Neuron.sigmoidDerivative(z3);
                double dz4 = Neuron.sigmoidDerivative(z4);

                double grad_third_w1 = predictionLoss * outputDerivative * out.w1 * dz3 * firstOut;
                double grad_third_w2 = predictionLoss * outputDerivative * out.w1 * dz3 * secondOut;
                double grad_third_b = predictionLoss * outputDerivative * out.w1 * dz3;

                double grad_fourth_w1 = predictionLoss * outputDerivative * out.w2 * dz4 * firstOut;
                double grad_fourth_w2 = predictionLoss * outputDerivative * out.w2 * dz4 * secondOut;
                double grad_fourth_b = predictionLoss * outputDerivative * out.w2 * dz4;

                // First hidden layer gradients
                double dz1 = Neuron.sigmoidDerivative(z1);
                double dz2 = Neuron.sigmoidDerivative(z2);

                double chain1 = (out.w1 * third.w1 * dz3) + (out.w2 * fourth.w1 * dz4);
                double chain2 = (out.w1 * third.w2 * dz3) + (out.w2 * fourth.w2 * dz4);

                double grad_first_w1 = predictionLoss * outputDerivative * chain1 * dz1 * x1;
                double grad_first_w2 = predictionLoss * outputDerivative * chain1 * dz1 * x2;
                double grad_first_b = predictionLoss * outputDerivative * chain1 * dz1;

                double grad_second_w1 = predictionLoss * outputDerivative * chain2 * dz2 * x1;
                double grad_second_w2 = predictionLoss * outputDerivative * chain2 * dz2 * x2;
                double grad_second_b = predictionLoss * outputDerivative * chain2 * dz2;

                // =========================
                // UPDATE WEIGHTS
                // =========================

                out.update(grad_out_w1, grad_out_w2, grad_out_b, lr);

                third.update(grad_third_w1, grad_third_w2, grad_third_b, lr);
                fourth.update(grad_fourth_w1, grad_fourth_w2, grad_fourth_b, lr);

                first.update(grad_first_w1, grad_first_w2, grad_first_b, lr);
                second.update(grad_second_w1, grad_second_w2, grad_second_b, lr);
            }

            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch +
                        " Loss: " + (totalLoss / data.size()));
            }
        }
    }

    /**
     * Reads CSV file into a 2D list of strings
     */
    public static List<List<String>> readCSV(String filePath) {

        List<List<String>> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {

            String line;

            while ((line = br.readLine()) != null) {
                data.add(Arrays.asList(line.split(",")));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return data;
    }

    public static void main(String[] args) {

        ArrayList<double[]> trainingData = new ArrayList<double[]>();
        ArrayList<Double> answers = new ArrayList<Double>();

        List<List<String>> csv = readCSV("/workspaces/bello/src/NeutralNetwork/src/Passwords.csv");

        for (int i = 1; i < csv.size(); i++) {

            List<String> row = csv.get(i);

            double rank = Double.parseDouble(row.get(1));

            double strength = Double.parseDouble(row.get(6));

            double normalizedRank = rank / 500.0;
            double normalizedStrength = strength / 10.0;

            trainingData.add(new double[] {
                    normalizedRank,
                    normalizedStrength
            });

            String timeUnit = row.get(5);

            if (timeUnit.equals("days") ||
                    timeUnit.equals("years") ||
                    timeUnit.equals("centuries")) {

                answers.add(1.0);

            } else {

                answers.add(0.0);
            }
        }

        Network network = new Network();

        network.train(trainingData, answers);

        System.out.println("===================================");

        // Strong password example
        double strongPrediction = network.predict(0.05, 0.9);

        // Weak password example
        double weakPrediction = network.predict(0.9, 0.2);

        System.out.println("Strong password prediction: " + strongPrediction);
        System.out.println("Weak password prediction: " + weakPrediction);

        if (strongPrediction > 0.5) {
            System.out.println("Prediction: Strong password should take a long time to crack.");
        }

        if (weakPrediction < 0.5) {
            System.out.println("Prediction: Weak password should be cracked quickly.");
        }
    }
}
