import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Network {
    Neuron first = new Neuron();
    Neuron second = new Neuron();
    Neuron out = new Neuron();

     /*
     * Data flows:
     * inputs → hidden layer → output layer
     */
    public double predict(double x1, double x2) {

        // Pass inputs into hidden neurons
        double firstOut = first.predict(x1, x2);
        double secondOut = second.predict(x1, x2);

        // Hidden outputs become inputs to final neuron
        return out.predict(firstOut, secondOut);
    }

    public static void main(String [] args){
        Network network = new Network();
        Double prediction = network.predict(2.2, 30.5);
        System.out.println("prediction: " + prediction);
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


}
