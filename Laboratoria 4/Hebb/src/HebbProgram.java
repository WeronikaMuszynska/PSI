import java.util.Random;

public class HebbProgram {
    public double weight[];
    double learning_rate;
    double forgetting_rate;

    public static final char[] letters = {'A','B','C','D','E','F','G','H','I','J','M','N','O','P','R','S','T','U','W','X'};
    public static final int[][] learning_letters = {
            {0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1},
            {1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,0},
            {0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0},
            {1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,0},
            {1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1},
            {1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0},
            {0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,1,1,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0},
            {1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1},
            {0,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0},
            {1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0},
            {1,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1},
            {1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1},
            {0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0},
            {1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0},
            {1,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,0,1,0,1,0,0,1,0,0,1,0,1,0,0,0,1},
            {0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0},
            {1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0},
            {1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0},
            {1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0},
            {1,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,1,0,0,0,1}
    };

    public HebbProgram(double learning_rate, double forgetting_rate) {
        weight = new double[35];
        this.learning_rate = learning_rate;
        this.forgetting_rate = forgetting_rate;
        initializeWeight();
    }


    public void Train() {
        for (int n = 0; n < 20; n++) {
            double sum;
            int[] tab;
            tab = learning_letters[n];
            sum= countSum(tab, weight);
            for (int i = 0; i < 35; i++) {
                weight[i] = weight[i] * forgetting_rate + learning_rate * activationFunction(sum) * (tab[i] - weight[i]);
            }
        }
    }

    public void Test(){
        for (int n = 0; n <20 ; n++) {
            double result;
            int[] tab;
            tab = learning_letters[n];

            double sum = countSum(tab, weight);
            result = activationFunction(sum);

            System.out.println("Letter "+ letters[n] +"-> Result: "+ result);
        }
    }

    public double countSum(int[] table, double[] weight){
        double sum = 0.0;
        for(int x=0; x < table.length; x++)
            sum += table[x] * weight[x];
        return sum;
    }

    private void initializeWeight() {
        Random random = new Random();
        for (int i = 0; i < 35; i++) {
            weight[i] = random.nextDouble();
        }
    }

    public double activationFunction(double s){
        double result = (1 - Math.exp(-s)) / (1 + Math.exp(-s));
        return result;
    }
}