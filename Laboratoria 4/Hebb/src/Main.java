


public class Main {

    public static void main(String[] args) {
        HebbProgram hebb = new HebbProgram(0.2,0);
        hebb.Train();
        hebb.Test();
    }
}