using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Encog;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.Neural.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Neural.NeuralData;


namespace ConsoleApp1
{
    class Program
    {
        private static readonly double AcceptableError = 0.001;
        private static readonly int MaxEpoch = 5000;
        private static readonly string[] _literki = { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "R","S", "T", "U" };
        private static readonly double[][] AndInput =
        {
            new double[25] {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1},
            new double[25] {1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0},
            new double[25] {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1},
            new double[25] {1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0},
            new double[25] {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1},
            new double[25] {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0},
            new double[25] {1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0},
            new double[25] {1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1},
            new double[25] {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0},
            new double[25] {1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0},
            new double[25] {1,0,0,1,0,1,0,1,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,1,0},
            new double[25] {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0},
            new double[25] {1,1,0,1,1,1,0,1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1},
            new double[25] {1,1,0,0,1,1,1,1,0,1,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1},
            new double[25] {1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1},
            new double[25] {1,1,1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0},
            new double[25] {1,1,1,1,0,1,0,0,1,0,1,1,1,1,0,1,0,1,0,0,1,0,0,1,0},
            new double[25] {1,1,1,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0},
            new double[25] {1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0},
            new double[25] {1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,1,1,1,0}
        };
        private static readonly double[][] AndIdeal = {
            new double[20] { 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            new double[20] { 0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            new double[20] { 0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            new double[20] { 0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            new double[20] { 0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            new double[20] { 0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
            new double[20] { 0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
            new double[20] { 0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
            new double[20] { 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
            new double[20] { 0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
            new double[20] { 0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
            new double[20] { 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
            new double[20] { 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
            new double[20] { 0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
            new double[20] { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
            new double[20] { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
            new double[20] { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
            new double[20] { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},
            new double[20] { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0},
            new double[20] { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}
        };

        static void Main(string[] args)
        {
            INeuralDataSet trainingSet = new BasicNeuralDataSet(AndInput, AndIdeal);
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(new ActivationRamp(), true, 25));
            network.AddLayer(new BasicLayer(new ActivationRamp(), true, 75));
            network.AddLayer(new BasicLayer(new ActivationRamp(), true, 50));
            network.AddLayer(new BasicLayer(new ActivationRamp(), true, 20));
            network.Structure.FinalizeStructure();
            network.Reset();

            ITrain train = new Backpropagation(network, trainingSet,0.02,0.3);
            
            int epoch = 1;
            do
            {
                train.Iteration();
               
                Console.WriteLine($"{train.Error}");
                epoch++;
            } while ((epoch < MaxEpoch) && (train.Error > AcceptableError));

            
            
            var input = new BasicMLData(new double[25] { 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1 });
           
            int best = network.Winner(input);

            Console.WriteLine($"Rozpoznano: {_literki[best]}" );

            foreach (IMLDataPair pair in trainingSet)
            {
                IMLData output = network.Compute(pair.Input);

                Console.WriteLine($"wynik : { output[0].ToString("0.000")} {output[1].ToString("0.000")} {output[2].ToString("0.000")} {output[3].ToString("0.000")} {output[4].ToString("0.000")} {output[5].ToString("0.000")} {output[6].ToString("0.000")} {output[7].ToString("0.000")} {output[8].ToString("0.000")} {output[9].ToString("0.000")} {output[10].ToString("0.000")} {output[11].ToString("0.000")} {output[12].ToString("0.000")} {output[13].ToString("0.000")} {output[14].ToString("0.000")} {output[15].ToString("0.000")} {output[16].ToString("0.000")} {output[17].ToString("0.000")} {output[18].ToString("0.000")} {output[19].ToString("0.000")}");
            }






           

            Console.ReadKey();
        }
        public static double[] findMaximumIndex(double[][] a)
        {
            double maxVal = -99999;
            double[] answerArray = new double[2];
            for (int row = 0; row < a.Length; row++)
            {
                for (int col = 0; col < a[row].Length; col++)
                {
                    if (a[row][col] > maxVal)
                    {
                        maxVal = a[row][col];
                        answerArray[0] = row;
                        answerArray[1] = col;
                    }
                }
            }
            return answerArray;
        }
    }
}
