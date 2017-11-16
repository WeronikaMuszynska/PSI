using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    class Program
    {
        private readonly Network _network = new Network();
        private readonly string[] _literki = {"A", "B","C","D","E","F","G","H","I","J", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j" };

        private  static int imageSize = 5;
        static void Main(string[] args)
        {
            var p = new Program();
            p._network.maximumIteration = int.Parse(10000.ToString());
            p.Test();
            p.Check();

            Console.ReadKey();
        }

        private void Test()
        {
            int length = _literki.Length;
            _network.Initialize(imageSize * imageSize, length);

            double[][] inputs = new double[length][];
            double[][] outputs = new double[length][];

            for (int i = 0; i < length; ++i)
            {
                outputs[i] = new double[length];

                for (int j = 0; j < length; ++j)
                {
                    outputs[i][j] = i == j ? 1.0 : 0.0;
                }

                _network.outputLayer[i].Value = _literki[i];
            }

            inputs[0] = new double[25] {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1};
            inputs[1] = new double[25] {1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0};
            inputs[2] = new double[25] {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1};
            inputs[3] = new double[25] {1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0};
            inputs[4] = new double[25] {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1};
            inputs[5] = new double[25] {1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
            inputs[6] = new double[25] {1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0};
            inputs[7] = new double[25] {1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1};
            inputs[8] = new double[25] {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
            inputs[9] = new double[25] {1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0};
            inputs[10] = new double[25] {1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};
            inputs[11] = new double[25] {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0};
            inputs[12] = new double[25] {0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0};
            inputs[13] = new double[25] {0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0};
            inputs[14] = new double[25] {0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0};
           inputs[15] = new double[25] {0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0};
            inputs[16] = new double[25] {1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0};
            inputs[17] = new double[25] {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0};
            inputs[18] = new double[25] {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0};
            inputs[19] = new double[25] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0};

            _network.TrainNetwork(inputs, outputs);

            Console.WriteLine("Errors:");
            for (int i = 0; i < _network.currentIteration; ++i)
            {
                Console.WriteLine( i + "\t"+ _network.errors[i].ToString("#0.000000"));
            }
        }

        private void Check()
        {
            double[] sample = new double[imageSize * imageSize];

            sample = new double[25] { 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0 };

            _network.Recognize(sample);

            Console.WriteLine("Wyniki:");
            var wyniki = new double[_literki.Length];
            for (int i = 0; i < _network.outputLayer.Length; ++i)
            {
                wyniki[i] = _network.outputLayer[i].Output;
                Console.WriteLine(_network.outputLayer[i].Value);
                Console.WriteLine(_network.outputLayer[i].Output.ToString("#0.000000"));
            }

            int w = wyniki.ToList().IndexOf(wyniki.Max());
            if(w > _literki.Length/2)
                Console.WriteLine("Wykryto małą literkę: " + _network.outputLayer[w].Value);
            else
            {
                Console.WriteLine("Wykryto dużą literkę: " + _network.outputLayer[w].Value);
            }
        }


    }

    class Network
    {
        public struct InputLayer
        {
            public double Value;
            public double[] Weights;
        }

        public struct OutputLayer
        {
            public double InputSum;
            public double Output;
            public double Error;
            public double Target;
            public string Value;
        }

        public double learningRate = 0.6;

        public int ImageSize = 0;
        public int InputNum = 0;
        public int OutputNum = 0;

        public InputLayer[] inputLayer = null;
        public OutputLayer[] outputLayer = null;

        public double[] errors = null;

        public int currentIteration = 0;
        public int maximumIteration = 1000;

        public void Initialize(int inputSize, int outputSize)
        {
            InputNum = inputSize;
            OutputNum = outputSize;

            inputLayer = new Network.InputLayer[inputSize];
            outputLayer = new Network.OutputLayer[outputSize];

            // Zainicjuj wagi losowymi wartościami z zakresu 0.01 - 0.02.
            Random random = new Random();

            for (int i = 0; i < InputNum; ++i)
            {
                inputLayer[i].Weights = new double[OutputNum];

                for (int j = 0; j < OutputNum; ++j)
                {
                    inputLayer[i].Weights[j] = random.Next(1, 3) / 100.0;
                }
            }
        }

        public bool TrainNetwork(double[][] inputs, double[][] outputs)
        {
            double currentError = 0.0, maximumError = 0.01;

            currentIteration = 0;

            // Utwórz tablicę do przechowywania wartości kolejnych błędów.
            errors = new double[maximumIteration];

            do
            {
                currentError = 0;

                for (int i = 0; i < inputs.Length; ++i)
                {
                    CalculateOutput(inputs[i], outputLayer[i].Value);
                    BackPropagation();

                    currentError += GetError();
                }

                errors[currentIteration] = currentError;

                ++currentIteration;
            }
            while (currentError > maximumError && currentIteration < maximumIteration);

            // Jeżeli maksymalny błąd został osiągnięty w mniejszej liczbie iteracji,
            // to nauka sieci zakończyła się pomyślnie.
            if (currentIteration <= maximumIteration)
            {
                return true;
            }

            return false;
        }

        private void CalculateOutput(double[] pattern, string output)
        {
            // Przypisz wejście do warstwy wejściowej.
            for (int i = 0; i < pattern.Length; i++)
            {
                inputLayer[i].Value = pattern[i];
            }

            // Oblicz wejście, wyjście, wartość oczekiwaną oraz błąd pierwszej warstwy.
            for (int i = 0; i < OutputNum; i++)
            {
                double total = 0.0;

                for (int j = 0; j < InputNum; j++)
                {
                    total += inputLayer[j].Value * inputLayer[j].Weights[i];
                }

                outputLayer[i].InputSum = total;
                outputLayer[i].Output = Activation(total);
                outputLayer[i].Target = outputLayer[i].Value.CompareTo(output) == 0 ? 1.0 : 0.0;
                outputLayer[i].Error = (outputLayer[i].Target - outputLayer[i].Output) * (outputLayer[i].Output) * (1 - outputLayer[i].Output);
            }
        }

        private void BackPropagation()
        {
            // Popraw wagi warstwy wejściowej.
            for (int j = 0; j < OutputNum; j++)
            {
                for (int i = 0; i < InputNum; i++)
                {
                    inputLayer[i].Weights[j] += learningRate * (outputLayer[j].Error) * inputLayer[i].Value;
                }
            }
        }

        private double GetError()
        {
            double total = 0.0;

            for (int j = 0; j < OutputNum; j++)
            {
                total += Math.Pow((outputLayer[j].Target - outputLayer[j].Output), 2.0) / 2.0;
            }

            return total;
        }

        private double Activation(double x)
        {
            return (1.0 / (1.0 + Math.Exp(-x)));
        }

        public void Recognize(double[] input)
        {
            for (int i = 0; i < InputNum; i++)
            {
                inputLayer[i].Value = input[i];
            }

            for (int i = 0; i < OutputNum; i++)
            {
                double total = 0.0;

                for (int j = 0; j < InputNum; j++)
                {
                    total += inputLayer[j].Value * inputLayer[j].Weights[i];
                }

                outputLayer[i].InputSum = total;
                outputLayer[i].Output = Activation(total);
            }
        }
    }

}
