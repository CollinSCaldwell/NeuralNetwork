using System;

using System.IO;

using CombinationFile;
using Templates;
using Utility;
using NeuralNet;
using ConvolutionFile;

public class Program
{
	public static void Main()
	{	
	
		int height = 100;
		int width = 100;
		
		
		Random rnd = new Random();
		
		float[] toAdd = new float[height*width];		
		for(int i = 0; i < height*width; i++){
			toAdd[i] = 100f/i+width/height+(rnd.Next()%100f)/100f;
		}
		
		InOutPair IO = new InOutPair(toAdd, new float[] {.25f, .6f, 0f, .75f, .01f});

		Epoch Ep = new Epoch(new InOutPair[] {IO}, 10000);
		

		
		NeuralNetwork CustomNeuralNetwork = new NeuralNetwork(10, .005f);

		CustomNeuralNetwork.AddLayer(5, "Sigmoid");
		CustomNeuralNetwork.AddLayer(5, "HypTan");
		CustomNeuralNetwork.AddLayer(5, "Sigmoid");
		CustomNeuralNetwork.AddLayer(7, "Sigmoid");
		CustomNeuralNetwork.AddLayer(5, "HypTan");
		
		
		
		Convolution CustomConvolution = new Convolution(height, width, .005f);

		CustomConvolution.AddLayer(10, 10, 3);
		CustomConvolution.AddLayer(5, 5, 1);
		CustomConvolution.AddLayer(2, 2, 2, "HypTan");
		CustomConvolution.AddLayer(5, 5, 1, true);
		CustomConvolution.AddLayer(2, 2, 1, "HypTan");
		






		Combination C = new Combination(CustomConvolution, CustomNeuralNetwork);
		//Combination C = new Combination(Directory.GetCurrentDirectory());



		C.dropOut = 1;		


		C.RunPrint(toAdd);
		
		Console.WriteLine();
		Console.WriteLine("Training Error Rate: " + C.Train(Ep));
		
		Console.WriteLine();
		
		C.RunPrint(toAdd);
		
		C.SaveCombination(Directory.GetCurrentDirectory());
		
	}
}

