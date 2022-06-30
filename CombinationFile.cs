using System;
using System.IO;
using System.Collections.Generic;

using Templates;
using Utility;
using NeuralNet;
using ConvolutionFile;

namespace CombinationFile{
	public class Combination{
		
		LearningRate LeRa = new SinRate();
			
		Error Er = new QuadraticLoss();
		
		public Convolution Conv;
		public NeuralNetwork Net;
		public int dropout;
		
		public Combination(Convolution C, NeuralNetwork N){
			Conv = C;
			Net = N;
		}
		
		public Combination(string path){
			Conv = new Convolution(path);
			Net = new NeuralNetwork(path);
		}
		
		public float[] Run(float[] Input){
			return Net.RunNetwork(Conv.RunNetwork(Input));
		}
		
		public void SaveCombination(string path){
			Net.SaveNetwork(path);
			Conv.SaveNetwork(path);
		}
		
		
		public float Train(Epoch E){
			for(int i = 0; i < E.Runs; i++){
				for(int j = 0; j < E.Set.Length; j++){
					float[] neuralError = Er.ErrorDeriv(Net.RunNetwork(Conv.RunNetwork(E.Set[j].Inputs)), E.Set[j].Goals);
					float[] convError = Net.BackPropigate(neuralError);
					Conv.backPropigate(convError);
					Net.BackPropigate(neuralError);
				}
				Conv.ApplyError(E.Set.Length, LeRa.CalcRate(Conv.learnRateConstant, i));
				Net.ApplyBackProp(E.Set.Length, LeRa.CalcRate(Net.fixedRate, i));
			}
			float totalError = 0;
			for(int i = 0; i < E.Set.Length; i++)
				totalError += Er.ErrorNumber(Net.RunNetwork(Conv.RunNetwork(E.Set[i].Inputs)), E.Set[i].Goals);
			return totalError /= E.Set.Length;
		}
		
		public void SetDropout(int dropOut){
			Net.dropout = dropOut;
			Conv.dropout = dropOut;	
		}
		
		public int dropOut
		{
			get
			{
				return dropout;
			}
			set
			{
				Net.dropout = value;
				Conv.dropout = value;	
				dropout = value;
			}
		}
		
		public void RunPrint(float[] Input){
			Console.WriteLine();
			float[] output = Net.RunNetwork(Conv.RunNetwork(Input));
			for(int i = 0; i < output.Length; i++)
				Console.WriteLine("Node " + (i+1) + ": " + output[i]);
			Console.WriteLine();
		}
		
	}
}