using System;
using System.IO;
using System.Collections.Generic;


using Templates;
using Utility;
namespace NeuralNet{

	public class NeuralNetwork{
		
		LearningRate LeRa = new SinRate();
		
		Error Er = new QuadraticLoss();
		
		public List<Activation> Act = new List<Activation>();
	
		public List<Matrix> Weights = new List<Matrix>();
		
		private List<Matrix> WeightChanges = new List<Matrix>();
		
		public List<Matrix> Nodes = new List<Matrix>();
		
		private List<Matrix> NodeChanges = new List<Matrix>();
		
		private List<Matrix> Bias = new List<Matrix>();
		
		private List<Matrix> BiasChange = new List<Matrix>();
		
		private float LowerBounds;
		private float UpperBounds;
		
		public float fixedRate;

		public int dropout;
		
		private Random rnd;
		
		
		public NeuralNetwork(int L, float lR = .1f, float LowBound = 0, float UpBound = 1, int dropOut = 1){
			rnd = new Random();
			fixedRate = lR;
			dropout = dropOut;
			LowerBounds = LowBound;
			UpperBounds = UpBound;
			
			Nodes.Add(new Matrix(L, 1));
			
			Bias.Add(new Matrix(L, 1, .2f));
			
			Weights.Add(new Matrix(1,1));
			
			Act.Add(new Default());
			
			ResetConstructives();
		}
		
		public NeuralNetwork(string filePath){
			filePath += @"\FullyConnected.txt";
			string[] text = File.ReadAllLines(filePath);
			string[] delimChars = {", ", "," , " ", "\n"};
			
			string[] line = text[0].Split(delimChars, System.StringSplitOptions.RemoveEmptyEntries);
			
			fixedRate = Single.Parse(line[0]);
			LowerBounds = Single.Parse(line[1]);
			UpperBounds = Single.Parse(line[2]);
			dropout = Int32.Parse(line[3]);
			rnd = new Random();
			
			for(int i = 1; i < text.Length; i++){
				
				line = text[i].Split(delimChars, System.StringSplitOptions.RemoveEmptyEntries);
				int count = Int32.Parse(line[0]);
				
				Nodes.Add(new Matrix(count, 1));
				Bias.Add(new Matrix(count, 1));
				F utilityF = new F();
				Act.Add(utilityF.CheckType(line[1]));
				
				i += 2;
				
				line = text[i].Split(delimChars, System.StringSplitOptions.RemoveEmptyEntries);
				int row = Int32.Parse(line[0]);
				int column = Int32.Parse(line[1]);
				Weights.Add(new Matrix(row, column));
				for(int j = 0; j < row; j++){
					i++;
					line = text[i].Split(delimChars, System.StringSplitOptions.RemoveEmptyEntries);
					for(int k = 0; k < line.Length; k++){
						Weights[Weights.Count-1][j, k] = Single.Parse(line[k]);
					}
				}
				i += 3;
				line = text[i].Split(delimChars, System.StringSplitOptions.RemoveEmptyEntries);
				for(int j = 0; j < count; j++){
					Bias[Bias.Count-1][j, 0] = Single.Parse(line[j]);
				}
				i++;
			}
			
			ResetConstructives();
			
		}
		
		
		public void SaveNetwork(string filePath){
			filePath += @"\FullyConnected.txt";
			if(File.Exists(filePath))
				File.Delete(filePath);
			
			string toAdd = "";
			
			toAdd += fixedRate + ", " + LowerBounds + ", " + UpperBounds + ", " + dropout;
			toAdd += Environment.NewLine;
			
			for(int i = 0; i < Weights.Count; i++){
				toAdd += Nodes[i].length;
				toAdd += ", " + Act[i].type + Environment.NewLine;
				toAdd += Environment.NewLine;
				toAdd += Weights[i].length + ", " + Weights[i].height + Environment.NewLine;
				for(int j = 0; j < Weights[i].length; j++){
					for(int k = 0; k < Weights[i].height; k++)
						toAdd = toAdd + Weights[i][j, k] + ", ";
					toAdd = toAdd + Environment.NewLine;
				}
				toAdd += Environment.NewLine;
			
				toAdd += Bias[i].length;
				toAdd += Environment.NewLine;
				for(int j = 0; j < Bias[i].length; j++){
						toAdd = toAdd + Bias[i][j, 0] + ", ";
				}
				toAdd = toAdd + Environment.NewLine;
				toAdd = toAdd + Environment.NewLine;
			
			}
			
			File.AppendAllText(filePath, toAdd);
			
			
		}
		
		
		private void NewWeights(int index){
			index--;
			
			for(int i = 0; i < Weights[index].height; i++)
			{
				for(int j = 0; j < Weights[index].length; j++)
					Weights[index][j,i] = (float)(Math.Sqrt(-2 * Math.Log(rnd.Next(1, 99) / 100f)) * Math.Cos(2 * 3.14159f * rnd.Next(1, 99) / 100f));
			}
		}
		
		
		
		
		public void AddLayer(int L, string activationType){
			Weights.Add(new Matrix(L, Nodes[Nodes.Count-1].length, 1));
			
			NewWeights(Weights.Count-1);
			
			Nodes.Add(new Matrix(L, 1));
			
			Bias.Add(new Matrix(L, 1, .2f));
			F utilityF = new F();
			Act.Add(utilityF.CheckType(activationType));
			
			ResetConstructives();
		}
		
		
		public float Train(Epoch Ep){
			ResetConstructives();
			float totalError = 0;
			for(int i = 0; i < Ep.Runs; i++){
				for(int j = 0; j < Ep.Set.Length; j++)
					BackPropigate(Er.ErrorDeriv(RunNetwork(Ep.Set[j].Inputs), Ep.Set[j].Goals));
				ApplyBackProp(Ep.Set.Length, LeRa.CalcRate(fixedRate, i));
			}
			for(int j = 0; j < Ep.Set.Length; j++)
				totalError += Er.ErrorNumber(RunNetwork(Ep.Set[j].Inputs), Ep.Set[j].Goals);
			
			return totalError /= Ep.Set.Length;
		}
		
		
		
		
		public void ApplyBackProp(int Input, float learnRate){
			
			
			for(int i = 1; i < Weights.Count; i++){
				if(rnd.Next(1, dropout) != 2)
				{
					Weights[i] -= (WeightChanges[i]/Input) * learnRate;
				}
			}
			for(int i = 0; i < Bias.Count; i++)
				for(int j = 0; j < Bias[i].length; j++)
					Bias[i][j, 0] -= BiasChange[i][j, 0]/Input * learnRate;
			ResetConstructives();
		}
		
		
		private void ResetConstructives(){
			BiasChange = new List<Matrix>();
			for(int i = 0; i < Bias.Count; i++){
				BiasChange.Add(new Matrix(Bias[i].length, 1, 0));
			}
			
			WeightChanges = new List<Matrix>();
			WeightChanges.Add(new Matrix(1,1));
			for(int i = 1; i < Nodes.Count; i++){
				WeightChanges.Add(new Matrix(Nodes[i].length, Nodes[i-1].length, 0));	
			}
			
			ResetNodeChanges();
		}
		
		private void ResetNodeChanges(){
			NodeChanges = new List<Matrix>();
			for(int i = 0; i < Nodes.Count; i++)
				NodeChanges.Add(new Matrix(Nodes[i].length, 1, 0));
		}
		
		
		
		public float[] BackPropigate(float[] Error){
			for(int i = 0; i < NodeChanges[NodeChanges.Count-1].length; i++){
				NodeChanges[NodeChanges.Count-1][i, 0] = Error[i] * Act[Act.Count-1].ActiveFunctionDeriv(Nodes[NodeChanges.Count-1][i, 0]);
			}
			
			for(int i = NodeChanges.Count - 2; i >= 0; i--)
				for(int j = 0; j < NodeChanges[i].length; j++)
					for(int k = 0; k < Weights[i+1].length; k++)
						NodeChanges[i][j, 0] += NodeChanges[i+1][k, 0] * Weights[i+1][k, j] * Act[i].ActiveFunctionDeriv(Nodes[i][j, 0]);
					
			for(int i = 0; i < BiasChange.Count; i++)
				for(int j = 0; j < BiasChange[i].length; j++){
					BiasChange[i][j, 0] = Bias[i][j, 0] * NodeChanges[i][j, 0];
				}
			
			for(int i = 1; i < WeightChanges.Count; i++)
				for(int j = 0; j < WeightChanges[i].length; j++)
					for(int k = 0; k < WeightChanges[i].height; k++)
						WeightChanges[i][j, k] += NodeChanges[i][j, 0] * Nodes[i-1][k, 0];
			
			
		
			float[] toReturn = new float[NodeChanges[0].length];
			for(int i = 0; i < toReturn.Length; i++){
				toReturn[i] = NodeChanges[0][i, 0];
			}
			
			ResetNodeChanges();
		
			return toReturn;
		}
		
		
		
		
		public float[] OutputNetwork(InOutPair IOP){
			float[] returnArray = RunNetwork(IOP.Inputs);
			Console.WriteLine("");
			Console.WriteLine("The output of the network is :");
			for(int i = 0; i < returnArray.Length; i++){
				Console.WriteLine("Node " + (i+1) + ": " + returnArray[i]);
			}
			Console.WriteLine("");
			return returnArray;
		}
		
		
		public float[] RunNetwork(float[] Inputs){

			for(int i = 0; i < Nodes[0].length; i++)
				Nodes[0][i, 0] = (Inputs[i] - LowerBounds) / (UpperBounds - LowerBounds);
		
			
			for(int i = 1; i < Nodes.Count; i++){
				Nodes[i] = ActivationMatrix((Weights[i] * Nodes[i-1]) + Bias[i], i);
			
			}
			float[] toReturn = new float[Nodes[Nodes.Count-1].length];
			
			for(int i = 0; i <  Nodes[Nodes.Count-1].length; i++)
				toReturn[i] = Nodes[Nodes.Count-1][i, 0];
			
			return toReturn;
		}
		
		
		private Matrix ActivationMatrix(Matrix input, int index){
			Matrix temp = new Matrix(input.length, input.height, 0);
			
			for(int i = 0; i < input.length; i++){
				for(int j = 0; j < input.height; j++){
					temp[i,j] = Act[index].ActiveFunction(input[i,j]);
				}
			}
			
			return temp;
		}
	}
}