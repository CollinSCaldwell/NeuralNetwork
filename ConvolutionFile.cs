using System;
using System.IO;
using System.Collections.Generic;


using Templates;
using Utility;
namespace ConvolutionFile{
	
	public class Convolution{
		
		LearningRate LeRa = new SinRate();
		
		Error Er = new QuadraticLoss();
		
		public List<float[]> Network = new List<float[]>();
		public List<float[]> Filters = new List<float[]>();
		public List<float> FilterHeights = new List<float>();
		public List<float> Heights = new List<float>();
		public List<int> Jump = new List<int>();
		public List<int> Method = new List<int>();
		
		public List<int> Dimension = new List<int>();
		
		public List<float[]> NodeErrors = new List<float[]>();
		public List<float[]> FilterErrors = new List<float[]>();
		
		public List<Activation> Act = new List<Activation>();
		
		public float[] Input;
		
		public float lowerBound;
		public float upperBound;
		
		public int dropout;
		
		public float learnRateConstant;
		
		public Random rnd;
		
		public Convolution(int h, int w, float learnR = .01f, float lowerBounds = 0, float upperBounds = 1, int dropO = 1){
			rnd = new Random();
			dropout = dropO;
			learnRateConstant = learnR;
			
			lowerBound = lowerBounds;
			upperBound = upperBounds;
			
			Network.Add(new float[h*w]);
			NodeErrors.Add(new float[h*w]);
			Filters.Add(new float[0]);
			FilterErrors.Add(new float[0]);
			FilterHeights.Add(1);
			Heights.Add(h);
			Jump.Add(1);
			Method.Add(0);
			Act.Add(new Default());
			
		}
		
		public Convolution(string filePath){
			
			rnd = new Random();
			filePath += @"\Convolution.txt";
			string[] text = File.ReadAllLines(filePath);
			string[] delimChars = {", ", "," , " ", "\n"};
			string[] line;
			F utilityF = new F();
			for(int i = 0; i < text.Length; i++){
				line = text[i].Split(delimChars, System.StringSplitOptions.RemoveEmptyEntries);
				Network.Add(new float[Int32.Parse(line[0])]);
				NodeErrors.Add(new float[Int32.Parse(line[0])]);
				Filters.Add(new float[Int32.Parse(line[1])]);
				FilterErrors.Add(new float[Int32.Parse(line[1])]);
				FilterHeights.Add(Single.Parse(line[2]));
				Heights.Add(Single.Parse(line[3]));
				Jump.Add(Int32.Parse(line[4]));
				Method.Add(Int32.Parse(line[5]));
				Act.Add(utilityF.CheckType(line[6]));
				dropout = Int32.Parse(line[7]);
				lowerBound = Single.Parse(line[8]);
				upperBound = Single.Parse(line[9]);
				
				i += 2;
				
				line = text[i].Split(delimChars, System.StringSplitOptions.RemoveEmptyEntries);
				for(int j = 0; j < Filters[Filters.Count-1].Length; j++){
					Filters[Filters.Count-1][j] = Single.Parse(line[j]);
				}
				i += 2;
			}
			
			clearLayers();
			
		}
		
		public void SaveNetwork(string filePath){
			filePath += @"\Convolution.txt";
			if(File.Exists(filePath))
				File.Delete(filePath);
			
			string toAdd = "";
			
			
			
			for(int i = 0; i < Network.Count; i++){
				toAdd += Network[i].Length;
				toAdd += ", " + Filters[i].Length;
				toAdd += ", " + FilterHeights[i];
				toAdd += ", " + Heights[i];
				toAdd += ", " + Jump[i];
				toAdd += ", " + Method[i];
				toAdd += ", " + Act[i].type;
				toAdd += ", " + dropout;
				toAdd += ", " + lowerBound;
				toAdd += ", " + upperBound;
				toAdd += Environment.NewLine;
				toAdd += Environment.NewLine;
				
				for(int j = 0; j < Filters[i].Length; j++){
					toAdd = toAdd + Filters[i][j] + ", ";
				}
				
				toAdd += Environment.NewLine;
				toAdd += Environment.NewLine;
				toAdd += Environment.NewLine;
			
			}
			
			File.AppendAllText(filePath, toAdd);
			
			
		}
		
		public float Train(Epoch E){
			
			for(int i = 0; i < E.Runs; i++){
				for(int j = 0; j < E.Set.Length; j++){
					backPropigate(Er.ErrorDeriv(RunNetwork(E.Set[j].Inputs), E.Set[j].Goals));
				}
				ApplyError(E.Set.Length, LeRa.CalcRate(learnRateConstant, i));
				clearFilters();
			}
			
			
			float totalError = 0;
			for(int i = 0; i < E.Set.Length; i++)
				totalError += Er.ErrorNumber(RunNetwork(E.Set[i].Inputs), E.Set[i].Goals);
			
			return totalError/E.Set.Length;
		}
		
		
		
		public void AddLayer(int fH, int fW, int j, string activationType){
			AddEntireLayer(fH, fW, j, false);
			F utilityF = new F();
			Act.Add(utilityF.CheckType(activationType));
		}
		
		
		
		public void AddLayer(int fH, int fW, int j = 1, bool pool = false){
			AddEntireLayer(fH, fW, j, pool);
			Act.Add(new Default());
		}
		
		private void AddEntireLayer(int fH, int fW, int j = 1, bool pool = false){
			
			int index = Filters.Count;
			
			
			if(pool)
				Method.Add(1);
			else
				Method.Add(0);
			
			Filters.Add(new float[fH * fW]);
			FilterErrors.Add(new float[fH*fW]);
			for(int i = 0; i < Filters[Filters.Count-1].Length; i++){
				Filters[index][i] = (float)(Math.Sqrt(-2 * Math.Log(rnd.Next(1, 99) / 100f)) * Math.Cos(2 * 3.14159f * rnd.Next(1, 99) / 100f));
				FilterErrors[index][i] = 0;
			}
			
			
			FilterHeights.Add(fH);
			Jump.Add(j);
			
			float tempNum = Heights[index-1] - (FilterHeights[index] - 1);
			if(j != 1)
				tempNum = Heights[index-1] / j;
			Heights.Add(tempNum);
			
			NodeErrors.Add(new float[0]);
			Network.Add(new float[0]);
			if(j == 1){
				Network[Network.Count-1] = new float[(int)(tempNum * ((Network[index-1].Length/Heights[index-1]) - ((Filters[index].Length / FilterHeights[index]) - 1)))];
				NodeErrors[NodeErrors.Count-1] = new float[Network[Network.Count-1].Length];
			}else{
				Network[Network.Count-1] = new float[(int)tempNum * ((int)(Network[index-1].Length/Heights[index-1])/j)];
				NodeErrors[NodeErrors.Count-1] = new float[Network[Network.Count-1].Length];
			}
		}
		
		
		public float[] RunNetwork(float[] Input){
			clearLayers();
			
			for(int i = 0; i < Input.Length; i++)
				Input[i] = (Input[i] - lowerBound) / (upperBound - lowerBound);
			
			
			Network[0] = Input;
			for(int i = 0; i < Network.Count; i++){
				calculateLayer(i);
			}
			
			
			return Network[Network.Count-1];
		}
		
		
		public void backPropigate(float[] error){
			clearNodes();
			for(int i = 0; i < error.Length; i++){
				NodeErrors[NodeErrors.Count-1][i] = error[i] * Act[Act.Count-1].ActiveFunctionDeriv(Network[Network.Count-1][i]);
			}
			
			for(int i = Network.Count-1; i > 0; i--){
				propigateLayer(i);
			}
			
		}
		
		private void clearLayers(){
			for(int i = 1; i < Network.Count; i++){
				for(int j = 0; j < Network[i].Length; j++){
					Network[i][j] = 0;
				}
			}
		}
		
		private void clearNodes(){
			for(int i = 1; i < NodeErrors.Count; i++){
				for(int j = 0; j < NodeErrors[i].Length; j++){
					NodeErrors[i][j] = 0;
				}
			}
		}
		
		private void clearFilters(){
			for(int i = 0; i < FilterErrors.Count; i++){
				for(int j = 0; j < FilterErrors[i].Length; j++){
					FilterErrors[i][j] = 0;
				}
			}
		}
		
		
		private void calculateLayer(int layer){
			int newWidth = (int)(Network[layer].Length / Heights[layer]);
			
			if(Jump[layer] != 1){
				newWidth = (int)(Network[layer-1].Length/Heights[layer-1])/(int)Jump[layer];
			}
			
			int hCounter = 0;
			int wCounter = 0;
			
			
			for(int i = 0; i < Network[layer].Length; i++){
				
				
				wCounter += Jump[layer];
				if(wCounter >= newWidth){
					wCounter = 0;
					hCounter += Jump[layer];
				}
				
				float[] temp = new float[Filters[layer].Length];
				
				for(int j = 0; j < FilterHeights[layer]; j++){
					for(int k = 0; k < Filters[layer].Length/FilterHeights[layer]; k++){
						if(Method[layer] == 0){
							Network[layer][i] += Network[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))] * Filters[layer][(int)(Filters[layer].Length/FilterHeights[layer]*j) + k];
							
						}else{
							if(Method[layer] == 1){
								temp[(int)(FilterHeights[layer]*j) + k] = Network[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))];
							}
						}
					}
					
				}
				
				
				if(Method[layer] == 1){
					float max = temp[0];
					for(int j = 0; j < temp.Length; j++){
						if(temp[j] >= max)
							max = temp[j];
					}
					Network[layer][i] = max;	
				}else{
					Network[layer][i] = Act[layer].ActiveFunction(Network[layer][i]);
				}
				
			}
			
		}
		
		
		private void propigateLayer(int layer){
			
			int newWidth = (int)(Network[layer].Length / Heights[layer]);
			
			if(Jump[layer] != 1){
				newWidth = (int)(Network[layer-1].Length/Heights[layer-1])/(int)Jump[layer];
			}
			
			int hCounter = 0;
			int wCounter = 0;
			
			Random rnd = new Random();
			
			
			float[] tempFilter = new float[Filters[layer].Length];
			
			for(int i = 0; i < tempFilter.Length; i++){
				tempFilter[i] = 0;
			}
			
			
			for(int i = 0; i < Network[layer].Length; i++){
				wCounter += Jump[layer];
				if(wCounter >= newWidth){
					wCounter = 0;
					hCounter += Jump[layer];
				}
				
				float[] temp = new float[Filters[layer].Length];
				int[] indices = new int[Filters[layer].Length];
				
				for(int j = 0; j < FilterHeights[layer]; j++){
					for(int k = 0; k < Filters[layer].Length/FilterHeights[layer]; k++){
						if(Method[layer] == 0){
							tempFilter[(int)(Filters[layer].Length/FilterHeights[layer]*j) + k] += Network[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))] * NodeErrors[layer][i];
							NodeErrors[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))] += NodeErrors[layer][i] * Filters[layer][(int)(Filters[layer].Length/FilterHeights[layer]*j) + k] * Act[layer-1].ActiveFunctionDeriv(Network[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))]);
						}else{
							if(Method[layer] == 1){
								indices[(int)(FilterHeights[layer]*j) + k] = (int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter));
								temp[(int)(FilterHeights[layer]*j) + k] = Network[layer-1][(int)((FilterHeights[layer] * (j+hCounter)) + (k+wCounter))];
							}
						}
					}
					
				}
				
				if(Method[layer] == 1){
					int index = 0;
					float max = temp[0];
					for(int j = 0; j < temp.Length; j++){
						if(temp[j] >= max){
							index = indices[j];
							max = temp[j];
						}
					}
					NodeErrors[layer-1][index] += max;	
				}
					
			}
			
			for(int i = 0; i < Filters[layer].Length; i++){
				FilterErrors[layer][i] += tempFilter[i]/Network[layer].Length;
			}
				
			
			
		}
		
		public void ApplyError(int total, float learnR){
			for(int i = 0; i < FilterErrors.Count; i++){
				for(int j = 0; j < FilterErrors[i].Length; j++){
					if(rnd.Next(1, dropout) != 2){	
						Filters[i][j] -= FilterErrors[i][j] / total * learnR;
					}
				}
			}
		}
	}
}