using System;
using System.IO;
using System.Collections.Generic;



using Templates;
namespace Utility{
	
	public class F{
		public Activation[] activationList = {new Default(), new HyperbolicTangent(), new Arctan(), new Sigmoid()};
		public F(){
			
		}

		public static void PrintArray(float[] input){
			Console.WriteLine();
			for(int i = 0; i < input.Length; i++)
				Console.WriteLine(input[i]);
			Console.WriteLine();
		}
		
		public static float Average(float[] input){
			float sum = 0;
			for(int i = 0;i < input.Length; i++)
				sum += input[i];
			return sum/input.Length;
		}
		
		
		public Activation CheckType(string input){
			for(int i = 0; i < activationList.Length; i++){
				if(activationList[i].type == input){
					return activationList[i];
				}
			}
			return activationList[0];
		}
	}
	
	
	
	
	public class SinRate : LearningRate{
		public override float CalcRate(float fixedRate, int counter){
			return (float)(Math.Sin(fixedRate*counter)/2)+.51f;
		}
	}
	
	
	
	public class DefaultRate : LearningRate{
		public override float CalcRate(float fixedRate, int counter){
			return fixedRate;
		}
	}



	
	public class QuadraticLoss : Error{
		
		public override float[] ErrorDeriv(float[] input, float[] desired){
			float[] output = new float[input.Length];
			for(int i = 0; i < output.Length; i++)
				output[i] = input[i]-desired[i];
			return output;
		}	
		
		public override float ErrorNumber(float[] input, float[] desired){
			float total = 0;
			for(int i = 0; i < input.Length; i++)
				total = ((float)Math.Pow(input[i]-desired[i], 2))/2;
			return total/input.Length;
		}
	}
	

	
	
	
	
	public class HyperbolicTangent : Activation
	{
		public override string type { get {return "HypTan";} set {}}
		
		public override float ActiveFunction(float input){		
			float positive = (float)Math.Pow(Math.E , input);
			float negative = (float)Math.Pow(Math.E , -input);
			if(Single.IsNaN((positive-negative)/(positive+negative))){
				return 0;
			}
			return (positive-negative)/(positive+negative);
		}
		
		public override float ActiveFunctionDeriv(float input){		
			return (1-(float)Math.Pow(input, 2));
		}
		
	}
	
	public class Arctan : Activation
	{
		public override string type { get {return "Arctan";} set {}}
		
		public override float ActiveFunction(float input){
			return (float)Math.Atan(input);
		}
		
		public override float ActiveFunctionDeriv(float input){
			return (float)(1/(Math.Pow(input, 2)+1));
		}
	}

	public class Sigmoid : Activation
	{
		public override string type { get {return "Sigmoid";} set {}}
		
		public override float ActiveFunction(float input){
			return (float)1/(float)(1+Math.Pow(Math.E, -input));
		}
		
		public override float ActiveFunctionDeriv(float input){
			return (float)(input)*(1-input);
		}
	}

	public class Default : Activation
	{
		public override string type { get {return "Default";} set {}}
		
		public override float ActiveFunction(float input){
			return input;
		}
		
		public override float ActiveFunctionDeriv(float input){
			return 1;
		}
		
	}


	public class InOutPair{
		public float[] Inputs;
		public float[] Goals;
		
		public InOutPair(float[] I, float[] G){
			Inputs = new float[I.Length];
			Goals = new float[G.Length];
			
			for(int i = 0; i < Inputs.Length; i++)
				Inputs[i] = I[i];
			for(int i = 0; i < Goals.Length; i++)
				Goals[i] = G[i];
			
		}
		
		
		
	}

	public class Epoch{
		public InOutPair[] Set;
		public int Runs;
		
		public Epoch(InOutPair[] S, int R = 10){
			Runs = R;
			
			Set = S;
		}
	}

	public class Matrix{
		private float[][] M;
		
		public int length;
		public int height;
		
		public float[] this[int i]
		{
			get
			{
				return M[i];
			}
			set
			{
				M[i] = value;
			}
		}
		
		public float this[int i, int j]
		{
			get
			{
				return M[i][j];
			}
			set
			{
				M[i][j] = value;
			}
		}
		
		
		
		public Matrix(float[][] Input){
			length = Input.Length;
			height = Input[0].Length;
			M = new float[length][];
			for(int i = 0; i < length; i++){
				M[i] = new float[height];
				for(int j = 0; j < height; j++){
					M[i][j] = Input[i][j];
				}
			}
		}
		
		
		
		public Matrix(int A, int B, float Init = 0){
			length = A;
			height = B;
			M = new float[length][];
			for(int i = 0; i < length; i ++){
				M[i] = new float[height];
				for(int j = 0; j < height; j++)
					M[i][j] = Init;
			}
		}
		
		
		
		public static Matrix operator/ (Matrix A, float B){
			float[][] newMatrix = new float[A.length][];
			
			for(int i = 0; i < A.length; i++){
				newMatrix[i] = new float[A.height];
				for(int j = 0; j < A.height; j++){
					newMatrix[i][j] = A[i,j]/B;	
				}
			}
			return new Matrix(newMatrix);
		}
		
		public static Matrix operator* (Matrix A, Matrix B){
			
			float[][] newMatrix = new float[A.length][];
			
			for(int i = 0; i < A.length; i++){
				newMatrix[i] = new float[B.height];
				for(int j = 0; j < B.height; j++){
					for(int k = 0; k < A.height; k++){
						newMatrix[i][j] += A[i, k] * B[k, j];
					}
				}
			}
			return new Matrix(newMatrix);
		}
		
		public static Matrix operator* (Matrix A, float B){		
			float[][] newMatrix = new float[A.length][];
			
			for(int i = 0; i < A.length; i++){
				newMatrix[i] = new float[A.height];
				for(int j = 0; j < A.height; j++){
					newMatrix[i][j] = A[i,j]*B;	
				}
			}
			return new Matrix(newMatrix);
		}
		
		
		public static Matrix operator+ (Matrix A, Matrix B){
			
			float[][] newMatrix = new float[A.length][];
			
			for(int i = 0; i < A.length; i++){
				newMatrix[i] = new float[A.height];
				for(int j = 0; j < A.height; j++){
					newMatrix[i][j] = A[i,j] + B[i,j];	
				}
			}
			return new Matrix(newMatrix);
		}
		
		public static Matrix operator- (Matrix A, Matrix B){
			
			float[][] newMatrix = new float[A.length][];
			
			for(int i = 0; i < A.length; i++){
				newMatrix[i] = new float[A.height];
				for(int j = 0; j < A.height; j++){
					newMatrix[i][j] = A[i,j] - B[i,j];	
				}
			}
			return new Matrix(newMatrix);
		}
		
	}


	




}