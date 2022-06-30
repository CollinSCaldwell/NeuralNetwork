using System;
using System.IO;
using System.Collections.Generic;

namespace Templates{
	public abstract class LearningRate{
		public abstract float CalcRate(float fixedRate, int counter);
	}
	
	public abstract class Error{
		public abstract float[] ErrorDeriv(float[] input, float[] desired);
		public abstract float ErrorNumber(float[] input, float[] desired);
	}
	
	
	public abstract class Activation{
		public abstract string type {get; set;}
		
		public abstract float ActiveFunction(float Input);
		
		public abstract float ActiveFunctionDeriv(float Input);
		
	}
}