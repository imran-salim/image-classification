package weka.classifiers.functions;

import weka.classifiers.RandomizableClassifier;
import weka.core.*;
import java.util.Random;

/**
 * A class implementing multiple linear regression, i.e., linear
 * regression based on several predictor attributes (aka independent
 * variables). Uses basic gradient descent with a user-specified
 * learning rate. Monitors the squared error: if the gradient descent
 * update increases the error, which means the search has jumped
 * too far in parameter space, the learning rate will be halved.
 *
 * We extend the RandomizableClassifier class from WEKA, which in turn
 * extends WEKA's AbstractClassifier class and adds a method for setting
 * and storing the seed of the random number generator used to initialise
 * the weight vector in this class.
 */
public class SteepestDescentLinearRegression extends RandomizableClassifier {

  /** An array for holding the weight vector. We will store the "bias"
      weight at the position of the class attribute. */
  private double[] m_w;

  /** The learning rate: a parameter that can be set by the user. */
  private double m_learningRate = 1.0;

  /** The threshold used to decide when to stop gradient descent. */
  private double THRESHOLD = 1.0e-6;

  /** Header information from the training data, to reference attributes for output. */
  private Instances m_header;
  
  /**
   * The method for training the predictive model: takes a training
   * dataset and finds the weight vector that minimises the squared
   * error using gradient descent.
   *
   * @param data the training data
   */
  public void buildClassifier(Instances data) throws Exception {

    // Can this algorithm handle the data based on its capabilities (see below)?
    getCapabilities().testWithFail(data);
    
    // Use the getSeed() method inherited from RandomizableClassifier
    // to initialise the random number generator.
    Random r = new Random(getSeed()); 

    // Make a shallow copy of the data, remove instances without
    // a class value, and shuffle the data.
    data = new Instances(data);
    data.deleteWithMissingClass();
    data.randomize(r);
    
    // Create an array to store the weight vector and initialise it
    // using normal random variates.
    m_w = new double[data.numAttributes()];
    for (int j = 0; j < data.numAttributes(); j++) {
      m_w[j] = r.nextGaussian();
    }

    // Set aside space to back up weight vector in case we need to change the learning rate.
    double[] backup = new double[m_w.length];
    System.arraycopy(m_w, 0, backup, 0, m_w.length);
      
    // Set initial sum of squared errors to maximum possible double.
    double oldSSE = Double.MAX_VALUE;

    // We want to locally adapt the learning rate if necessary.
    double learningRate = m_learningRate;
    
    // Perform passes through the training data, i.e., gradient descent
    // update steps, until a stopping condition is met.
    do {

      // Set aside space for storing the gradient vector.
      double[] gradient = new double[data.numAttributes()];

      // We need to keep track of the squared error.
      double newSSE = 0;
      
      // Calculate the gradient by summing up the contribution
      // to the gradient vector from each training instance.
      for (Instance inst : data) {

        // Calculate the difference between actual and predicted
        // target value for the training instance.
        double err = inst.classValue() - classifyInstance(inst);

        // Update sum of squared errors.
        newSSE += err * err;
        
        // Go through the gradient vector and update it by
        // adding the appropriate partial derivative.
        for (int j = 0; j < data.numAttributes(); j++) {
          if (j != inst.classIndex()) {
            gradient[j] += -2 * err * inst.value(j);
          } else {
            gradient[j] += -2 * err * 1; // Gradient for the "bias" weight.
          }
        }
      }

      // Output the current SSE if the user has set the -output-debug-info flag
      // (see AbstractClassifier).
      if (getDebug()) {
        System.err.println("Current SSE: " + newSSE);
      }
      
      // Has the sum of squared errors increased based on the previous update?
      if (newSSE > oldSSE) {
        learningRate /= 2.0; // Half the learning rate.
        System.arraycopy(backup, 0, m_w, 0, m_w.length);; // Restore the old weight vector.
        continue; // Go back to start of the while loop.
      }
      oldSSE = newSSE; // Store the current SSE
      System.arraycopy(m_w, 0, backup, 0, m_w.length); // Back up the weight vector.

      // Calculate the length of the gradient vector and stop
      // performing gradient descent when the update becomes too small.
      double sumSquared = 0;
      for (int j = 0; j < data.numAttributes(); j++ ){
        sumSquared += gradient[j] * gradient[j];
      }
      if (learningRate * Math.sqrt(sumSquared) < THRESHOLD) {
        break;
      }
      
      // We have the gradient: do a gradient descent step now to
      // update the weight vector so that we move closer to the minimum
      // of the squared error.
      for (int j = 0; j < data.numAttributes(); j++ ){
        m_w[j] -= learningRate * gradient[j];
      }
    } while (true);

    // Store the header info from the training data (i.e., attribute info, etc.).
    m_header = new Instances(data, 0);
  }

  /**
   * The method for getting a prediction from the model for a given instance.
   *
   * @param inst the instance for which we want a prediction of the target value
   *
   * @return the predicted target value.
   */
  public double classifyInstance(Instance inst) {

    double sum = 0;

    // The prediction based on the current model is just a weighted sum.
    for (int i = 0; i < inst.numAttributes(); i++) {
      if (i != inst.classIndex()) {
        sum += m_w[i] * inst.value(i);
      } else {
        sum += m_w[i];
      }
    }
    
    return sum;
  }

  /**
   * A standard method that returns a textual description of the model.
   *
   * @return an (ugly) string describing the contents of the model
   */
  public String toString() {

    if (m_w == null) {
      return "SteepestDescentLinearRegression: no classifier built yet!";
    } else {
      StringBuffer sb = new StringBuffer();
      for (int i = 0; i < m_header.numAttributes(); i++) {
        if (i != m_header.classIndex()) {
          sb.append("+" + Utils.doubleToString(m_w[i], getNumDecimalPlaces()) + "*" +
                    m_header.attribute(i).name() + "\n");
        }
      }
      sb.append("+" + Utils.doubleToString(m_w[m_header.classIndex()], getNumDecimalPlaces()));
      return sb.toString();
    }
  }

  /**
   * This WEKA-specific code is to implement the command-line and GUI option handling for the 
   * learning rate parameter.
   */
  @OptionMetadata(displayName = "learning rate",
                  description = "The initial learning rate to use for gradient descent.",
                  displayOrder = 1,
                  commandLineParamName = "L",
                  commandLineParamSynopsis = "-L <double>")
  public double getLearningRate() { return m_learningRate; }
  public void setLearningRate(double l) {m_learningRate = l; }
  
  
  /**
   * The capabilities of the classifier: all attributes must be
   * numeric, no missing predictor attribute values are permitted.
   */
  public Capabilities getCapabilities() {

    Capabilities result = super.getCapabilities();
    result.disableAll();
    result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capabilities.Capability.NUMERIC_CLASS);
    result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

    return result;
  }
  
  /**
   * The main method for running this class from the command-line, making
   * use of the runClassifier() method from AbstractClassifier to 
   * provide all standard WEKA options for running learning algorithms.
   */
  public static void main(String[] args) {
    
    runClassifier(new SteepestDescentLinearRegression(), args);
  }
}

