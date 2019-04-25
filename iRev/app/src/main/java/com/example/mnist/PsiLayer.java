package com.example.mnist;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.Collection;
import java.util.Map;
import org.nd4j.linalg.api.buffer.DataType;


public class PsiLayer extends ConvolutionLayer{
    private int blockSize;

    public PsiLayer(Builder builder) {
        super(builder);
        this.blockSize = builder.blockSize;
    }

    public int getBlockSize() {
        //We also need setter/getter methods for our layer configuration fields (if any) for JSON serialization
        return this.blockSize;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        //The instantiate method is how we go from the configuration class (i.e., this class) to the implementation class
        // (i.e., a CustomLayerImpl instance)
        //For the most part, it's the same for each type of layer

        PsiLayerImpl irevlayerimpl = new PsiLayerImpl(conf, networkDataType);

        irevlayerimpl.setListeners(trainingListeners);             //Set the iteration listeners, if any
        irevlayerimpl.setIndex(layerIndex);                         //Integer index of the layer

        //Parameter view array: In Deeplearning4j, the network parameters for the entire network (all layers) are
        // allocated in one big array. The relevant section of this parameter vector is extracted out for each layer,
        // (i.e., it's a "view" array in that it's a subset of a larger array)
        // This is a row vector, with length equal to the number of parameters in the layer
        irevlayerimpl.setParamsViewArray(layerParamsView);

        //Initialize the layer parameters. For example,
        // Note that the entries in paramTable (2 entries here: a weight array of shape [nIn,nOut] and biases of shape [1,nOut]
        // are in turn a view of the 'layerParamsView' array.
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        irevlayerimpl.setParamTable(paramTable);
        irevlayerimpl.setConf(conf);
        return irevlayerimpl;
    }

    @Override
    public ParamInitializer initializer() {
        //This method returns the parameter initializer for this type of layer
        //In this case, we can use the DefaultParamInitializer, which is the same one used for DenseLayer
        //For more complex layers, you may need to implement a custom parameter initializer
        //See the various parameter initializers here:
        //https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j-core/src/main/java/org/deeplearning4j/nn/params

        return DefaultParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null || inputType.getType() != InputType.Type.CNN) {
            throw new IllegalStateException("Invalid input for Convolution layer (layer name=\"" + getLayerName()
                    + "\"): Expected CNN input, got " + inputType);
        }

        InputType.InputTypeConvolutional outputType = (InputType.InputTypeConvolutional)InputTypeUtil.getOutputTypeCnnLayers(inputType, kernelSize, stride, padding, dilation,
                convolutionMode, nOut, layerIndex, getLayerName(), ConvolutionLayer.class);
        outputType.setHeight(inputType.getShape()[1] / blockSize);
        outputType.setWidth(inputType.getShape()[2] / blockSize);
        return outputType;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //Memory report is used to estimate how much memory is required for the layer, for different configurations
        //If you don't need this functionality for your custom layer, you can return a LayerMemoryReport
        // with all 0s, or

        //This implementation: based on DenseLayer implementation
        InputType outputType = getOutputType(-1, inputType);

        long numParams = initializer().numParams(this);
        int updaterStateSize = (int)getIUpdater().stateSize(numParams);

        int trainSizeFixed = 0;
        int trainSizeVariable = 0;
        if(getIDropout() != null){
            //Assume we dup the input for dropout
            trainSizeVariable += inputType.arrayElementsPerExample();
        }

        //Also, during backprop: we do a preOut call -> gives us activations size equal to the output size
        // which is modified in-place by activation function backprop
        // then we have 'epsilonNext' which is equivalent to input size
        trainSizeVariable += outputType.arrayElementsPerExample();

        return new LayerMemoryReport.Builder(layerName, PsiLayer.class, inputType, outputType)
                .standardMemory(numParams, updaterStateSize)
                .workingMemory(0, 0, trainSizeFixed, trainSizeVariable)     //No additional memory (beyond activations) for inference
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching in DenseLayer
                .build();
    }

    //Here's an implementation of a builder pattern, to allow us to easily configure the layer
    //Note that we are inheriting all of the FeedForwardLayer.Builder options: things like n
    public static class Builder extends ConvolutionLayer.Builder {

        private int blockSize;

        //This is an example of a custom property in the configuration

        /**
         * A custom property used in this custom layer example. See the CustomLayerExampleReadme.md for details
         *
         * @param secondActivationFunction Second activation function for the layer
         */

        /**
         * A custom property used in this custom layer example. See the CustomLayerExampleReadme.md for details
         *
         */
        public Builder BlockSize(int blockSize){
            this.blockSize = blockSize;
            return this;
        }


        @Override
        @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
        public PsiLayer build() {
            return new PsiLayer(this);
        }
    }




}
