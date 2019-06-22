package com.example.cifar;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.Collection;
import java.util.Map;
import org.nd4j.linalg.api.buffer.DataType;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.layers.NoParamLayer;


public class PsiLayer extends NoParamLayer {
    private int blockSize;
    private int blockSizeSq;

    public PsiLayer(Builder builder) {
        super(builder);
        this.blockSize = builder.blockSize;
        this.blockSizeSq = this.blockSize * this.blockSize;
    }

    public int getBlockSize() {
        //We also need setter/getter methods for our layer configuration fields (if any) for JSON serialization
        return this.blockSize;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
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
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        int[] hwd = ConvolutionUtils.getHWDFromInputType(inputType);
        int outH = hwd[0] / this.blockSize;
        int outW = hwd[1] / this.blockSize;
        return InputType.convolutional(outH, outW, hwd[2] * this.blockSizeSq);
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //Memory report is used to estimate how much memory is required for the layer, for different configurations
        //If you don't need this functionality for your custom layer, you can return a LayerMemoryReport
        // with all 0s, or

        //This implementation: based on DenseLayer implementation
        InputType outputType = getOutputType(-1, inputType);

        return new LayerMemoryReport.Builder(layerName, PsiLayer.class, inputType, outputType)
                .standardMemory(0, 0) //No params
                //Inference and training is same - just output activations, no working memory in addition to that
                .workingMemory(0, 0, MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS)
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();

    }

    //Here's an implementation of a builder pattern, to allow us to easily configure the layer
    //Note that we are inheriting all of the FeedForwardLayer.Builder options: things like n
    public static class Builder extends Layer.Builder<Builder> {

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
