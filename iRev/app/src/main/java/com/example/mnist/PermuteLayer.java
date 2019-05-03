package com.example.mnist;

import android.util.Log;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;

import java.util.HashMap;
import java.util.Map;


public class PermuteLayer extends SameDiffLayer {
    private int axis1;
    private int axis2;
    private int axis3;


    public PermuteLayer(int axis0, int axis1,  int axis2, int axis3) {
        this.axis1 = axis1;
        this.axis2 = axis2;
        this.axis3 = axis3;
    }

    /**
     * In the defineLayer method, you define the actual layer forward pass
     * For this layer, we are returning out = activationFn( input*weights + bias)
     *
     * @param sd         The SameDiff instance for this layer
     * @param layerInput A SDVariable representing the input activations for the layer
     * @param paramTable A map of parameters for the layer. These are the SDVariables corresponding to whatever you defined
     *                   in the defineParameters method
     * @return
     */
    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput, Map<String, SDVariable> paramTable, SDVariable mask) {
        // parameters
       SDVariable output = layerInput.permute(0, this.axis1, this.axis2, this.axis3);
       return output;
    }

    /**
     * This method is used to initialize the parameter.
     * For example, we are setting the bias parameter to 0, and using the specified DL4J weight initialization type
     * for the weights
     * @param params Map of parameters. These are the INDArrays corresponding to whatever you defined in the
     *               defineParameters method
     */
    @Override
    public void initializeParameters(Map<String, INDArray> params) {}


    @Override
    public void defineParameters(SDLayerParams params) {}

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        //In this method: you define the type of output/activations, if appropriate, given the type of input to the layer
        //This is used in a few methods in DL4J to calculate activation shapes, memory requirements etc
        int[] hwd = ConvolutionUtils.getHWDFromInputType(inputType);
        int[] dhw = new int[]{hwd[2], hwd[0], hwd[1]};
        long[] outputShape = new long[3];
        outputShape[0] = dhw[axis1 - 1];
        outputShape[1] = dhw[axis2 - 1];
        outputShape[2] = dhw[axis3 - 1];
        return InputType.convolutional(outputShape[1], outputShape[2], outputShape[0]);
    }
}
