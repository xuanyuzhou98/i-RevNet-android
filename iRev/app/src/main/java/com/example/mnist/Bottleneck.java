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


public class Bottleneck extends SameDiffLayer {
    private int stride;
    private float dpRate;
    private int in_ch;
    private int out_ch;
    private int mult;
    private boolean first;
    private Map<String, SDVariable> paramTable;

    public Bottleneck(int in_ch, int out_ch, int stride, boolean first, float dropout_rate,
                      int mult, WeightInit weightInit) {
        this.in_ch = in_ch;
        this.out_ch = out_ch;
        this.stride = stride;
        this.first = first;
        this.dpRate = dropout_rate;
        this.mult = mult;
        this.weightInit = weightInit;
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
        this.paramTable = paramTable;
        SDVariable conv1Weight = paramTable.get("conv1Weight");
//        SDVariable conv2Weight = paramTable.get("conv1Weight");
////        SDVariable conv3Weight = paramTable.get("conv1Weight");
        sd.var(conv1Weight);
//        sdTest.var(conv2Weight);
//        sdTest.var(conv3Weight);
        //32 32 3
        Conv2DConfig c1 = Conv2DConfig.builder()
                .kH(2).kW(2)
                .pH(0).pW(0)
                .dH(1).dW(1)
                .isSameMode(false)
                .sH(this.stride).sW(this.stride)
                .build();
        SDVariable conv1 = sd.cnn().conv2d("conv1", new SDVariable[]{layerInput, conv1Weight}, c1);
        SDVariable bn1 = sd.nn().batchNorm("bn1", conv1, sd.mean(conv1, 1),
                                      sd.variance(conv1, false, 1), sd.scalar("conv1Gamma", 1.),
                                      sd.scalar("conv1Beta", 0.), 1e-5, 1);
        SDVariable act1 = sd.nn().relu(bn1, 0.);
//        Conv2DConfig c2 = Conv2DConfig.builder()
//                .kH(3).kW(3)
//                .pH(1).pW(1)
//                .dH(1).dW(1)
//                .isSameMode(false)
//                .dataFormat("NCHW")
//                .sH(1).sW(1)
//                .build();
//        SDVariable conv2 = sd.conv2d("conv2", new SDVariable[]{act1, conv2Weight}, c2);
//        SDVariable bn2 = sd.batchNorm("bn2", conv2, sd.mean(conv2, 1),
//                sd.variance(conv2, false, 1), sd.scalar("conv2Gamma", 1.),
//                sd.scalar("conv2Beta", 0.), 1e-5, 1);
//        SDVariable act2 = sd.relu(bn2, 0.);
//        Conv2DConfig c3 = Conv2DConfig.builder()
//                .kH(3).kW(3)
//                .pH(1).pW(1)
//                .dH(1).dW(1)
//                .isSameMode(false)
//                .dataFormat("NCHW")
//                .sH(1).sW(1)
//                .build();
//        SDVariable conv3 = sd.conv2d("output", new SDVariable[]{act2, conv3Weight}, c3);
        return act1;
    }

    /**
     * This method is used to initialize the parameter.
     * For example, we are setting the bias parameter to 0, and using the specified DL4J weight initialization type
     * for the weights
     * @param params Map of parameters. These are the INDArrays corresponding to whatever you defined in the
     *               defineParameters method
     */
    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        initWeights(in_ch, out_ch, weightInit, params.get("conv1Weight"));
//        initWeights(out_ch/mult, out_ch/mult, weightInit, params.get("conv2Weight"));
//        initWeights(out_ch/mult, out_ch, weightInit, params.get("conv3Weight"));
    }


    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam("conv1Weight", 2, 2, in_ch, out_ch);
//        params.addWeightParam("conv2Weight", 3, 3, out_ch/mult, out_ch/mult);
//        params.addWeightParam("conv3Weight", 3, 3, out_ch/mult, out_ch);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        //In this method: you define the type of output/activations, if appropriate, given the type of input to the layer
        //This is used in a few methods in DL4J to calculate activation shapes, memory requirements etc
        int[] hwd = ConvolutionUtils.getHWDFromInputType(inputType);
        Log.d("input shape", hwd[0] + " " + hwd[1]);
        int outH = (hwd[0] - 2) / this.stride + 1;
        int outW = (hwd[1] - 2) / this.stride + 1;
        return InputType.convolutional(outH, outW, this.out_ch);
    }

    /**
     * Gradients without referring to the stored activation.
     * @param x [N, Cin/2, H, W]. Input activation.
     * @param dy [N, Cout/2, H, W]. Output gradient.
     */
    public INDArray[] gradient(INDArray x, INDArray dy) {
        SameDiff sd = SameDiff.create();
        SDVariable layerInput = sd.var("input", x);
        layerInput.isPlaceHolder();
        SDVariable output = defineLayer(sd, layerInput, this.paramTable, null);
        String[] w_names = new String[]{"conv1Weight", "input"};
        Map<String, INDArray> placeHolders = new HashMap();
        placeHolders.put("input", x);
        sd.execBackwards(placeHolders);
        INDArray[] grads = new INDArray[w_names.length];
        for (int i = 0; i < w_names.length; i++) {
            grads[i] = sd.getGradForVariable(w_names[i]).getArr();
        }
        return grads;
    }
}
