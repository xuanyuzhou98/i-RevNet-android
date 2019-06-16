package com.example.cifar;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;


public class Bottleneck extends SameDiffLayer {
    private int stride;
    private int in_ch;
    private int out_ch;
    private int mult;
    private int filterSize = 3;
    private boolean first;
    private Map<String, SDVariable> paramTable;
    private Map<String, INDArray> params;

    public Bottleneck(int in_ch, int out_ch, int stride,
                      int mult, WeightInit weightInit, boolean first) {
        this.in_ch = in_ch;
        this.out_ch = out_ch;
        this.stride = stride;
        this.mult = mult;
        this.weightInit = weightInit;
        this.first = first;
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
        this.paramTable = paramTable;

        SDVariable conv1Weight = paramTable.get("conv1Weight");
        SDVariable conv2Weight = paramTable.get("conv2Weight");
        SDVariable conv3Weight = paramTable.get("conv3Weight");
//        SDVariable mean1 = paramTable.get("mean1");
//        SDVariable var1 = paramTable.get("var1");
//        SDVariable gamma1 = paramTable.get("gamma1");
//        SDVariable beta1 = paramTable.get("beta1");
//        SDVariable mean2 = paramTable.get("mean2");
//        SDVariable var2 = paramTable.get("var2");
//        SDVariable gamma2 = paramTable.get("gamma2");
//        SDVariable beta2 = paramTable.get("beta2");
//        SDVariable mean3 = paramTable.get("mean3");
//        SDVariable var3 = paramTable.get("var3");
//        SDVariable gamma3 = paramTable.get("gamma3");
//        SDVariable beta3 = paramTable.get("beta3");
        sd.var(conv1Weight);
        sd.var(conv2Weight);
        sd.var(conv3Weight);
//        sd.var(mean1);
//        sd.var(var1);
//        sd.var(gamma1);
//        sd.var(beta1);
//        sd.var(mean2);
//        sd.var(var2);
//        sd.var(gamma2);
//        sd.var(beta2);
//        sd.var(mean3);
//        sd.var(var3);
//        sd.var(gamma3);
//        sd.var(beta3);
        if (!this.first) {
            layerInput = sd.nn().relu("act0", layerInput, 0.);
        }
        Conv2DConfig c1 = Conv2DConfig.builder()
                .kH(this.filterSize).kW(this.filterSize)
                .pH(1).pW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .sH(this.stride).sW(this.stride)
                .build();
        SDVariable conv1 = sd.cnn().conv2d("conv1", new SDVariable[]{layerInput, conv1Weight}, c1);
//        SDVariable bn1 = sd.nn().batchNorm("bn1", conv1, mean1, var1, gamma1, beta1, 1e-5, 1);
        SDVariable act1 = sd.nn().relu("act1", conv1, 0.);
        Conv2DConfig c2 = Conv2DConfig.builder()
                .kH(this.filterSize).kW(this.filterSize)
                .pH(1).pW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .sH(1).sW(1)
                .build();
        SDVariable conv2 = sd.cnn().conv2d("conv2", new SDVariable[]{act1, conv2Weight}, c2);
//        SDVariable bn2 = sd.nn().batchNorm("bn2", conv2, mean2, var2, gamma2, beta2, 1e-5, 1);
        SDVariable act2 = sd.nn().relu("act2", conv2, 0.);
        Conv2DConfig c3 = Conv2DConfig.builder()
                .kH(this.filterSize).kW(this.filterSize)
                .pH(1).pW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .sH(1).sW(1)
                .build();
        SDVariable conv3 = sd.cnn().conv2d("conv3", new SDVariable[]{act2, conv3Weight}, c3);
        return conv3;
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
        int fanIn1 = in_ch * filterSize * filterSize;    // fan-in = num input feature maps * filter height * filter width
        int fanOut1 = (out_ch / mult) * filterSize * filterSize;
        initWeights(fanIn1, fanOut1, WeightInit.XAVIER_UNIFORM, params.get("conv1Weight"));
        int fanIn2 = (out_ch/mult) * filterSize * filterSize;
        int fanOut2 = (out_ch/mult) * filterSize * filterSize;
        initWeights(fanIn2, fanOut2, WeightInit.XAVIER_UNIFORM, params.get("conv2Weight"));
        int fanIn3 = (out_ch/mult) * filterSize * filterSize;
        int fanOut3 = out_ch * filterSize * filterSize;
        initWeights(fanIn3, fanOut3, WeightInit.XAVIER_UNIFORM, params.get("conv3Weight"));
//        initWeights(out_ch/mult, out_ch/mult, weightInit, params.get("mean1"));
//        initWeights(out_ch/mult, out_ch/mult, weightInit, params.get("var1"));
//        initWeights(out_ch/mult, out_ch/mult, WeightInit.ONES, params.get("gamma1"));
//        initWeights(out_ch/mult, out_ch/mult, WeightInit.ZERO, params.get("beta1"));
//        initWeights(out_ch/mult, out_ch/mult, weightInit, params.get("mean2"));
//        initWeights(out_ch/mult, out_ch/mult, weightInit, params.get("var2"));
//        initWeights(out_ch/mult, out_ch/mult, WeightInit.ONES, params.get("gamma2"));
//        initWeights(out_ch/mult, out_ch/mult, WeightInit.ZERO, params.get("beta2"));
//        initWeights(out_ch/mult, out_ch/mult, weightInit, params.get("mean3"));
//        initWeights(out_ch/mult, out_ch/mult, weightInit, params.get("var3"));
//        initWeights(out_ch/mult, out_ch/mult, WeightInit.ONES, params.get("gamma3"));
//        initWeights(out_ch/mult, out_ch/mult, WeightInit.ZERO, params.get("beta3"));
        this.params = params;
    }


    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam("conv1Weight", filterSize, filterSize, in_ch, out_ch/mult);
        params.addWeightParam("conv2Weight", filterSize, filterSize, out_ch/mult, out_ch/mult);
        params.addWeightParam("conv3Weight", filterSize, filterSize, out_ch/mult, out_ch);
//        params.addWeightParam("mean1", out_ch/mult);
//        params.addWeightParam("var1", out_ch/mult);
//        params.addWeightParam("gamma1", out_ch/mult);
//        params.addBiasParam("beta1", out_ch/mult);
//        params.addWeightParam("mean2", out_ch/mult);
//        params.addWeightParam("var2", out_ch/mult);
//        params.addWeightParam("gamma2", out_ch/mult);
//        params.addBiasParam("beta2", out_ch/mult);
//        params.addWeightParam("mean3", out_ch/mult);
//        params.addWeightParam("var3", out_ch/mult);
//        params.addWeightParam("gamma3", out_ch/mult);
//        params.addBiasParam("beta3", out_ch/mult);
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        //In this method: you define the type of output/activations, if appropriate, given the type of input to the layer
        //This is used in a few methods in DL4J to calculate activation shapes, memory requirements etc
        int[] hwd = ConvolutionUtils.getHWDFromInputType(inputType);
        int outH = (hwd[0] - 3 + 2 * 1) / this.stride + 1;
        int outW = (hwd[1] - 3 + 2 * 1) / this.stride + 1;
        outH = (outH - 3 + 2 * 1) + 1;
        outW = (outW - 3 + 2 * 1) + 1;
        outH = (outH - 3 + 2 * 1) + 1;
        outW = (outW - 3 + 2 * 1) + 1;
        return InputType.convolutional(outH, outW, this.out_ch);
    }

    /**
     * Gradients without referring to the stored activation.
     * @param x [N, Cin/2, H, W]. Input activation.
     */
    public INDArray forward(INDArray x) {
        SameDiff sd = SameDiff.create();
        SDVariable layerInput = sd.var("input", x);
        layerInput.isPlaceHolder();
        defineLayer(sd, layerInput, this.paramTable, null);
        Map<String, INDArray> placeHolders = new HashMap();
        placeHolders.put("input", x);
        INDArray btnkOut = sd.execSingle(placeHolders, "conv3");
        return btnkOut;
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
        SDVariable outputGrad = sd.var("outputGrad", dy);
        outputGrad.isPlaceHolder();
        SDVariable output = defineLayer(sd, layerInput, this.paramTable, null);
        SDVariable mul = output.mul(outputGrad);
        String[] w_names = new String[]{"input", "conv1Weight", "conv2Weight", "conv3Weight"};
        Map<String, INDArray> placeHolders = new HashMap();
        placeHolders.put("input", x);
        placeHolders.put("outputGrad", dy);
        sd.execBackwards(placeHolders);
        INDArray[] grads = new INDArray[w_names.length];
        for (int i = 0; i < w_names.length; i++) {
            grads[i] = sd.getGradForVariable(w_names[i]).getArr();
        }
        return grads;
    }
}
