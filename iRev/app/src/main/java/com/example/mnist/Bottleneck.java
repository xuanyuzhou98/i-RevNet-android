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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ops.impl.layers.convolution.BatchNorm;

import java.util.HashMap;
import java.util.Map;


public class Bottleneck extends SameDiffLayer {
    private int batchsize;
    private int inputH;
    private int inputW;
    private int outputH;
    private int outputW;
    private int stride;
    private int in_ch;
    private int out_ch;
    private int mult;
    private Map<String, SDVariable> paramTable;

    public Bottleneck(int batchsize, int inputH, int inputW, int in_ch, int out_ch, int stride,
                      int mult, WeightInit weightInit) {
        this.batchsize = batchsize;
        this.inputH = inputH;
        this.inputW = inputW;
        this.in_ch = in_ch;
        this.out_ch = out_ch;
        this.stride = stride;
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
        // parameters
        this.paramTable = paramTable;
        int filterSize = 3;
        SDVariable conv1Weight = paramTable.get("conv1Weight");
        SDVariable conv2Weight = paramTable.get("conv2Weight");
        SDVariable conv3Weight = paramTable.get("conv3Weight");
        sd.var(conv1Weight);
        sd.var(conv2Weight);
        sd.var(conv3Weight);

        Conv2DConfig c1 = Conv2DConfig.builder()
                .kH(filterSize).kW(filterSize)
                .pH(1).pW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .sH(this.stride).sW(this.stride)
                .build();
        SDVariable conv1 = sd.cnn().conv2d("conv1", new SDVariable[]{layerInput, conv1Weight}, c1);

        int[] conv1Shape = Utils.getConvLayerOutShape(inputH, inputW, filterSize, this.stride, 1);
        // TODO shape of gamma and beta should be fixed
        /**
         * Neural network batch normalization operation.<br>
         * For details, see <a href="http://arxiv.org/abs/1502.03167">http://arxiv.org/abs/1502.03167</a>
         *
         * @param name     Name of the output variable
         * @param input    Input variable.
         * @param mean     Mean value. For 1d axis, this should match input.size(axis)
         * @param variance Variance value. For 1d axis, this should match input.size(axis)
         * @param gamma    Gamma value. For 1d axis, this should match input.size(axis)
         * @param beta     Beta value. For 1d axis, this should match input.size(axis)
         * @param epsilon  Epsilon constant for numerical stability (to avoid division by 0)
         * @param axis     For 2d CNN activations: 1 for NCHW format activations, or 3 for NHWC format activations.<br>
         *                 For 3d CNN activations: 1 for NCDHW format, 4 for NDHWC<br>
         *                 For 1d/RNN activations: 1 for NCW format, 2 for NWC
         * @return Output variable for batch normalization
         */
        SDVariable gamma = sd.one("gamma", this.batchsize, out_ch/mult, conv1Shape[0], conv1Shape[1]);  // N=64 C=out_ch/mult H=conv1Height W=conv1Weight
        SDVariable beta = sd.zero("beta", this.batchsize, out_ch/mult, conv1Shape[0], conv1Shape[1]);
        SDVariable bn1 = sd.nn().batchNorm("bn1", conv1, sd.mean(conv1, 0),
                sd.variance(conv1, true, 0), gamma, beta, 1e-6,  new int[]{1});

        SDVariable act1 = sd.nn().relu("act1", bn1, 0.);
        Conv2DConfig c2 = Conv2DConfig.builder()
                .kH(filterSize).kW(filterSize)
                .pH(1).pW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .sH(1).sW(1)
                .build();
        SDVariable conv2 = sd.cnn().conv2d("conv2", new SDVariable[]{act1, conv2Weight}, c2);

        int[] conv2Shape = Utils.getConvLayerOutShape(conv1Shape[0], conv1Shape[1], filterSize, 1, 1);
//        gamma = sd.one("gamma", 64, in_ch, act2Shape[0], act2Shape[1]); // N=64 C=in_ch H=conv1Height W=conv1Weight
//        beta = sd.zero("beta", 64, in_ch, act2Shape[0], act2Shape[1]);
//        SDVariable bn2 = sd.nn().batchNorm("bn1", act2, sd.mean(act1, 0),
//                sd.variance(act1, true, 0), gamma, beta, 1e-6,  new int[]{1});

        SDVariable act2 = sd.nn().relu("act2", conv2, 0.);
        Conv2DConfig c3 = Conv2DConfig.builder()
                .kH(filterSize).kW(filterSize)
                .pH(1).pW(1)
                .dH(1).dW(1)
                .isSameMode(false)
                .sH(1).sW(1)
                .build();
        SDVariable conv3 = sd.cnn().conv2d("conv3", new SDVariable[]{act2, conv3Weight}, c3);

        int[] conv3Shape = Utils.getConvLayerOutShape(conv2Shape[0], conv2Shape[1], filterSize, 1, 1);
        this.outputH = conv3Shape[0];
        this.outputW = conv3Shape[1];

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
        initWeights(in_ch/2, out_ch/mult, weightInit, params.get("conv1Weight"));
        initWeights(out_ch/mult, out_ch/mult, weightInit, params.get("conv2Weight"));
        initWeights(out_ch/mult, out_ch, weightInit, params.get("conv3Weight"));
    }


    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam("conv1Weight", 3, 3, in_ch, out_ch/mult);
        params.addWeightParam("conv2Weight", 3, 3, out_ch/mult, out_ch/mult);
        params.addWeightParam("conv3Weight", 3, 3, out_ch/mult, out_ch);
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

    public int[] getOutputShape() {
        return new int[]{this.outputH, this.outputW};
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
        //Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
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
        //Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        return grads;
    }
}
