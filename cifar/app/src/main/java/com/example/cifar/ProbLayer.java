package com.example.cifar;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Map;


public class ProbLayer extends SameDiffLayer {
    private int in_ch;
    private int out_ch;
    private Map<String, SDVariable> paramTable;
    protected static final Logger log = LoggerFactory.getLogger(ProbLayer.class);

    public ProbLayer(int in_ch, int out_ch) {
        this.in_ch = in_ch;
        this.out_ch = out_ch;
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
        SDVariable denseWeight = sd.var(paramTable.get("denseWeight"));
        SDVariable denseBias = sd.var(paramTable.get("denseBias"));
        SDVariable outputRelu = sd.nn().relu("outputRelu", layerInput, 0.);
        SDVariable outputPool = outputRelu.mean("outputPool", false, 2, 3);
        SDVariable mmul = sd.mmul("mmul", outputPool, denseWeight);
        SDVariable outputDense = mmul.add("outputDense", denseBias);
        return outputDense;
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
        initWeights(in_ch, out_ch, WeightInit.XAVIER, params.get("denseWeight"));
        initWeights(in_ch, out_ch, WeightInit.UNIFORM, params.get("denseBias"));

    }


    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam("denseWeight", in_ch, out_ch);
        params.addBiasParam("denseBias", 1, out_ch);

    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        //In this method: you define the type of output/activations, if appropriate, given the type of input to the layer
        //This is used in a few methods in DL4J to calculate activation shapes, memory requirements etc
        return InputType.feedForward(out_ch);
    }

    /**
     * Gradients without referring to the stored activation.
     * @param x [N, Cin/2, H, W]. Input activation.
     */
    public INDArray[] gradient(INDArray x, INDArray label) {
        SameDiff sd = SameDiff.create();
        SDVariable layerInput = sd.var("input", x);
        SDVariable labelInput = sd.constant("label", label);
        SDVariable outputDense = defineLayer(sd, layerInput, paramTable, null);
        SDVariable loss = sd.loss().softmaxCrossEntropy("loss", labelInput, outputDense, LossReduce.NONE);
        sd.execBackwards(Collections.EMPTY_MAP);
        log.info("loss: " + loss.eval().mean().toString());
        String[] w_names = new String[]{"input", "denseWeight", "denseBias"};
        INDArray[] grads = new INDArray[w_names.length];
        for (int i = 0; i < w_names.length; i++) {
            grads[i] = sd.grad(w_names[i]).getArr();
        }
        return grads;
    }
}
