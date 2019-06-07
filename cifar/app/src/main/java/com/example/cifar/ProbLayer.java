package com.example.cifar;

import android.util.Log;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;

import java.util.HashMap;
import java.util.Map;


public class ProbLayer extends SameDiffOutputLayer {
    private int in_ch;
    private int out_ch;
    private int height;
    private int width;
    private Map<String, SDVariable> paramTable;
    protected WeightInit weightInit;

    public ProbLayer(int in_ch, int out_ch, int height, int width, WeightInit weightInit) {
        this.in_ch = in_ch;
        this.out_ch = out_ch;
        this.weightInit = weightInit;
        this.height = height;
        this.width = width;
    }

    /**
     * Define the output layer
     * @param sd   SameDiff instance
     * @param layerInput Input to the layer
     * @param labels     Labels variable (or null if {@link #labelsRequired()} returns false
     * @param paramTable Parameter table - keys as defined by {@link #defineParameters(SDLayerParams)}
     * @return The final layer variable corresponding to the score/loss during forward pass. This must be a single scalar value.
     */
    @Override
    public SDVariable defineLayer(SameDiff sd, SDVariable layerInput, SDVariable labels,
                                  Map<String, SDVariable> paramTable) {
        // parameters
        this.paramTable = paramTable;
        SDVariable denseWeight = paramTable.get("denseWeight");
        SDVariable denseBias = paramTable.get("denseBias");
        sd.var(denseWeight);
        sd.var(denseBias);
        SDVariable outputRelu = sd.nn().relu("outputRelu", layerInput, 0.);
        Pooling2DConfig c = Pooling2DConfig.builder()
                .kH(this.height).kW(this.width)
                .pH(0).pW(0)
                .isSameMode(false)
                .sH(1).sW(1)
                .build();
        SDVariable outputPool = sd.cnn().avgPooling2d("outputPool", outputRelu, c);
        SDVariable outputSqueeze = sd.squeeze("squeeze", outputPool, 2);
        SDVariable outputReshape = sd.squeeze("reshape", outputSqueeze, 2);
        SDVariable outputDense = sd.nn().linear("outputDense", outputReshape, denseWeight, denseBias);
        SDVariable loss = sd.loss().softmaxCrossEntropy("loss", labels, outputDense);
        return loss;
    }

    /**
     * Output layers should terminate in a single scalar value (i.e., a score) - however, sometimes the output activations
     * (such as softmax probabilities) need to be returned. When this is the case, we need to know the name of the
     * SDVariable that corresponds to these.<br>
     * If the final network activations are just the input to the layer, simply return "input"
     *
     * @return The name of the activations to return when performing forward pass
     */
    @Override
    public String activationsVertexName(){
        return "loss";
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
        initWeights(in_ch, out_ch, weightInit, params.get("denseWeight"));
        initWeights(1, out_ch, weightInit, params.get("denseBias"));
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
        return InputType.feedForward(this.out_ch);
    }

    /**
     * Gradients without referring to the stored activation.
     * @param x [N, Cin/2, H, W]. Input activation.
     */
    public INDArray[] gradient(INDArray x, INDArray label) {
        SameDiff sd = SameDiff.create();
        SDVariable layerInput = sd.var("input", x);
        SDVariable labelInput = sd.constant("label", label);
        layerInput.isPlaceHolder();
        SDVariable loss = defineLayer(sd, layerInput, labelInput, this.paramTable);
        String[] w_names = new String[]{"input", "denseWeight", "denseBias"};
        Map<String, INDArray> placeHolders = new HashMap();
        placeHolders.put("input", x);
        sd.execBackwards(placeHolders);
        INDArray[] grads = new INDArray[w_names.length];
        for (int i = 0; i < w_names.length; i++) {
            grads[i] = sd.getGradForVariable(w_names[i]).getArr();
            Log.d("grad", grads[i].toString());
        }
        return grads;
    }
}
