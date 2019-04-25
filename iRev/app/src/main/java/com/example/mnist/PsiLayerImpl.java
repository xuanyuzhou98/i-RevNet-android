package com.example.mnist;

import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import android.util.Log;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;



public class PsiLayerImpl extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {
    protected int blockSize;

    public PsiLayerImpl(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
        this.blockSize = ((PsiLayer) conf().getLayer()).getBlockSize();
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        long blockSizeSq = this.blockSize * this.blockSize;
        input = input.permute(0, 2, 3, 1);
        long[] shape = input.shape();
        long batchSize = shape[0];
        long sHeight = shape[1];
        long sWidth = shape[2];
        long sDepth = shape[3];
        Log.d("first", String.valueOf(input.shape()[0]));
        Log.d("second", String.valueOf(input.shape()[1]));
        Log.d("third", String.valueOf(input.shape()[2]));
        Log.d("fourth", String.valueOf(input.shape()[3]));
        long dDepth = sDepth * blockSizeSq; //new depth
        long dHeight = sHeight / this.blockSize; // new height
        int numOfSplits = (int)sWidth / this.blockSize; //
        INDArray[] t_1 = Utils.split(input, numOfSplits, 2);
        INDArray[] stack = new INDArray[numOfSplits];
        for (int i = 0; i < numOfSplits; i += 1) {
            INDArray t_t = t_1[i];
            stack[i] = t_t.reshape(batchSize, dHeight, dDepth);
        }
        INDArray output = Nd4j.stack(1, stack);
        output = output.permute(0, 2, 1, 3);
        output = output.permute(0, 3, 1, 2);
        long[] outShape = output.shape();
        INDArray out = workspaceMgr.create(ArrayType.ACTIVATIONS, this.dataType, outShape);
        out.assign(output);
        return out;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }



    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        long[] origShape = epsilon.shape();
        //Don't collapse dims case: error should be [minibatch, vectorSize, 1] or [minibatch, channels, 1, 1]
        //Reshape it to 2d, to get rid of the 1s

        epsilon = epsilon.reshape(epsilon.ordering(), origShape[0], origShape[1]);
        INDArray epsilonNd;
        epsilonNd = inverse(epsilon, this.blockSize);
        Gradient retGradient = new DefaultGradient(); //Empty: no params

        epsilonNd = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilonNd);
        return new Pair<>(retGradient, epsilonNd);

    }

    public static INDArray inverse(INDArray in, int blockSize) {
        //in: [1, 7, 7, 24]
        //in:[1, 24, 7, 7]
        Log.d("double check x2", in.shapeInfoToString());
        long blockSizeSq = blockSize * blockSize; //4
        INDArray out = in.permute(0, 2, 3, 1); // [1, 7, 7, 24]
        Log.d("See x2 after permute", out.shapeInfoToString());
        long[] shape = out.shape();
        long batchSize = shape[0];
        long dHeight = shape[1];
        long dWidth = shape[2];
        long dDepth = shape[3];
        long sDepth = dDepth / blockSizeSq;
        long sWidth = dWidth * blockSize;
        long sHeight = dHeight * blockSize;
        INDArray t_1 = out.reshape(batchSize, dHeight, dWidth, blockSizeSq, sDepth);
        int numOfSplits = (int)(blockSizeSq / blockSize); //
        INDArray[] spl = Utils.split(t_1, numOfSplits, 3);
        INDArray[] stack = new INDArray[numOfSplits];
        for(int i = 0; i < numOfSplits; i++) {
            INDArray t_t = spl[i];
            stack[i] = t_t.reshape(batchSize, dHeight, sWidth, sDepth);
        }
        INDArray output = Nd4j.stack(0, stack);
        output = output.transpose();
        Log.d("see stack", output.shapeInfoToString());
        output = output.permute(0, 2, 1, 3, 4).reshape(batchSize, sHeight, sWidth, sDepth); // transpose need to double check.
        output = output.permute(0, 3, 1, 2);
        return output;
    }

}