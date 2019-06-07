package com.example.cifar;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.primitives.Pair;

import org.deeplearning4j.nn.layers.AbstractLayer;
public class PsiLayerImpl extends AbstractLayer<PsiLayer> {
    protected int blockSize;

    public PsiLayerImpl(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
        this.blockSize = ((PsiLayer) conf().getLayer()).getBlockSize();
    }

    @Override
    public void clearNoiseWeightParams() {
        //No op
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false); //input (3, 16, 32, 3)
        long blockSizeSq = this.blockSize * this.blockSize;
        input = input.permute(0, 2, 3, 1);
        long[] shape = input.shape();
        long batchSize = shape[0]; // 3
        long sHeight = shape[1]; // 16
        long sWidth = shape[2]; // 32
        long sDepth = shape[3];  // 3
        long dDepth = sDepth * blockSizeSq; // new depth 14
        long dHeight = sHeight / this.blockSize; // new height 8
        int numOfSplits = (int)sWidth / this.blockSize; // 16
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
        out.assign(output); // (3, 8, 16, 12)
        return out;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }



    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        //long[] origShape = epsilon.shape();
        //Don't collapse dims case: error should be [minibatch, vectorSize, 1] or [minibatch, channels, 1, 1]
        //Reshape it to 2d, to get rid of the 1s

        //epsilon = epsilon.reshape(epsilon.ordering(), origShape[0], origShape[1], );
        INDArray epsilonNd;
        epsilonNd = inverse(epsilon, this.blockSize);
        Gradient retGradient = new DefaultGradient(); //Empty: no params
        epsilonNd = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilonNd);
        return new Pair<>(retGradient, epsilonNd);

    }

    public static INDArray inverse(INDArray in, int blockSize) {
        //in: [1, 7, 7, 24]
        //in:[1, 24, 7, 7]
        long blockSizeSq = blockSize * blockSize; //4
        INDArray out = in.permute(0, 2, 3, 1); // [1, 7, 7, 24]
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
        output = output.permute(0, 2, 1, 3, 4).reshape(batchSize, sHeight, sWidth, sDepth); // transpose need to double check.
        output = output.permute(0, 3, 1, 2);
        return output;
    }

}