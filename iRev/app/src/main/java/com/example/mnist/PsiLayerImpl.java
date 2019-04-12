package com.example.mnist;

import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import android.util.Log;
import org.deeplearning4j.nn.workspace.ArrayType;





public class PsiLayerImpl extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {
    protected int blockSize;

    public PsiLayerImpl(NeuralNetConfiguration conf) {
        super(conf);
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
        INDArray out = workspaceMgr.create(ArrayType.ACTIVATIONS, outShape, 'c');
        out.assign(output);
        return out;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }
}