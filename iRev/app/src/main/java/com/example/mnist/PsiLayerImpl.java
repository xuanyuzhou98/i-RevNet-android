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
    public INDArray inverse(INDArray in, LayerWorkspaceMgr workspaceMgr) {
//        output = input.permute(0, 2, 3, 1)
//        (batch_size, d_height, d_width, d_depth) = output.size()
//        s_depth = int(d_depth / self.block_size_sq)
//        s_width = int(d_width * self.block_size)
//        s_height = int(d_height * self.block_size)
//        t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
//        spl = t_1.split(self.block_size, 3)
//        stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) for t_t in spl]
//        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch_size, s_height, s_width, s_depth)
//        output = output.permute(0, 3, 1, 2)
//        return output.contiguous()
        long blockSizeSq = this.blockSize * this.blockSize;
        INDArray out = in.permute(0, 2, 3, 1);
        long[] shape = in.shape();
        long batchSize = shape[0];
        long dHeight = shape[1];
        long dWidth = shape[2];
        long dDepth = shape[3];
        long sDepth = dDepth / blockSizeSq;
        long sWidth = dWidth * this.blockSize;
        long sHeight = dHeight * this.blockSize;
        INDArray t_1 = out.reshape(batchSize, dHeight, dWidth, blockSizeSq, sDepth);
        int numOfSplits = (int)(sDepth / this.blockSize); //
        INDArray[] spl = Utils.split(t_1, numOfSplits, 3);
        INDArray[] stack = new INDArray[numOfSplits];
        for(int i = 0; i < numOfSplits; i++) {
            INDArray t_t = spl[i];
            stack[i] = t_t.reshape(batchSize, dHeight, sWidth, blockSizeSq, sDepth);

        }
        INDArray output = Nd4j.stack(1, stack);
        output = output.transpose().permute(0, 2, 1, 3, 4).reshape(batchSize, sHeight, sWidth, sDepth); // transpose need to double check.
        output = output.permute(0, 3, 1, 2);
        long[] outShape = output.shape();
        INDArray o = workspaceMgr.create(ArrayType.ACTIVATIONS, outShape, 'c');
        o.assign(output);
        return o;


    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }
}