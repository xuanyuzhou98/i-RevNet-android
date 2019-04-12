## Backward pass flop count

### Affine layer

```{python}
def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None

  dx = dout.dot(w.T)
  dx = dx.reshape(x.shape)
  N = x.shape[0]
  D = w.shape[0]
  x_reshaped = x.reshape(N, D)
  dw = x_reshaped.T.dot(dout)
  dw = dw.reshape(w.shape)
  ones = np.ones((1, dout.shape[0]))
  db = ones.dot(dout)
  db = db[0]

  return dx, dw, db
```

There are 3 critical multiplications in the backward layer. 

(1)  `dx = dout.dot(w.T)` , where `dout` has shape `N*M`, and `w.T` has shape `M*D`. `N` is the batch size.

- Consider `N = 1`. A matrix multiply is simply a whole bunch of dot products. Each dot product has `M` elements and therefore this counts as `M` MACCs. We have to compute `D` of these dot products, and so the total number of MACCs is `M × D` the same size as the weight matrix. So there are `(2M - 1) × D` FLOPS. 

(2)  `dw = x_reshaped.T.dot(dout)` , where `x_reshaped.T` has shape `D*N` and `dout` has shape `N*M`. `N` is the batch size.

- Consider `N = 1`. If you draw out the shape of the matrix product, the result is a `D*M` matrix, so there is only 1 Flop for each resulting matrix entry. So the total number of FLOPs is `D × M`.

(3)  `db = ones.dot(dout)` , where one has shape `1*N` and `dout` has shape `N*M`. `N` is the batch size.

- `M` Flops. However, note that the calculation of db is negligible.

So the summation is `(2M - 1) × D + D × M` (We multiply batch size at the end).

### Convolution layer


```{python}
def conv_backward_strides(dout, cache):
  x, w, b, conv_param, x_cols = cache
  stride, pad = conv_param['stride'], conv_param['pad']

  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  _, _, out_h, out_w = dout.shape

  db = np.sum(dout, axis=(0, 2, 3))

  dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
  dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

  dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
  dx_cols.shape = (C, HH, WW, N, out_h, out_w)
  dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

  return dx, dw, db

```

(1)  `db = np.sum(dout, axis=(0, 2, 3))`, where `dout` has shape `N*F*outH*outW`. `N` is the batch size.

- `db` is counting the sum along axis 0, 2 and 3, which has size `N*outH*outW`. So when `N = 1` there are `outH*outW` FLOPS. 

(2) `dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)`

- `dout_reshaped` has shape `(F, N * outH * outW)`. `x_cols` has shape `(C * HH * WW, N * outH * outW)`, wehre `HH` and `WW` are filter size. As with other matrix multiplication, the FLOP count is `F * ((2 * outH * outW - 1) * (C * HH * WW))`.

(3) `dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)`

- `w.reshape(F,-1)` has size `(F, C * HH * WW)` and `dout_reshaped` has shape `(F, N * outH * outW)`. So FLOP count is `(C * HH * WW) * ((2 * F - 1) * N * outH * outW)`.

(4) `dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)`

```{python}
abcdcdef col2im_6d_cython_inner(np.ndarray[DTYPE_t, ndim=6] cols,
                            np.ndarray[DTYPE_t, ndim=4] x_padded,
                            int N, int C, int H, int W, int HH, int WW,
                            int out_h, int out_w, int pad, int stride):

    cdef int c, hh, ww, n, h, w
    for n in range(N):
        for c in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    for h in range(out_h):
                        for w in range(out_w):
                            x_padded[n, c, stride * h + hh, stride * w + ww] += cols[c, hh, ww, n, h, w]
    

def col2im_6d_cython(np.ndarray[DTYPE_t, ndim=6] cols, int N, int C, int H, int W,
        int HH, int WW, int pad, int stride):
    cdef np.ndarray x = np.empty((N, C, H, W), dtype=cols.dtype)
    cdef int out_h = (H + 2 * pad - HH) / stride + 1
    cdef int out_w = (W + 2 * pad - WW) / stride + 1
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad),
                                                  dtype=cols.dtype)

    col2im_6d_cython_inner(cols, x_padded, N, C, H, W, HH, WW, out_h, out_w, pad, stride)

    if pad > 0:
        return x_padded[:, :, pad:-pad, pad:-pad]
    return x_padded 

```

- So the Flop count is `N * C * HH * WW * outH * outW`.



