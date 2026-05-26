import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# from res3dcnn import ResNet3D18
# from vivi import build_viviechoformer
# from lifes import get_lifes
from vitl import get_model_vitl
# from old_mlr import get_model_old_mlr
# from mlr import get_model_mlr
def _humanize_flops(n):
    units = ["FLOPs","KFLOPs","MFLOPs","GFLOPs","TFLOPs"]
    k = 0
    n = float(n)
    while n >= 1000.0 and k < len(units)-1:
        n /= 1000.0
        k += 1
    return f"{n:.3f} {units[k]}"

def get_flops(model: tf.keras.Model, batch_size=1):
    """
    Compute FLOPs for a Keras model (supports multi-input).
    """

    # Build TensorSpec list for all inputs
    input_signature = []
    for t in model.inputs:
        shape = list(t.shape)
        shape[0] = batch_size
        input_signature.append(tf.TensorSpec(shape, t.dtype))

    # Create a tf.function wrapper
    @tf.function
    def model_fn(*args):
        return model(args)

    # Important: pass signature as a *tuple* for multi-input models
    concrete_fn = model_fn.get_concrete_function(*input_signature)

    # Freeze to constants
    frozen_func = convert_variables_to_constants_v2(concrete_fn)
    graph_def = frozen_func.graph.as_graph_def()

    # Run TF v1 profiler
    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=graph,
            run_meta=run_meta,
            options=opts
        )

    total_flops = flops.total_float_ops if flops is not None else 0
    return total_flops, model.count_params()


# -------------------------
# Use it on your Siamese Vision Mamba model
# -------------------------

model = get_model_vitl()
print(model.summary())
flops, params = get_flops(model, batch_size=1)

print(f"FLOPs: {_humanize_flops(flops)}")
print(f"Params: {params:,}")